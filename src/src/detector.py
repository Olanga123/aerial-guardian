"""
detector.py — YOLOv8n detector with SAHI and CLAHE preprocessing.

Key adaptations for drone footage:
- SAHI slicing: breaks large frames into overlapping tiles so small
  (10-40px) persons are detected reliably.
- CLAHE: enhances contrast in low-visibility aerial conditions.
- conf=0.25: tuned lower than default for VisDrone small-object distribution.
"""

import numpy as np
import cv2
from ultralytics import YOLO


class DroneDetector:
    def __init__(self, det_cfg, sahi_cfg):
        self.conf = det_cfg.get("confidence", 0.25)
        self.iou = det_cfg.get("iou", 0.45)
        self.device = det_cfg.get("device", "cpu")
        self.imgsz = det_cfg.get("imgsz", 640)
        self.target_class = det_cfg.get("target_class", 0)

        self.sahi_enabled = sahi_cfg.get("enabled", True)
        self.slice_h = sahi_cfg.get("slice_height", 640)
        self.slice_w = sahi_cfg.get("slice_width", 640)
        self.overlap = sahi_cfg.get("overlap_ratio", 0.2)

        print(f"  Loading detector: {det_cfg.get('model', 'yolov8n.pt')} on {self.device}")
        self.model = YOLO(det_cfg.get("model", "yolov8n.pt"))
        self.model.to(self.device)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame):
        """Run detection. Returns np.ndarray (N, 5): [x1, y1, x2, y2, conf]"""
        enhanced = self._preprocess(frame)
        if self.sahi_enabled:
            return self._detect_sahi(enhanced, frame.shape)
        return self._detect_full(enhanced)

    def _preprocess(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

    def _detect_full(self, frame):
        results = self.model(
            frame, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, classes=[self.target_class], verbose=False
        )
        return self._parse_results(results[0])

    def _detect_sahi(self, frame, original_shape):
        H, W = frame.shape[:2]
        stride_h = int(self.slice_h * (1 - self.overlap))
        stride_w = int(self.slice_w * (1 - self.overlap))
        all_dets = []

        for y in range(0, H, stride_h):
            for x in range(0, W, stride_w):
                y2 = min(y + self.slice_h, H)
                x2 = min(x + self.slice_w, W)
                y1t, x1t = max(0, y2 - self.slice_h), max(0, x2 - self.slice_w)
                tile = frame[y1t:y2, x1t:x2]

                results = self.model(
                    tile, conf=self.conf, iou=self.iou,
                    imgsz=self.imgsz, classes=[self.target_class], verbose=False
                )
                tile_dets = self._parse_results(results[0])

                if len(tile_dets) > 0:
                    tile_dets[:, 0] += x1t
                    tile_dets[:, 2] += x1t
                    tile_dets[:, 1] += y1t
                    tile_dets[:, 3] += y1t
                    all_dets.append(tile_dets)

        if not all_dets:
            return np.empty((0, 5), dtype=np.float32)

        return self._nms(np.vstack(all_dets), iou_threshold=0.5)

    def _parse_results(self, result):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, confs]).astype(np.float32)

    @staticmethod
    def _nms(dets, iou_threshold=0.5):
        if len(dets) == 0:
            return dets
        boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in dets[:, :4]]
        scores = dets[:, 4].tolist()
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, iou_threshold)
        if len(indices) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return dets[indices.flatten()]
