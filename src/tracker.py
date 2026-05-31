"""
tracker.py — Self-contained ByteTrack implementation.

ByteTrack's two-round matching is critical for drone MOT:
- Round 1: High-confidence detections matched to active tracks (IoU).
- Round 2: Low-confidence detections rescue lost tracks (occlusions/small persons).
"""

import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanTrack:
    """Single-object track with Kalman filter state estimation."""
    _count = 0

    def __init__(self, bbox, conf):
        KalmanTrack._count += 1
        self.track_id = KalmanTrack._count
        self.conf = conf
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"
        self.history = deque(maxlen=60)

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.eye(4, 8)
        self.kf.R *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._xyxy_to_state(bbox)

    @staticmethod
    def _xyxy_to_state(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([[cx], [cy], [w / (h + 1e-6)], [h]])

    def predict(self):
        if self.kf.x[6] + self.kf.x[3] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, conf):
        self.kf.update(self._xyxy_to_state(bbox))
        self.hits += 1
        self.time_since_update = 0
        self.conf = conf
        if self.hits >= 3:
            self.state = "confirmed"
        self.history.append(self.get_bbox())

    def get_bbox(self):
        cx, cy, asp, h = self.kf.x[:4].flatten()
        w = asp * h
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_t = (bb_test[..., 2]-bb_test[..., 0]) * (bb_test[..., 3]-bb_test[..., 1])
    area_g = (bb_gt[..., 2]-bb_gt[..., 0])   * (bb_gt[..., 3]-bb_gt[..., 1])
    return inter / (area_t + area_g - inter + 1e-6)


def hungarian_match(iou_matrix, threshold):
    if iou_matrix.size == 0:
        return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= threshold]
    matched_r = {m[0] for m in matched}
    matched_c = {m[1] for m in matched}
    unmatched_trk = [r for r in range(iou_matrix.shape[0]) if r not in matched_r]
    unmatched_det = [c for c in range(iou_matrix.shape[1]) if c not in matched_c]
    return matched, unmatched_trk, unmatched_det


class ByteTracker:
    def __init__(self, cfg):
        self.track_thresh = cfg.get("track_thresh", 0.5)
        self.match_thresh = cfg.get("match_thresh", 0.8)
        self.low_thresh   = cfg.get("low_thresh", 0.1)
        self.track_buffer = cfg.get("track_buffer", 30)
        self.active_tracks = []
        self.lost_tracks   = []

    def update(self, detections, frame):
        """Update tracker. Returns (M, 6): [x1, y1, x2, y2, track_id, conf]"""
        if len(detections) == 0:
            for trk in self.active_tracks:
                trk.predict()
                trk.time_since_update += 1
            self._manage_lost()
            return self._output()

        scores = detections[:, 4]
        high_mask = scores >= self.track_thresh
        low_mask  = (scores >= self.low_thresh) & ~high_mask
        high_dets = detections[high_mask]
        low_dets  = detections[low_mask]

        pred_boxes = np.array([t.predict() for t in self.active_tracks]) \
            if self.active_tracks else np.empty((0, 4))

        # Round 1: high-conf dets vs active tracks
        matched1, unmatched_trk1, unmatched_det1 = [], list(range(len(self.active_tracks))), list(range(len(high_dets)))
        if len(self.active_tracks) > 0 and len(high_dets) > 0:
            iou_mat = iou_batch(pred_boxes, high_dets[:, :4])
            matched1, unmatched_trk1, unmatched_det1 = hungarian_match(iou_mat, self.match_thresh)

        for ti, di in matched1:
            self.active_tracks[ti].update(high_dets[di, :4], high_dets[di, 4])

        # Round 2: low-conf dets rescue lost/unmatched tracks
        candidate_tracks = [self.active_tracks[i] for i in unmatched_trk1] + self.lost_tracks
        if len(candidate_tracks) > 0 and len(low_dets) > 0:
            cand_boxes = np.array([t.get_bbox() for t in candidate_tracks])
            iou_mat2 = iou_batch(cand_boxes, low_dets[:, :4])
            matched2, _, _ = hungarian_match(iou_mat2, 0.5)
            for ti, di in matched2:
                candidate_tracks[ti].update(low_dets[di, :4], low_dets[di, 4])

        # New tracks for unmatched high-conf dets
        for di in unmatched_det1:
            if high_dets[di, 4] >= self.track_thresh:
                self.active_tracks.append(KalmanTrack(high_dets[di, :4], high_dets[di, 4]))

        self._manage_lost()
        return self._output()

    def _manage_lost(self):
        still_active = [t for t in self.active_tracks if t.time_since_update <= 1]
        newly_lost   = [t for t in self.active_tracks if t.time_since_update > 1]
        self.active_tracks = still_active
        self.lost_tracks.extend(newly_lost)
        recovered = [t for t in self.lost_tracks if t.time_since_update == 0]
        self.active_tracks.extend(recovered)
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update > 0 and t.time_since_update <= self.track_buffer
        ]

    def _output(self):
        out = []
        for t in self.active_tracks:
            if t.state == "confirmed":
                out.append([*t.get_bbox(), t.track_id, t.conf])
        return np.array(out, dtype=np.float32) if out else np.empty((0, 6), dtype=np.float32)
