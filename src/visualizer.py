"""
visualizer.py — Draws bounding boxes, track IDs, and trajectory tails.
"""

import cv2
import numpy as np
from collections import defaultdict, deque


def _id_to_color(track_id):
    palette = [
        (86, 255, 86), (86, 86, 255), (255, 200, 50), (255, 86, 200),
        (50, 220, 255), (200, 50, 255), (255, 140, 50), (50, 255, 200),
        (180, 90, 255), (255, 255, 100),
    ]
    return palette[track_id % len(palette)]


class TrackVisualizer:
    def __init__(self, cfg):
        self.tail_length  = cfg.get("tail_length", 30)
        self.show_fps     = cfg.get("show_fps", True)
        self.box_thickness = cfg.get("box_thickness", 2)
        self.font_scale   = cfg.get("font_scale", 0.6)
        self._tails = defaultdict(lambda: deque(maxlen=self.tail_length))

    def draw(self, frame, tracks, fps, frame_idx):
        vis = frame.copy()

        for track in tracks:
            x1, y1, x2, y2, tid, conf = track
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = _id_to_color(tid)

            self._tails[tid].append((cx, cy))
            pts = list(self._tails[tid])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(vis, pts[i-1], pts[i], faded, max(1, int(3*alpha)), cv2.LINE_AA)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

            label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            lx, ly = x1, max(y1 - 4, th + 4)
            cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + 2), color, -1)
            cv2.putText(vis, label, (lx+2, ly-2), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        if self.show_fps:
            text = f"FPS: {fps:.1f}  |  Frame: {frame_idx}  |  Tracks: {len(tracks)}"
            cv2.rectangle(vis, (0, 0), (len(text)*9, 28), (0, 0, 0), -1)
            cv2.putText(vis, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return vis
