"""utils.py — Shared utility functions."""

import os
import cv2


def get_video_info(cap):
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
