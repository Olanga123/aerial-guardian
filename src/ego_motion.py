"""
ego_motion.py — Camera ego-motion compensation using Farneback optical flow.

Drone camera movement causes the entire scene to translate between frames,
confusing the tracker. This module estimates and compensates for it.
"""

import cv2
import numpy as np


class EgoMotionCompensator:
    """
    Estimates camera ego-motion and compensates detection positions.

    1. Compute dense Farneback optical flow between frames.
    2. Take median flow as global camera translation (robust to foreground).
    3. Subtract translation from detection boxes before tracking.
    """

    def __init__(self):
        self._smoothed_dx = 0.0
        self._smoothed_dy = 0.0
        self._alpha = 0.7  # EMA smoothing factor

    def estimate(self, prev_gray, curr_gray):
        """Returns (dx, dy) camera shift. Returns (0,0) on first frame."""
        if prev_gray is None:
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Median is robust against foreground object motion
        dx = float(np.median(flow[..., 0]))
        dy = float(np.median(flow[..., 1]))

        # Smooth to reduce jitter
        self._smoothed_dx = self._alpha * dx + (1 - self._alpha) * self._smoothed_dx
        self._smoothed_dy = self._alpha * dy + (1 - self._alpha) * self._smoothed_dy

        return self._smoothed_dx, self._smoothed_dy

    def compensate_detections(self, detections, cam_shift):
        """Shift detection boxes by negative camera translation."""
        if len(detections) == 0:
            return detections
        dx, dy = cam_shift
        compensated = detections.copy()
        compensated[:, 0] -= dx  # x1
        compensated[:, 2] -= dx  # x2
        compensated[:, 1] -= dy  # y1
        compensated[:, 3] -= dy  # y2
        return compensated
