from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class BulletImpact:
    """Represents a bullet impact detection."""

    frame_idx: int
    center: tuple[float, float]
    confidence: float


class BulletDetector:
    """Simple bullet impact detector based on motion/contrast changes."""

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def detect(self, frame: np.ndarray, prev_frame: np.ndarray | None, frame_idx: int) -> list[BulletImpact]:
        """Detect candidate bullet impacts using frame differencing."""
        if prev_frame is None:
            return []
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_current, gray_prev)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        impacts: list[BulletImpact] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w / 2.0, y + h / 2.0)
            confidence = min(0.99, area / 1000.0)
            impacts.append(BulletImpact(frame_idx=frame_idx, center=center, confidence=confidence))
        return impacts
