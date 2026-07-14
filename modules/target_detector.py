from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Target:
    """Represents a detected target."""

    target_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


class TargetDetector:
    """Placeholder target detector specialized for target-like objects."""

    def __init__(self) -> None:
        self.targets: list[Target] = []

    def detect(self, detections: list[Any]) -> list[Target]:
        """Filter detections to target-like classes."""
        targets: list[Target] = []
        for idx, detection in enumerate(detections):
            if detection.label.lower() in {"person", "target", "bullseye", "shooting target"}:
                targets.append(Target(target_id=idx + 1, bbox=detection.bbox, confidence=detection.confidence))
        return targets
