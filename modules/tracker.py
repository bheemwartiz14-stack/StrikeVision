from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""

    object_id: int
    label: str
    bbox: tuple[float, float, float, float]
    history: list[tuple[float, float, float, float]] = field(default_factory=list)
    last_seen: int = 0


class Tracker:
    """A lightweight tracker for maintaining object continuity."""

    def __init__(self) -> None:
        self.objects: dict[int, TrackedObject] = {}
        self.next_id = 1

    def update(self, detections: list[Any], frame_idx: int) -> list[TrackedObject]:
        """Assign IDs to detections and update state."""
        tracked: list[TrackedObject] = []
        for detection in detections:
            obj_id = self.next_id
            self.next_id += 1
            tracked_obj = TrackedObject(
                object_id=obj_id,
                label=detection.label,
                bbox=detection.bbox,
                history=[detection.bbox],
                last_seen=frame_idx,
            )
            self.objects[obj_id] = tracked_obj
            tracked.append(tracked_obj)
        return tracked
