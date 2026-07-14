from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from config import MODEL_PATH


@dataclass
class Detection:
    """Represents a single detection."""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    label: str


class Detector:
    """Wrapper around YOLO detection model."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or MODEL_PATH
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Load the YOLO model or fall back to CPU."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        return YOLO(str(self.model_path))

    def predict(self, frame: np.ndarray, conf: float = 0.45) -> list[Detection]:
        """Run inference on a frame."""
        results = self.model(frame, stream=False, conf=conf, imgsz=640, half=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        class_id=cls_id,
                        confidence=conf_val,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        label=label,
                    )
                )
        return detections
