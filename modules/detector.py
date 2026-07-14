from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
)
@dataclass
class Detection:
    """Represents a single detection."""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    label: str


class Detector:
    """Wrapper around the YOLO detection model."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        self.model = self._load_model()

    def _download_model(self) -> None:
        """Download the YOLO model if it is missing."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading model to {self.model_path}...")
        urlretrieve(MODEL_URL, self.model_path)
        print("Model download completed.")

    def _load_model(self) -> YOLO:
        """
        Load the YOLO model.

        If the model file doesn't exist, download it automatically.
        """

        if not self.model_path.exists():
            self._download_model()

        return YOLO(str(self.model_path))

    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.45,
    ) -> list[Detection]:
        """Run inference on a frame."""

        results = self.model(
            frame,
            conf=conf,
            imgsz=640,
            stream=False,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidence = float(box.conf[0].cpu().item())
                class_id = int(box.cls[0].cpu().item())

                label = self.model.names.get(
                    class_id,
                    f"class_{class_id}",
                )

                detections.append(
                    Detection(
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ),
                        label=label,
                    )
                )

        return detections