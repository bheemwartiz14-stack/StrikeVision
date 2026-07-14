from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"


@dataclass
class ProcessingSettings:
    frame_skip: int = 2
    confidence_threshold: float = 0.45
    max_targets: int = 8
    hit_radius: int = 18
    use_gpu: bool = False
    enable_heatmap: bool = True
    save_frames: bool = True


DEFAULT_SETTINGS = ProcessingSettings()


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MODEL_PATH = MODELS_DIR / "yolo11n.pt"
LOG_LEVEL = os.getenv("SHOOTING_AI_LOG_LEVEL", "INFO")
