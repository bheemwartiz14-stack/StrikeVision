from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import cv2


def configure_logging(log_dir: Path | None = None) -> logging.Logger:
    """Configure application logging."""
    log_dir = log_dir or Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shooting_ai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_system_info() -> dict[str, Any]:
    """Collect lightweight system information."""
    disk_usage = shutil.disk_usage("/")
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "memory_gb": round(disk_usage.total / (1024**3), 2),
    }


def safe_int(value: Any, default: int = 0) -> int:
    """Safely cast to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_bytes(size: int) -> str:
    """Format file size in a human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def timestamp() -> str:
    """Return a timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def estimate_eta(start_time: float, processed: int, total: int) -> str:
    """Estimate ETA in seconds."""
    if processed <= 0 or total <= 0:
        return "--:--"
    elapsed = time.time() - start_time
    remaining = max(0, int((elapsed / processed) * (total - processed)))
    minutes, seconds = divmod(remaining, 60)
    return f"{minutes:02d}:{seconds:02d}"


def resize_frame(frame: np.ndarray, width: int = 1280) -> np.ndarray:
    """Resize a frame while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    ratio = width / w
    new_h = max(1, int(h * ratio))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
