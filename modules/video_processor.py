from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DEFAULT_SETTINGS
from modules.bullet_detector import BulletDetector, BulletImpact
from modules.detector import Detection, Detector
from modules.scoring import ScoringEngine, ShotResult
from modules.target_detector import TargetDetector
from modules.tracker import Tracker
from modules.utils import ensure_dir, resize_frame


@dataclass
class AnalysisResult:
    """Container for processing outputs."""

    run_dir: Path
    video_path: Path
    processed_path: Path
    report_dir: Path
    shot_table: pd.DataFrame
    target_table: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    frame_count: int = 0
    processed_frames: int = 0
    total_targets: int = 0
    total_shots: int = 0
    accuracy: float = 0.0


class VideoProcessor:
    """Process video files and generate analysis outputs."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or DEFAULT_SETTINGS
        self.detector = Detector()
        self.tracker = Tracker()
        self.bullet_detector = BulletDetector()
        self.target_detector = TargetDetector()
        self.scoring_engine = ScoringEngine()

    def process_video(self, video_path: Path, run_dir: Path, progress_callback: Any | None = None) -> AnalysisResult:
        """Process an MP4/video file end to end."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        ensure_dir(run_dir)
        report_dir = run_dir / "reports"
        ensure_dir(report_dir)
        output_path = run_dir / "processed.mp4"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        prev_frame: np.ndarray | None = None
        annotated_frames = 0
        shot_hits: list[ShotResult] = []
        frame_count = 0
        start_time = time.time()

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1
                if frame_count % (self.settings.frame_skip + 1) != 0:
                    continue

                detections = self.detector.predict(frame, conf=self.settings.confidence_threshold)
                targets = self.target_detector.detect(detections)
                tracked = self.tracker.update(detections, frame_count)
                impacts = self.bullet_detector.detect(frame, prev_frame, frame_count)
                if impacts:
                    shot_hits.extend(self.scoring_engine.score_hits(impacts, targets, frame_count))

                annotated = self._annotate_frame(frame, detections, targets, impacts, tracked)
                writer.write(annotated)
                annotated_frames += 1
                prev_frame = frame

                if progress_callback:
                    progress_callback(frame_count, total_frames, time.time() - start_time)
                pbar.update(1)

        cap.release()
        writer.release()

        shot_table = self.scoring_engine.build_score_table()
        target_table = self.scoring_engine.build_target_summary()
        hits = int((shot_table["hit"] == True).sum()) if not shot_table.empty else 0
        total_shots = len(shot_table)
        accuracy = round((hits / total_shots) * 100, 2) if total_shots else 0.0
        summary = {
            "video_path": str(video_path),
            "processed_frames": annotated_frames,
            "frame_count": frame_count,
            "total_targets": len(target_table),
            "total_shots": total_shots,
            "hits": hits,
            "accuracy": accuracy,
        }

        return AnalysisResult(
            run_dir=run_dir,
            video_path=video_path,
            processed_path=output_path,
            report_dir=report_dir,
            shot_table=shot_table,
            target_table=target_table,
            summary=summary,
            frame_count=frame_count,
            processed_frames=annotated_frames,
            total_targets=len(target_table),
            total_shots=total_shots,
            accuracy=accuracy,
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        targets: list[Any],
        impacts: list[BulletImpact],
        tracked: list[Any],
    ) -> np.ndarray:
        """Annotate frame with bounding boxes and hits."""
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(annotated, detection.label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        for target in targets:
            x1, y1, x2, y2 = target.bbox
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"Target {target.target_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for impact in impacts:
            cx, cy = impact.center
            cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), 2)
            cv2.putText(annotated, "Impact", (int(cx) + 8, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        return annotated
