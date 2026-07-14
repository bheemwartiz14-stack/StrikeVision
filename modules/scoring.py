from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ShotResult:
    """Stores scoring outcome for each shot."""

    shot_id: int
    frame_idx: int
    target_id: int
    impact_x: float
    impact_y: float
    score: float
    hit: bool
    confidence: float


class ScoringEngine:
    """Calculate hits and scores for detected impacts."""

    def __init__(self) -> None:
        self.shots: list[ShotResult] = []

    def score_hits(self, impacts: list[Any], targets: list[Any], frame_idx: int) -> list[ShotResult]:
        """Associate impacts with nearest targets and calculate score."""
        results: list[ShotResult] = []
        for shot_idx, impact in enumerate(impacts, start=1):
            best_target = None
            best_distance = float("inf")
            for target in targets:
                x1, y1, x2, y2 = target.bbox
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                dist = hypot(center_x - impact.center[0], center_y - impact.center[1])
                if dist < best_distance:
                    best_distance = dist
                    best_target = target
            hit = best_target is not None and best_distance < 120.0
            score = 10.0 if hit else 0.0
            result = ShotResult(
                shot_id=len(self.shots) + 1,
                frame_idx=frame_idx,
                target_id=best_target.target_id if best_target else 0,
                impact_x=impact.center[0],
                impact_y=impact.center[1],
                score=score,
                hit=hit,
                confidence=impact.confidence,
            )
            self.shots.append(result)
            results.append(result)
        return results

    def build_score_table(self) -> pd.DataFrame:
        """Create a DataFrame summary for the scores."""
        return pd.DataFrame(
            [
                {
                    "shot_id": shot.shot_id,
                    "frame": shot.frame_idx,
                    "target_id": shot.target_id,
                    "impact_x": round(shot.impact_x, 2),
                    "impact_y": round(shot.impact_y, 2),
                    "score": shot.score,
                    "hit": shot.hit,
                    "confidence": round(shot.confidence, 3),
                }
                for shot in self.shots
            ]
        )

    def build_target_summary(self) -> pd.DataFrame:
        """Create a target summary table."""
        if not self.shots:
            return pd.DataFrame(columns=["target_id", "hits", "shots", "accuracy"])
        records: list[dict[str, Any]] = []
        grouped: dict[int, list[ShotResult]] = {}
        for shot in self.shots:
            grouped.setdefault(shot.target_id, []).append(shot)
        for target_id, shots in grouped.items():
            hits = sum(1 for shot in shots if shot.hit)
            accuracy = round((hits / len(shots)) * 100, 2) if shots else 0.0
            records.append({"target_id": target_id, "hits": hits, "shots": len(shots), "accuracy": accuracy})
        return pd.DataFrame(records)
