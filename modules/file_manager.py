from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from config import OUTPUT_DIR
from modules.utils import ensure_dir, format_bytes, timestamp


class OutputManager:
    """Manage output folders and exports."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or OUTPUT_DIR
        ensure_dir(self.base_dir)

    def create_run_folder(self, name: str | None = None) -> Path:
        """Create a new run folder."""
        run_name = name or f"run_{timestamp()}"
        target_dir = self.base_dir / run_name
        ensure_dir(target_dir)
        ensure_dir(target_dir / "reports")
        ensure_dir(target_dir / "logs")
        ensure_dir(target_dir / "images")
        ensure_dir(target_dir / "targets")
        return target_dir

    def list_runs(self) -> list[dict[str, Any]]:
        """List available output runs."""
        runs: list[dict[str, Any]] = []
        if not self.base_dir.exists():
            return runs
        for path in sorted(self.base_dir.iterdir(), key=lambda p: p.name):
            if path.is_dir():
                files = list(path.rglob("*"))
                runs.append(
                    {
                        "name": path.name,
                        "path": path,
                        "file_count": len([f for f in files if f.is_file()]),
                        "size": sum(f.stat().st_size for f in files if f.is_file()),
                        "created": path.stat().st_mtime,
                    }
                )
        return runs

    def save_json(self, path: Path, data: Any) -> None:
        """Persist JSON data."""
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_csv(self, path: Path, dataframe: pd.DataFrame) -> None:
        """Persist CSV data."""
        dataframe.to_csv(path, index=False)

    def delete_run(self, run_dir: Path) -> None:
        """Delete a run directory."""
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def get_folder_summary(self, path: Path) -> dict[str, Any]:
        """Summarize a folder."""
        files = [f for f in path.rglob("*") if f.is_file()]
        return {
            "folder_size": format_bytes(sum(f.stat().st_size for f in files)),
            "file_count": len(files),
            "created_date": path.stat().st_mtime,
        }
