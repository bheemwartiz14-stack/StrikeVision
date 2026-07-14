from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fpdf import FPDF

from modules.utils import ensure_dir


class ReportGenerator:
    """Generate reports in PDF, CSV, and JSON formats."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.report_dir = ensure_dir(run_dir / "reports")

    def generate(self, video_path: Path, summary: dict[str, Any], shot_table: pd.DataFrame, target_table: pd.DataFrame) -> dict[str, Path]:
        """Generate all reports and return the paths."""
        report_csv = self.report_dir / "report.csv"
        report_json = self.report_dir / "report.json"
        report_pdf = self.report_dir / "report.pdf"

        shot_table.to_csv(report_csv, index=False)
        report_json.write_text(json.dumps({"summary": summary, "shots": shot_table.to_dict(orient="records"), "targets": target_table.to_dict(orient="records")}, indent=2), encoding="utf-8")

        self._generate_pdf(report_pdf, video_path, summary, shot_table, target_table)
        return {"csv": report_csv, "json": report_json, "pdf": report_pdf}

    def _generate_pdf(self, path: Path, video_path: Path, summary: dict[str, Any], shot_table: pd.DataFrame, target_table: pd.DataFrame) -> None:
        """Create a PDF report with key metrics."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AI Shooting Range Scoring Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Video: {video_path.name}", ln=True)
        pdf.cell(0, 8, f"Shots: {summary.get('total_shots', 0)}", ln=True)
        pdf.cell(0, 8, f"Targets: {summary.get('total_targets', 0)}", ln=True)
        pdf.cell(0, 8, f"Accuracy: {summary.get('accuracy', 0)}%", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Shot Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for _, row in shot_table.head(10).iterrows():
            pdf.cell(0, 6, f"Shot {int(row['shot_id'])}: hit={bool(row['hit'])}, score={row['score']}", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Target Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for _, row in target_table.head(10).iterrows():
            pdf.cell(0, 8, f"Target {int(row['target_id'])}: hits={int(row['hits'])}, accuracy={row['accuracy']}%", ln=True)
        pdf.output(str(path))
