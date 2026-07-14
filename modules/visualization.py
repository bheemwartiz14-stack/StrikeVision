from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class VisualizationBuilder:
    """Build Plotly charts for the dashboard."""

    def build_score_chart(self, shot_table: pd.DataFrame) -> go.Figure:
        """Render shot score timeline."""
        if shot_table.empty:
            return px.bar(title="No shots detected")
        fig = px.line(shot_table, x="shot_id", y="score", color_discrete_sequence=["#38bdf8"])
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def build_accuracy_chart(self, target_table: pd.DataFrame) -> go.Figure:
        """Render target accuracy chart."""
        if target_table.empty:
            return px.bar(title="No target data")
        fig = px.bar(target_table, x="target_id", y="accuracy", color="accuracy", color_continuous_scale="Viridis")
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def build_hit_distribution(self, shot_table: pd.DataFrame) -> go.Figure:
        """Render hit distribution."""
        if shot_table.empty:
            return px.pie(values=[1], names=["No Data"])
        counts = shot_table["hit"].value_counts()
        fig = px.pie(names=["Hit" if k else "Miss" for k in counts.index], values=counts.values, color_discrete_sequence=["#22c55e", "#ef4444"])
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        return fig
