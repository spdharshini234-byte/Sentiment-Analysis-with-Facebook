"""Rule-based insights built from computed statistics."""
from __future__ import annotations

import pandas as pd

from src.analysis.insights import generate_insights


def build(df: pd.DataFrame, evaluation: dict | None, data_report: dict | None) -> list[dict]:
    return generate_insights(df, evaluation, data_report)
