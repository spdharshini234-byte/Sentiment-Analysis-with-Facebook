"""Central configuration. All paths are built with pathlib (no OS-specific paths)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FSI_ROOT", Path(__file__).resolve().parents[1])).resolve()

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PHASES_DIR = REPORTS_DIR / "phases"

DEFAULT_DATASET = DATA_RAW / "Facebook_data_txt.csv"
DB_PATH = Path(os.environ.get("FSI_DB_PATH", DATA_PROCESSED / "sentiment.db"))

PIPELINE_PATH = MODELS_DIR / "sentiment_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
LABEL_MAP_PATH = MODELS_DIR / "label_mapping.json"

DATA_REPORT_PATH = REPORTS_DIR / "data_report.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"
EVALUATION_PATH = REPORTS_DIR / "evaluation.json"

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]

# Confidence buckets (probability of the predicted class)
CONF_HIGH = 0.70
CONF_MEDIUM = 0.50


def find_dataset() -> Path | None:
    """Locate the Kaggle CSV. Order: FSI_DATA_PATH env var, default file, any CSV in data/raw."""
    env = os.environ.get("FSI_DATA_PATH")
    if env and Path(env).is_file():
        return Path(env)
    if DEFAULT_DATASET.is_file():
        return DEFAULT_DATASET
    csvs = sorted(DATA_RAW.glob("*.csv")) if DATA_RAW.is_dir() else []
    return csvs[0] if csvs else None
