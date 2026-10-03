"""Create the SQLite schema and fill it from the prepared Kaggle data scored by the trained model."""
from __future__ import annotations

import sqlite3

import pandas as pd

from backend.app.database.database import connect, db_path
from src import config
from src.models.predict import SentimentPredictor

SCHEMA = """
DROP TABLE IF EXISTS posts;
CREATE TABLE posts (
    post_id INTEGER, timestamp TEXT, text TEXT NOT NULL, clean_text TEXT, category TEXT,
    sentiment TEXT NOT NULL, language TEXT, has_link INTEGER, likes INTEGER, comments INTEGER,
    shares INTEGER, is_spam INTEGER, split TEXT,
    pred_label TEXT, pred_conf REAL, p_negative REAL, p_neutral REAL, p_positive REAL
);
CREATE INDEX IF NOT EXISTS idx_posts_sentiment ON posts(sentiment);
CREATE INDEX IF NOT EXISTS idx_posts_time ON posts(timestamp);
CREATE TABLE IF NOT EXISTS predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, text TEXT, sentiment TEXT, confidence REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

POST_COLUMNS = ["post_id", "timestamp", "text", "clean_text", "category", "sentiment", "language", "has_link",
                "likes", "comments", "shares", "is_spam", "split", "pred_label", "pred_conf",
                "p_negative", "p_neutral", "p_positive"]


def init_schema() -> None:
    db_path().parent.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.executescript(SCHEMA)


def build_database(prepared_csv=None, predictor: SentimentPredictor | None = None) -> int:
    """Score every prepared post with the saved model and store it. Returns the number of rows."""
    src = prepared_csv or (config.DATA_PROCESSED / "prepared_dataset.csv")
    if not src.is_file():
        raise FileNotFoundError("data/processed/prepared_dataset.csv not found. Run `python scripts/train_model.py` first.")
    predictor = predictor or SentimentPredictor()
    df = pd.read_csv(src)
    for col in POST_COLUMNS:
        if col not in df.columns and not col.startswith(("pred", "p_")):
            df[col] = None
    preds = predictor.predict_many(df["text"].tolist())
    df["pred_label"] = [p["sentiment"] for p in preds]
    df["pred_conf"] = [p["confidence"] for p in preds]
    for lab in ("negative", "neutral", "positive"):
        df[f"p_{lab}"] = [p["probabilities"].get(lab, 0.0) for p in preds]
    init_schema()
    with connect() as c:
        df[POST_COLUMNS].to_sql("posts", c, if_exists="append", index=False)
        c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('model_name', ?)", (predictor.model_name,))
        c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('rows', ?)", (str(len(df)),))
    return int(len(df))
