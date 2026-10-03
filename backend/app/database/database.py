"""SQLite helpers. The DB path comes from src.config (FSI_DB_PATH env var overrides it)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src import config


def db_path() -> Path:
    return Path(config.DB_PATH)


@contextmanager
def connect():
    conn = sqlite3.connect(str(db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def database_ready() -> bool:
    p = db_path()
    if not p.is_file():
        return False
    try:
        with connect() as c:
            n = c.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        return n > 0
    except sqlite3.Error:
        return False


def load_posts_df() -> pd.DataFrame:
    with connect() as c:
        df = pd.read_sql_query("SELECT * FROM posts", c)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def log_prediction(source: str, text: str, sentiment: str, confidence: float) -> None:
    """Best-effort logging. Never lets a logging problem break a prediction."""
    try:
        with connect() as c:
            c.execute(
                "INSERT INTO predictions_log (created_at, source, text, sentiment, confidence) "
                "VALUES (datetime('now'), ?, ?, ?, ?)",
                (source, text[:500], sentiment, float(confidence)),
            )
    except Exception:
        pass
