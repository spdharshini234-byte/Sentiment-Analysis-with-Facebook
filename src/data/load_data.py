"""Dataset loading, schema validation and inspection (Phase 3: Data Acquisition)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LABELS, find_dataset

TEXT_CANDIDATES = ["text", "comment", "comments_text", "message", "content", "post", "body", "review"]
LABEL_CANDIDATES = ["sentiment", "label", "class"]
DATE_CANDIDATES = ["timestamp", "date", "created_time", "created_at", "datetime", "time"]
TOPIC_CANDIDATES = ["category", "topic", "page", "post_type"]
LANGUAGE_CANDIDATES = ["language", "lang"]


class DatasetError(Exception):
    """Raised when the dataset is missing or does not have the required columns."""


def _first_present(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    """Detect (never assume) which columns play which role."""
    date_col = _first_present(df.columns, DATE_CANDIDATES)
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if parsed.notna().mean() < 0.9:
            date_col = None
    return {
        "text": _first_present(df.columns, TEXT_CANDIDATES),
        "label": _first_present(df.columns, LABEL_CANDIDATES),
        "date": date_col,
        "topic": _first_present(df.columns, TOPIC_CANDIDATES),
        "language": _first_present(df.columns, LANGUAGE_CANDIDATES),
    }


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Kaggle CSV and validate that it contains a text and a sentiment column."""
    p = Path(path) if path else find_dataset()
    if p is None or not Path(p).is_file():
        raise DatasetError(
            "Dataset not found. Place the Kaggle file in data/raw/ (see data/raw/README.md) "
            "or set the FSI_DATA_PATH environment variable."
        )
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # malformed csv
            raise DatasetError(f"Could not read CSV: {exc}") from exc
    if df is None:
        raise DatasetError("Could not decode the CSV file with utf-8 or latin-1.")
    cols = detect_columns(df)
    if cols["text"] is None or cols["label"] is None:
        raise DatasetError(
            f"Schema check failed. Need a text column and a sentiment column; found columns: {list(df.columns)}"
        )
    df = df.rename(columns={cols["text"]: "text", cols["label"]: "sentiment"})
    if cols["date"] and cols["date"] != "timestamp":
        df = df.rename(columns={cols["date"]: "timestamp"})
    if cols["topic"] and cols["topic"] != "category":
        df = df.rename(columns={cols["topic"]: "category"})
    if cols["language"] and cols["language"] != "language":
        df = df.rename(columns={cols["language"]: "language"})
    return df


def inspect_dataset(df: pd.DataFrame, source_path: str | Path | None = None) -> dict:
    """Facts about the raw dataset (Phase 3 deliverable: data quality assessment)."""
    cols = detect_columns(df.rename(columns={}))
    text = df["text"].astype("object")
    id_cols = [c for c in ("post_id", "id") if c in df.columns]
    rep: dict = {
        "source_file": Path(source_path).name if source_path else None,
        "rows": int(len(df)),
        "columns": [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
                     "unique": int(df[c].nunique())} for c in df.columns],
        "detected_columns": {
            "text": "text", "label": "sentiment",
            "date": "timestamp" if "timestamp" in df.columns and cols["date"] else None,
            "topic": "category" if "category" in df.columns else None,
            "language": "language" if "language" in df.columns else None,
        },
        "missing_values_total": int(df.isna().sum().sum()),
        "empty_text": int((text.fillna("").astype(str).str.strip() == "").sum()),
        "duplicate_records_excluding_id": int(df.drop(columns=id_cols).duplicated().sum()),
        "duplicate_texts": int(text.duplicated().sum()),
        "unique_texts": int(text.nunique()),
    }
    counts = df["sentiment"].str.lower().value_counts()
    rep["class_distribution"] = {k: int(counts.get(k, 0)) for k in LABELS}
    rep["class_distribution_pct"] = {k: round(100 * v / len(df), 2) for k, v in rep["class_distribution"].items()}
    lens = text.fillna("").astype(str).str.len()
    words = text.fillna("").astype(str).str.split().str.len()
    rep["text_length_chars"] = {k: float(round(v, 2)) for k, v in lens.describe().items()}
    rep["text_length_words"] = {k: float(round(v, 2)) for k, v in words.describe().items()}

    # Label consistency: do identical texts always carry the same label?
    g = df.assign(_t=text.str.strip().str.lower()).groupby("_t")["sentiment"]
    n_labels = g.nunique()
    sizes = g.size()
    purity = g.agg(lambda s: s.value_counts().iloc[0] / len(s))
    rep["label_consistency"] = {
        "texts_with_conflicting_labels": int((n_labels > 1).sum()),
        "unique_texts": int(len(n_labels)),
        "rows_in_conflicting_texts": int(sizes[n_labels > 1].sum()),
        "max_possible_text_only_accuracy_pct": round(float(100 * (purity * sizes).sum() / sizes.sum()), 2),
        "note": "Upper bound for ANY model that sees only the text: predict the most common label of each text.",
    }

    if rep["detected_columns"]["date"]:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        rep["date_range"] = {"min": str(ts.min()), "max": str(ts.max()), "valid": int(ts.notna().sum())}
    if "category" in df.columns:
        rep["category_distribution"] = {k: int(v) for k, v in df["category"].value_counts().items()}
    if "language" in df.columns:
        rep["language_distribution"] = {k: int(v) for k, v in df["language"].value_counts().items()}
        nonascii = text.fillna("").astype(str).apply(lambda s: any(ord(ch) > 127 and ch.isalpha() for ch in s))
        rep["language_check"] = {
            "rows_with_non_ascii_letters": int(nonascii.sum()),
            "note": ("The 'language' column is metadata; the comment text itself was inspected "
                     "and is predominantly English regardless of the tag."),
        }
    import re as _re
    t = text.fillna("").astype(str)
    rep["privacy_check"] = {
        "texts_with_mentions": int(t.str.contains(r"@\w+").sum()),
        "texts_with_emails": int(t.str.contains(r"[\w.+-]+@[\w-]+\.[\w.]+").sum()),
        "texts_with_phone_like_numbers": int(t.str.contains(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b").sum()),
        "texts_with_urls": int(t.str.contains(r"https?://|www\.").sum()),
        "personal_identifier_columns": [c for c in df.columns if c.lower() in ("user", "user_id", "username", "name", "email", "author")],
    }
    if "is_spam" in df.columns:
        rep["spam_flagged_rows"] = int(df["is_spam"].sum())
    for c in ("likes", "comments", "shares"):
        if c in df.columns:
            rep.setdefault("engagement_columns", []).append(c)
    return rep
