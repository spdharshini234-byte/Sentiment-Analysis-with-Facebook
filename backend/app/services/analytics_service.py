"""Dashboard analytics. Every number is computed from the SQLite table `posts` (the Kaggle data scored by the model)."""
from __future__ import annotations

import json
import threading

import pandas as pd

from backend.app.database import database as db
from backend.app.services import insight_service, keyword_service
from src import config
from src.analysis import sentiment as sa
from src.config import LABELS


class ServiceUnavailable(Exception):
    """Raised when the database / reports needed for an endpoint are missing."""


_lock = threading.Lock()
_df: pd.DataFrame | None = None


def get_df() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df
    with _lock:
        if _df is None:
            if not db.database_ready():
                raise ServiceUnavailable(
                    "Analytics database is not ready. Run `python scripts/train_model.py` and "
                    "`python scripts/build_database.py`, then restart the backend."
                )
            _df = db.load_posts_df()
    return _df


def reset_cache() -> None:
    global _df
    _df = None
    _top_cache.clear()


def _read_json(path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def evaluation() -> dict | None:
    return _read_json(config.EVALUATION_PATH)


def data_report() -> dict | None:
    return _read_json(config.DATA_REPORT_PATH)


def metadata() -> dict | None:
    return _read_json(config.METADATA_PATH)


# ---------------------------------------------------------------- endpoints
def sentiment_distribution() -> dict:
    df = get_df()
    d = sa.distribution(df)
    pred = sa.distribution(df.rename(columns={"sentiment": "_label", "pred_label": "sentiment"}))
    return {
        "source": "dataset labels",
        **d,
        "health": {"score": sa.health_score(d["counts"], d["total"]),
                   "formula": "100 x (positive + 0.5 x neutral) / total posts"},
        "voice_index": {"score": sa.voice_index(d["counts"]),
                        "formula": "100 x (positive - negative) / (positive + negative); range -100 to +100"},
        "model_predicted": {"counts": pred["counts"], "percentages": pred["percentages"]},
        "by_category": sa.by_category(df),
        "confidence": sa.confidence_distribution(df),
    }


def sentiment_trend(granularity: str, start: str | None, end: str | None) -> dict:
    df = get_df()
    res = sa.trend(df, granularity, start, end)
    if not res["available"]:
        res["fallback"] = {"title": "Sentiment by record range", "data": sa.sentiment_by_record_range(df)}
    return res


def common_words(limit: int, sentiment: str | None) -> dict:
    return keyword_service.common_words(get_df(), limit, sentiment)


def keyword_sentiment(top_n: int, min_count: int) -> dict:
    return keyword_service.keyword_sentiment(get_df(), top_n, min_count)


_top_cache: dict[str, list[dict]] = {}


def _top_comments(sentiment: str, limit: int) -> list[dict]:
    """Distinct comment texts the model classifies as `sentiment` AND that carry that label in the dataset,
    ranked by the model's probability for that sentiment."""
    if sentiment not in _top_cache:
        df = get_df()
        col = f"p_{sentiment}"
        rows = []
        for _, g in df.groupby("clean_text", sort=False):
            first = g.iloc[0]
            labelled = g[g["sentiment"] == sentiment]
            if first["pred_label"] != sentiment or labelled.empty:
                continue
            latest = labelled["timestamp"].max()
            rows.append({
                "comment": first["text"],
                "predicted_sentiment": first["pred_label"],
                "confidence": round(float(first[col]), 4),
                "occurrences": int(len(g)),
                "labelled_in_dataset": int(len(labelled)),
                "category": g["category"].mode().iloc[0] if g["category"].notna().any() else None,
                "latest_date": None if pd.isna(latest) else str(latest.date()),
            })
        rows.sort(key=lambda r: (-r["confidence"], -r["labelled_in_dataset"]))
        _top_cache[sentiment] = rows
    return _top_cache[sentiment][:limit]


def negative_feedback(limit: int) -> dict:
    return {
        "definition": ("Distinct comments that the model classifies as negative AND that carry a negative label in the "
                       "dataset, sorted by model probability of 'negative'."),
        "emotion_available": False,
        "source_field": "category (the dataset has no platform/source column)",
        "items": _top_comments("negative", limit),
    }


def positive_feedback(limit: int) -> dict:
    return {"items": _top_comments("positive", limit)}


def model_performance() -> dict:
    ev = evaluation()
    if ev is None:
        raise ServiceUnavailable("Model evaluation report not found. Run `python scripts/train_model.py`.")
    return {
        "selected_model": ev["selected_model"], "selection_rule": ev["selection_rule"],
        "models": [{"model": r["model"], "best_params": r["best_params"], "cv_f1_macro": r["cv_f1_macro_mean"],
                    "train": _slim(r["train"]), "validation": _slim(r["validation"]), "test": _slim(r["test"])}
                   for r in ev["models"]],
        "final_test_metrics": ev["final_test_metrics"],
        "baseline": {"accuracy": ev["baseline_test_accuracy"], "f1_macro": ev["baseline_test_f1_macro"]},
        "lift_over_baseline_f1_macro": ev["lift_over_baseline_f1_macro"],
        "overfitting_check": ev["overfitting_check"], "success_criteria": ev["success_criteria"],
        "top_confusions": ev["top_confusions"], "error_examples": ev["error_examples"],
        "labels": LABELS,
    }


def _slim(m: dict) -> dict:
    keys = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted",
            "f1_weighted", "n")
    return {k: m[k] for k in keys}


def dataset_info() -> dict:
    rep = data_report()
    if rep is None:
        raise ServiceUnavailable("Dataset report not found. Run `python scripts/train_model.py`.")
    out = dict(rep)
    out["live"] = {"posts_in_database": int(len(get_df())) if db.database_ready() else 0}
    return out


def _date_range(df: pd.DataFrame) -> dict | None:
    t = sa.trend(df, "month")
    return t.get("date_range") if t["available"] else None


def dashboard() -> dict:
    df = get_df()
    dist = sentiment_distribution()
    ev, rep, meta = evaluation(), data_report(), metadata()
    words = keyword_service.concerns_and_drivers(df)
    model = None
    if ev and meta:
        m = ev["final_test_metrics"]
        model = {"name": ev["selected_model"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                 "f1_weighted": m["f1_weighted"], "evaluated_on": "held-out test split",
                 "trained_at_utc": meta.get("trained_at_utc")}
    return {
        "available": True,
        "title": "Facebook Sentiment Intelligence",
        "totals": {k: dist[k] for k in ("total", "counts", "percentages", "overall_sentiment")},
        "health": dist["health"], "voice_index": dist["voice_index"],
        "confidence": dist["confidence"],
        "common_words": keyword_service.common_words(df, 10, None)["words"],
        "top_negative_comments": _top_comments("negative", 5),
        "top_positive_comments": _top_comments("positive", 5),
        "concern_areas": words["concerns"], "positive_drivers": words["drivers"],
        "model": model,
        "insights": insight_service.build(df, ev, rep),
        "date_range": _date_range(df),
    }
