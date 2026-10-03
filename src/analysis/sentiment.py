"""Sentiment analytics computed from a dataframe of posts. Nothing here is hard-coded."""
from __future__ import annotations

import pandas as pd

from src import config
from src.config import LABELS

NO_DATE_MESSAGE = ("Sentiment trend cannot be generated because the source dataset has no usable date/time information.")


def distribution(df: pd.DataFrame, col: str = "sentiment") -> dict:
    total = int(len(df))
    counts = {k: int((df[col] == k).sum()) for k in LABELS}
    pct = {k: (round(100 * v / total, 2) if total else 0.0) for k, v in counts.items()}
    return {"total": total, "counts": counts, "percentages": pct, "overall_sentiment": overall_sentiment(counts, total)}


def overall_sentiment(counts: dict, total: int) -> str:
    """Label with the most posts; 'MIXED' when the top two labels are within 3 percentage points."""
    if not total:
        return "UNAVAILABLE"
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and 100 * (ranked[0][1] - ranked[1][1]) / total < 3:
        return "MIXED"
    return ranked[0][0].upper()


def health_score(counts: dict, total: int) -> float | None:
    """Sentiment Health (0-100) = 100 * (positive + 0.5 * neutral) / total."""
    if not total:
        return None
    return round(100 * (counts["positive"] + 0.5 * counts["neutral"]) / total, 1)


def voice_index(counts: dict) -> float | None:
    """Customer Voice Index (-100..+100) = 100 * (positive - negative) / (positive + negative)."""
    polar = counts["positive"] + counts["negative"]
    if not polar:
        return None
    return round(100 * (counts["positive"] - counts["negative"]) / polar, 1)


def trend(df: pd.DataFrame, granularity: str = "month", start: str | None = None, end: str | None = None) -> dict:
    if "timestamp" not in df.columns or df["timestamp"].notna().sum() == 0:
        return {"available": False, "message": NO_DATE_MESSAGE, "points": []}
    d = df.dropna(subset=["timestamp"]).copy()
    date_min, date_max = d["timestamp"].min(), d["timestamp"].max()
    if start:
        d = d[d["timestamp"] >= pd.to_datetime(start, errors="coerce")]
    if end:
        d = d[d["timestamp"] < pd.to_datetime(end, errors="coerce") + pd.Timedelta(days=1)]
    granularity = granularity if granularity in ("day", "week", "month") else "month"
    if d.empty:
        return {"available": True, "granularity": granularity, "points": [],
                "date_range": {"min": str(date_min.date()), "max": str(date_max.date())},
                "message": "No posts fall inside the selected date range."}
    freq = {"day": "D", "week": "W-MON", "month": "MS"}[granularity]
    if granularity == "week":
        d["period"] = d["timestamp"].dt.to_period("W-SUN").dt.start_time
    elif granularity == "month":
        d["period"] = d["timestamp"].dt.to_period("M").dt.start_time
    else:
        d["period"] = d["timestamp"].dt.normalize()
    pv = d.pivot_table(index="period", columns="sentiment", values="text", aggfunc="count", fill_value=0)
    for k in LABELS:
        if k not in pv.columns:
            pv[k] = 0
    pv = pv[LABELS].sort_index()
    pv["total"] = pv.sum(axis=1)
    fmt = "%Y-%m" if granularity == "month" else "%Y-%m-%d"
    pts = []
    for period, row in pv.iterrows():
        t = int(row["total"])
        pts.append({
            "period": period.strftime(fmt),
            **{k: int(row[k]) for k in LABELS},
            **{f"{k}_pct": round(100 * int(row[k]) / t, 2) if t else 0.0 for k in LABELS},
            "total": t,
        })
    return {"available": True, "granularity": granularity, "points": pts,
            "date_range": {"min": str(date_min.date()), "max": str(date_max.date())}}


def sentiment_by_record_range(df: pd.DataFrame, bins: int = 10) -> list[dict]:
    """Fallback visual when there is no date column: sentiment across consecutive record blocks."""
    n = len(df)
    if n == 0:
        return []
    size = max(1, n // bins)
    out = []
    for i in range(0, n, size):
        blk = df.iloc[i:i + size]
        out.append({"range": f"{i + 1}-{min(i + size, n)}", **{k: int((blk['sentiment'] == k).sum()) for k in LABELS}})
    return out


def by_category(df: pd.DataFrame) -> list[dict]:
    if "category" not in df.columns:
        return []
    rows = []
    for cat, g in df.groupby("category"):
        d = distribution(g)
        rows.append({"category": cat, "total": d["total"], **d["counts"],
                     **{f"{k}_pct": v for k, v in d["percentages"].items()}})
    return sorted(rows, key=lambda r: -r["total"])


def confidence_distribution(df: pd.DataFrame) -> dict:
    """High / medium / low confidence of the model, measured on the held-out test split."""
    scope = "held-out test split"
    d = df[df["split"] == "test"] if "split" in df.columns else df
    if d.empty:
        d, scope = df, "all posts"
    c = d["pred_conf"]
    buckets = {
        "high": int((c >= config.CONF_HIGH).sum()),
        "medium": int(((c >= config.CONF_MEDIUM) & (c < config.CONF_HIGH)).sum()),
        "low": int((c < config.CONF_MEDIUM).sum()),
    }
    total = int(len(d))
    return {"scope": scope, "total": total, "counts": buckets,
            "percentages": {k: (round(100 * v / total, 2) if total else 0.0) for k, v in buckets.items()},
            "thresholds": {"high": f">= {config.CONF_HIGH}", "medium": f"{config.CONF_MEDIUM} to < {config.CONF_HIGH}",
                           "low": f"< {config.CONF_MEDIUM}"}}
