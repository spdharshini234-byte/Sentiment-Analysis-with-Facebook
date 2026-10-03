"""Keyword analytics: frequency, TF-IDF, keyword x sentiment, concern areas and positive drivers."""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LABELS
from src.data.preprocess import keyword_tokens


def add_tokens(df: pd.DataFrame) -> pd.DataFrame:
    if "kw_tokens" not in df.columns:
        df = df.copy()
        df["kw_tokens"] = [sorted(set(keyword_tokens(t))) for t in df["clean_text"]]
    return df


def common_words(df: pd.DataFrame, limit: int = 15, sentiment: str | None = None) -> list[dict]:
    """Document frequency: number of posts that contain the word (stop-words/URLs/symbols removed)."""
    df = add_tokens(df)
    if sentiment in LABELS:
        df = df[df["sentiment"] == sentiment]
    total = len(df)
    c = Counter(w for toks in df["kw_tokens"] for w in toks)
    return [{"word": w, "count": n, "percent_of_posts": round(100 * n / total, 2) if total else 0.0}
            for w, n in c.most_common(limit)]


def tfidf_keywords(df: pd.DataFrame, limit: int = 15) -> list[dict]:
    df = add_tokens(df)
    docs = [" ".join(t) for t in df["kw_tokens"]]
    docs = [d for d in docs if d]
    if not docs:
        return []
    vec = TfidfVectorizer(lowercase=False, token_pattern=r"\S+", sublinear_tf=True)
    X = vec.fit_transform(docs)
    scores = X.mean(axis=0).A1
    names = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:limit]
    return [{"word": names[i], "score": round(float(scores[i]), 5)} for i in order]


def keyword_sentiment(df: pd.DataFrame, top_n: int = 15, min_count: int = 30) -> list[dict]:
    """Sentiment split for the most frequent keywords (only words that really occur)."""
    df = add_tokens(df)
    rows = []
    exploded = df[["sentiment", "kw_tokens"]].explode("kw_tokens").dropna()
    freq = exploded["kw_tokens"].value_counts()
    for w in [w for w, n in freq.items() if n >= min_count][:top_n]:
        sub = exploded[exploded["kw_tokens"] == w]["sentiment"].value_counts()
        n = int(sub.sum())
        row = {"keyword": w, "count": n}
        for k in LABELS:
            row[k] = int(sub.get(k, 0))
            row[f"{k}_pct"] = round(100 * int(sub.get(k, 0)) / n, 1)
        row["dominant"] = max(LABELS, key=lambda k: row[k])
        rows.append(row)
    return rows


def associated_words(df: pd.DataFrame, target: str, limit: int = 8, min_docs: int = 25) -> list[dict]:
    """Words most strongly associated with `target` sentiment (smoothed log-odds ratio vs the other classes)."""
    df = add_tokens(df)
    in_cls = df["sentiment"] == target
    n_in, n_out = int(in_cls.sum()), int((~in_cls).sum())
    if n_in == 0 or n_out == 0:
        return []
    a, b = Counter(), Counter()
    for toks, flag in zip(df["kw_tokens"], in_cls):
        (a if flag else b).update(toks)
    out = []
    for w in set(a) | set(b):
        ai, bi = a[w], b[w]
        if ai + bi < min_docs:
            continue
        lo = math.log((ai + 0.5) / (n_in - ai + 0.5)) - math.log((bi + 0.5) / (n_out - bi + 0.5))
        share = ai / (ai + bi)
        out.append({"word": w, "posts": ai + bi, "share_in_class_pct": round(100 * share, 1),
                    "baseline_pct": round(100 * n_in / (n_in + n_out), 1), "log_odds": round(lo, 3)})
    out = [o for o in out if o["log_odds"] > 0]
    return sorted(out, key=lambda r: -r["log_odds"])[:limit]
