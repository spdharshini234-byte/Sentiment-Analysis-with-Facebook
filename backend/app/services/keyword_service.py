"""Thin wrapper over src.analysis.keywords."""
from __future__ import annotations

import pandas as pd

from src.analysis import keywords as kw


def common_words(df: pd.DataFrame, limit: int, sentiment: str | None) -> dict:
    return {"method": "document frequency (posts containing the word)", "sentiment_filter": sentiment or "all",
            "words": kw.common_words(df, limit, sentiment), "tfidf_keywords": kw.tfidf_keywords(df, limit)}


def keyword_sentiment(df: pd.DataFrame, top_n: int, min_count: int) -> dict:
    return {"min_posts_per_keyword": min_count, "keywords": kw.keyword_sentiment(df, top_n, min_count)}


def concerns_and_drivers(df: pd.DataFrame, limit: int = 8) -> dict:
    return {"concerns": kw.associated_words(df, "negative", limit), "drivers": kw.associated_words(df, "positive", limit)}
