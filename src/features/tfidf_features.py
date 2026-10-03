"""TF-IDF feature extraction (Phase 6/7: feature engineering)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(ngram_range=(1, 2), min_df=2, max_features=20000) -> TfidfVectorizer:
    """Vectorizer for text that has ALREADY been cleaned by src.data.preprocess.clean_text.

    The cleaner already separated tokens with spaces (including emojis and ! ?), so we simply
    split on whitespace and keep all tokens.
    """
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"\S+",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
