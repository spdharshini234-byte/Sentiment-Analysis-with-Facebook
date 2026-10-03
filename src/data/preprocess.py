"""Text cleaning and dataset preparation (Phase 4: Data Preparation).

Design rule: clean enough to remove noise, but never destroy sentiment.
 - negations (not, no, never ...) are KEPT even though they are usually stop-words
 - "!" and "?" and emojis are KEPT as separate tokens (they carry tone)
 - URLs, HTML, @mentions are removed; #hashtags keep their word
"""
from __future__ import annotations

import html
import re
from typing import Iterable

import pandas as pd

from src.config import LABELS

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictographs, emoticons, symbols
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F000-\U0001F2FF"
    "\U00002B00-\U00002BFF"
    "]",
    flags=re.UNICODE,
)
KEEP_PUNCT_RE = re.compile(r"([!?])")
NON_TOKEN_RE = re.compile(r"[^\w\s!?']|_", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")

NEGATIONS = {
    "no", "not", "nor", "never", "none", "nothing", "neither", "cannot",
    "without", "dont", "doesnt", "didnt", "isnt", "wasnt", "wont", "cant",
    "couldnt", "shouldnt", "wouldnt", "arent", "havent", "hasnt", "hadnt",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "won't", "can't",
    "couldn't", "shouldn't", "wouldn't", "aren't", "haven't", "hasn't", "hadn't",
}

_BASE_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "should", "now", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "would", "could", "also", "get", "got", "us", "im", "ive",
    "i'm", "i've", "it's", "that's", "let's", "ll", "re", "ve", "d", "m",
}
# Negations are removed from the stop-word list on purpose.
STOPWORDS = frozenset(_BASE_STOPWORDS - NEGATIONS)
# Extra tokens that are useless as business keywords.
KEYWORD_STOPWORDS = frozenset({"num", "like", "one", "today", "week", "really"})


def clean_text(text: object, remove_stopwords: bool = True) -> str:
    """Return a normalised version of one comment."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = html.unescape(s)
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r" \1 ", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.lower()
    s = NUMBER_RE.sub(" num ", s)
    s = EMOJI_RE.sub(lambda m: f" {m.group(0)} ", s)
    s = KEEP_PUNCT_RE.sub(r" \1 ", s)
    s = NON_TOKEN_RE.sub(lambda m: f" {m.group(0)} " if EMOJI_RE.match(m.group(0)) else " ", s)
    s = SPACE_RE.sub(" ", s).strip()
    if remove_stopwords:
        s = " ".join(t for t in s.split(" ") if t not in STOPWORDS)
    return s


def clean_many(texts: Iterable[object]) -> list[str]:
    """Vectorised helper used as the first step of the saved sklearn pipeline."""
    return [clean_text(t) for t in texts]


def keyword_tokens(cleaned: str) -> list[str]:
    """Meaningful words (for keyword analytics) from an already cleaned text."""
    out = []
    for t in cleaned.split(" "):
        t = t.strip("'")
        if len(t) >= 3 and t.isalpha() and t not in STOPWORDS and t not in KEYWORD_STOPWORDS and t not in NEGATIONS:
            out.append(t)
    return out


def prepare_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> tuple[pd.DataFrame, dict]:
    """Clean a raw dataframe. Returns (clean_df, report)."""
    report: dict = {"rows_initial": int(len(df))}
    d = df.copy()

    # 1. missing / empty comments
    d[text_col] = d[text_col].astype("object")
    missing = d[text_col].isna() | (d[text_col].astype(str).str.strip() == "")
    report["dropped_missing_or_empty_text"] = int(missing.sum())
    d = d[~missing].copy()

    # 2. invalid labels
    d[label_col] = d[label_col].astype(str).str.strip().str.lower()
    bad_label = ~d[label_col].isin(LABELS)
    report["dropped_invalid_label"] = int(bad_label.sum())
    d = d[~bad_label].copy()

    # 3. exact duplicate records (every column identical except the id column)
    id_cols = [c for c in ("post_id", "id") if c in d.columns]
    dup_mask = d.drop(columns=id_cols).duplicated(keep="first")
    report["dropped_exact_duplicate_records"] = int(dup_mask.sum())
    d = d[~dup_mask].copy()

    # 4. cleaned text; drop rows that become empty
    d["clean_text"] = [clean_text(t) for t in d[text_col]]
    empty_after = d["clean_text"].str.len() == 0
    report["dropped_empty_after_cleaning"] = int(empty_after.sum())
    d = d[~empty_after].copy()

    d["text_group"] = d["clean_text"]  # identical cleaned text => same group (prevents leakage)
    d = d.reset_index(drop=True)
    report["rows_final"] = int(len(d))
    report["unique_texts_final"] = int(d["text_group"].nunique())
    return d, report
