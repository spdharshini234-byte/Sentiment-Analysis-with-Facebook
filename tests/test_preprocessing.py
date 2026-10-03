import pandas as pd

from src.data.preprocess import clean_text, keyword_tokens, prepare_dataframe
from src.models.train import grouped_stratified_split


def test_removes_urls_html_mentions():
    out = clean_text("Check <b>this</b> @bob https://x.co/a now")
    assert "http" not in out and "<b>" not in out and "@" not in out and "bob" not in out


def test_hashtag_keeps_word_and_lowercases():
    assert "relatable" in clean_text("So true #Relatable")


def test_negation_and_tone_preserved():
    out = clean_text("This is NOT good!")
    assert "not" in out.split() and "!" in out.split()


def test_emoji_kept_as_token():
    assert "🙌" in clean_text("Feeling happy 🙌").split()


def test_missing_and_empty_inputs():
    assert clean_text(None) == "" and clean_text(float("nan")) == "" and clean_text("   ") == ""


def test_keyword_tokens_drop_stopwords_and_negations():
    toks = keyword_tokens(clean_text("The price is not good today"))
    assert "price" in toks and "good" in toks and "not" not in toks and "the" not in toks


def test_prepare_dataframe_drops_bad_rows():
    df = pd.DataFrame({
        "post_id": [1, 2, 3, 4, 5],
        "text": ["good product", None, "   ", "good product", "https://only.link"],
        "sentiment": ["positive", "positive", "neutral", "positive", "neutral"],
    })
    out, rep = prepare_dataframe(df)
    assert rep["dropped_missing_or_empty_text"] == 2
    assert rep["dropped_exact_duplicate_records"] == 1
    assert rep["dropped_empty_after_cleaning"] == 1
    assert len(out) == 1


def test_split_has_no_text_leakage():
    texts = [f"remark {chr(97 + i % 26)}{chr(97 + (i % 60) // 26)}xyz" for i in range(600)]
    labels = ["positive", "neutral", "negative"] * 200
    df = pd.DataFrame({"text": texts, "sentiment": labels, "likes": list(range(600))})
    df, _ = prepare_dataframe(df)
    tr, va, te = grouped_stratified_split(df)
    g = df["text_group"].values
    assert not (set(g[tr]) & set(g[te])) and not (set(g[tr]) & set(g[va])) and not (set(g[va]) & set(g[te]))
