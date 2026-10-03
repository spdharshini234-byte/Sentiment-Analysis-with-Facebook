import pandas as pd

from src.analysis import keywords as kw
from src.analysis import sentiment as sa
from src.analysis.insights import generate_insights
from src.data.preprocess import clean_text


def _df(n_pos=6, n_neg=2, n_neu=2, with_dates=True):
    labels = ["positive"] * n_pos + ["negative"] * n_neg + ["neutral"] * n_neu
    texts = ["great price and quality"] * n_pos + ["terrible price"] * n_neg + ["just a post about service"] * n_neu
    df = pd.DataFrame({"text": texts, "sentiment": labels, "category": ["a"] * len(texts)})
    df["clean_text"] = [clean_text(t) for t in df["text"]]
    if with_dates:
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="10D")
    return df


def test_distribution_is_computed_not_hardcoded():
    d = sa.distribution(_df())
    assert d["total"] == 10 and d["counts"]["positive"] == 6 and d["percentages"]["negative"] == 20.0
    assert d["overall_sentiment"] == "POSITIVE"


def test_scores_formula():
    c = {"positive": 6, "negative": 2, "neutral": 2}
    assert sa.health_score(c, 10) == 70.0          # 100*(6+1)/10
    assert sa.voice_index(c) == 50.0               # 100*(6-2)/8
    assert 0 <= sa.health_score({"positive": 0, "negative": 5, "neutral": 0}, 5) <= 100


def test_trend_with_dates():
    t = sa.trend(_df(), "month")
    assert t["available"] and sum(p["total"] for p in t["points"]) == 10


def test_trend_without_dates_is_safe():
    t = sa.trend(_df(with_dates=False), "month")
    assert t["available"] is False and "no usable date" in t["message"]


def test_keyword_sentiment_only_real_words():
    rows = kw.keyword_sentiment(_df(), top_n=5, min_count=2)
    words = {r["keyword"] for r in rows}
    assert "price" in words and "unicorn" not in words
    price = next(r for r in rows if r["keyword"] == "price")
    assert price["count"] == 8 and price["negative"] == 2


def test_associated_words_negative():
    out = kw.associated_words(_df(), "negative", min_docs=2)
    assert any(o["word"] == "terrible" for o in out)


def test_insights_are_generated_from_data():
    ins = generate_insights(_df(60, 20, 20))
    assert ins and all("text" in i and "evidence" in i for i in ins)
