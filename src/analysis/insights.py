"""Rule-based insight generator. Every statement is built from computed statistics (no generative AI)."""
from __future__ import annotations

import pandas as pd

from src.analysis import keywords as kw
from src.analysis import sentiment as sa
from src.config import LABELS


def generate_insights(df: pd.DataFrame, evaluation: dict | None = None, data_report: dict | None = None) -> list[dict]:
    ins: list[dict] = []
    if df.empty:
        return ins
    d = sa.distribution(df)
    c, p, t = d["counts"], d["percentages"], d["total"]

    ins.append({"type": "overall", "severity": "info",
                "text": (f"Overall sentiment is {d['overall_sentiment']}: {p['positive']}% positive, "
                         f"{p['neutral']}% neutral and {p['negative']}% negative across {t:,} posts."),
                "evidence": {"counts": c}})

    cats = [r for r in sa.by_category(df) if r["total"] >= 50]
    if cats:
        worst = max(cats, key=lambda r: r["negative_pct"])
        best = max(cats, key=lambda r: r["positive_pct"])
        if worst["negative_pct"] > p["negative"]:
            ins.append({"type": "category_risk", "severity": "warning",
                        "text": (f"Negative sentiment is highest in '{worst['category']}' posts "
                                 f"({worst['negative_pct']}% negative vs {p['negative']}% overall, {worst['total']:,} posts)."),
                        "evidence": worst})
        ins.append({"type": "category_strength", "severity": "positive",
                    "text": (f"'{best['category']}' posts are the most positive "
                             f"({best['positive_pct']}% positive, {best['total']:,} posts)."),
                    "evidence": best})

    ks = kw.keyword_sentiment(df, top_n=40, min_count=30)
    if ks:
        neg = max(ks, key=lambda r: r["negative_pct"])
        if neg["negative_pct"] > p["negative"]:
            ins.append({"type": "keyword_negative", "severity": "warning",
                        "text": (f"Negative sentiment is concentrated around the keyword '{neg['keyword']}': "
                                 f"{neg['negative_pct']}% of the {neg['count']:,} posts that mention it are negative "
                                 f"(overall: {p['negative']}%)."),
                        "evidence": neg})
        pos = max(ks, key=lambda r: r["positive_pct"])
        ins.append({"type": "keyword_positive", "severity": "positive",
                    "text": (f"Posts mentioning '{pos['keyword']}' are the most positive: {pos['positive_pct']}% of "
                             f"{pos['count']:,} posts."), "evidence": pos})

    # recent trend: last 3 months of data vs everything before
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        dd = df.dropna(subset=["timestamp"])
        cutoff = dd["timestamp"].max() - pd.Timedelta(days=90)
        recent, earlier = dd[dd["timestamp"] > cutoff], dd[dd["timestamp"] <= cutoff]
        if len(recent) >= 100 and len(earlier) >= 100:
            rn = 100 * (recent["sentiment"] == "negative").mean()
            en = 100 * (earlier["sentiment"] == "negative").mean()
            diff = round(rn - en, 1)
            if abs(diff) >= 3:
                word = "higher" if diff > 0 else "lower"
                ins.append({"type": "trend", "severity": "warning" if diff > 0 else "positive",
                            "text": (f"Over the most recent 90 days negative sentiment is {abs(diff)} points {word} "
                                     f"({rn:.1f}%) than before ({en:.1f}%)."),
                            "evidence": {"recent_negative_pct": round(rn, 2), "earlier_negative_pct": round(en, 2)}})
            else:
                ins.append({"type": "trend", "severity": "info",
                            "text": (f"Negative sentiment is stable: {rn:.1f}% in the latest 90 days vs {en:.1f}% earlier."),
                            "evidence": {"recent_negative_pct": round(rn, 2), "earlier_negative_pct": round(en, 2)}})

    if data_report:
        lc = data_report.get("label_consistency", {})
        if lc.get("unique_texts"):
            ins.append({"type": "data_quality", "severity": "info",
                        "text": (f"{lc['texts_with_conflicting_labels']:,} of {lc['unique_texts']:,} unique comment texts "
                                 f"appear with more than one sentiment label, which caps text-only accuracy at about "
                                 f"{lc['max_possible_text_only_accuracy_pct']}%."),
                        "evidence": lc})
    if evaluation:
        m = evaluation["final_test_metrics"]
        ins.append({"type": "model", "severity": "info",
                    "text": (f"The selected model ({evaluation['selected_model']}) reaches macro-F1 {m['f1_macro']:.2f} and "
                             f"accuracy {m['accuracy']:.2f} on unseen test posts, versus macro-F1 "
                             f"{evaluation['baseline_test_f1_macro']:.2f} for a majority-class baseline."),
                    "evidence": {"f1_macro": m["f1_macro"], "accuracy": m["accuracy"]}})
    return ins
