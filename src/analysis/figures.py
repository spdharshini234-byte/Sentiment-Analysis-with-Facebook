"""Publication-quality EDA and evaluation figures saved to reports/figures/ (Phases 5 and 8)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.analysis import keywords as kw  # noqa: E402
from src.config import FIGURES_DIR, LABELS  # noqa: E402

PALETTE = {"negative": "#e5484d", "neutral": "#8b95a7", "positive": "#30a46c"}


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(df: pd.DataFrame, data_report: dict, evaluation: dict) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["n_chars"] = d["text"].str.len()
    d["n_words"] = d["text"].str.split().str.len()

    # 1 class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = d["sentiment"].value_counts().reindex(LABELS)
    ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({100 * v / len(d):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Sentiment class distribution"); ax.set_ylabel("Posts"); ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, "01_class_distribution.png")

    # 2 text length
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(d["n_chars"], bins=30, ax=axes[0], color="#4c78d0"); axes[0].set_title("Comment length (characters)")
    sns.boxplot(data=d, x="sentiment", y="n_chars", order=LABELS, palette=PALETTE, hue="sentiment", legend=False, ax=axes[1])
    axes[1].set_title("Length by sentiment"); axes[1].set_xlabel("")
    _save(fig, "02_text_length.png")

    # 3 top words per sentiment
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, lab in zip(axes, LABELS):
        words = kw.common_words(d, limit=10, sentiment=lab)
        ax.barh([w["word"] for w in words][::-1], [w["count"] for w in words][::-1], color=PALETTE[lab])
        ax.set_title(f"Top words in {lab} posts"); ax.set_xlabel("Posts containing word")
    _save(fig, "03_top_words_by_sentiment.png")

    # 4 sentiment by category
    if "category" in d.columns:
        ct = pd.crosstab(d["category"], d["sentiment"], normalize="index")[LABELS] * 100
        ct = ct.sort_values("negative")
        fig, ax = plt.subplots(figsize=(8, 5))
        ct.plot(kind="barh", stacked=True, color=[PALETTE[k] for k in LABELS], ax=ax)
        ax.set_xlabel("% of posts"); ax.set_ylabel(""); ax.set_title("Sentiment mix by post category")
        ax.legend(title="", loc="lower right")
        _save(fig, "04_sentiment_by_category.png")

    # 5 monthly trend
    if d["timestamp"].notna().any():
        m = d.dropna(subset=["timestamp"]).assign(month=lambda x: x["timestamp"].dt.to_period("M").dt.start_time)
        pv = m.pivot_table(index="month", columns="sentiment", values="text", aggfunc="count", fill_value=0)[LABELS]
        fig, ax = plt.subplots(figsize=(10, 4))
        for lab in LABELS:
            ax.plot(pv.index, pv[lab], label=lab, color=PALETTE[lab], lw=2)
        ax.set_title("Monthly sentiment trend (post counts)"); ax.set_ylabel("Posts"); ax.legend()
        _save(fig, "05_monthly_trend.png")

    # 6 sentiment by language tag
    if "language" in d.columns:
        lt = pd.crosstab(d["language"], d["sentiment"], normalize="index")[LABELS] * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        lt.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in LABELS], ax=ax)
        ax.set_ylabel("% of posts"); ax.set_xlabel("language tag"); ax.set_title("Sentiment mix by language tag")
        plt.setp(ax.get_xticklabels(), rotation=0)
        _save(fig, "06_sentiment_by_language.png")

    # 7 label consistency (data-quality finding)
    g = d.groupby("text_group")["sentiment"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    n = d.groupby("text_group").size()
    multi = g[n >= 5]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(multi, bins=15, ax=ax, color="#7c5cd6")
    ax.set_xlabel("Share of the majority label for a repeated comment text"); ax.set_ylabel("Number of distinct texts")
    ax.set_title("Label consistency of repeated comments (texts seen 5+ times)")
    _save(fig, "07_label_consistency.png")

    # 8 model comparison
    models = evaluation["models"]
    names = [r["model"].replace(" (majority class)", "\n(majority)") for r in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, [r["validation"]["f1_macro"] for r in models], w, label="Validation macro-F1", color="#4c78d0")
    ax.bar(x + w / 2, [r["test"]["f1_macro"] for r in models], w, label="Test macro-F1", color="#f28e2b")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.set_title("Model comparison (macro-F1)")
    ax.legend()
    _save(fig, "08_model_comparison.png")

    # 9 confusion matrix (selected, test)
    cm = np.array(evaluation["final_test_metrics"]["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axes[0], cbar=False)
    axes[0].set_title(f"Confusion matrix - {evaluation['selected_model']} (test)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Greens", xticklabels=LABELS,
                yticklabels=LABELS, ax=axes[1], cbar=False)
    axes[1].set_title("Row-normalised (recall per class)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("")
    _save(fig, "09_confusion_matrix.png")

    # 10 per-class f1
    pc = evaluation["final_test_metrics"]["per_class"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(3); w = 0.26
    for i, metric in enumerate(("precision", "recall", "f1")):
        ax.bar(xs + (i - 1) * w, [pc[k][metric] for k in LABELS], w, label=metric)
    ax.set_xticks(xs); ax.set_xticklabels(LABELS); ax.set_ylim(0, 1); ax.set_title("Per-class test metrics"); ax.legend()
    _save(fig, "10_per_class_metrics.png")
