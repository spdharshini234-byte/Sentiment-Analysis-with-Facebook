"""Training workflow (Phases 6, 7, 8). Run with:  python scripts/train_model.py"""
from __future__ import annotations

import json
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src import config
from src.config import LABELS, RANDOM_STATE
from src.data.load_data import inspect_dataset, load_raw
from src.data.preprocess import clean_many, prepare_dataframe
from src.features.tfidf_features import build_vectorizer
from src.models.evaluate import compute_metrics, top_confusions


def grouped_stratified_split(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """70/15/15 split where identical (cleaned) texts never straddle two splits.

    StratifiedGroupKFold keeps label proportions similar while honouring the groups.
    """
    y, g = df["sentiment"].values, df["text_group"].values
    idx = np.arange(len(df))
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(idx, y, g))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(inner.split(rest_idx, y[rest_idx], g[rest_idx]))
    train_idx, val_idx = rest_idx[tr_rel], rest_idx[va_rel]
    # leakage check
    sets = [set(g[train_idx]), set(g[val_idx]), set(g[test_idx])]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), "text leakage between splits"
    return train_idx, val_idx, test_idx


def _pipe(clf, ngram=(1, 2), min_df=2) -> Pipeline:
    return Pipeline([
        ("clean", FunctionTransformer(clean_many, validate=False)),
        ("tfidf", build_vectorizer(ngram_range=ngram, min_df=min_df)),
        ("clf", clf),
    ])


def candidate_models():
    """name -> (pipeline, param grid). The baseline is a majority-class dummy."""
    grid_common = {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2]}
    return {
        "Baseline (majority class)": (_pipe(DummyClassifier(strategy="most_frequent")), {}),
        "Multinomial Naive Bayes": (_pipe(MultinomialNB()), {**grid_common, "clf__alpha": [0.1, 0.5, 1.0]}),
        "Logistic Regression": (
            _pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            {**grid_common, "clf__C": [0.1, 1.0, 10.0]}),
        "Linear SVM": (
            _pipe(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            {**grid_common, "clf__C": [0.05, 0.1, 0.5, 1.0]}),
    }


def _params_to_json(p: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}


def run_training(verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. load + validate schema
    path = config.find_dataset()
    raw = load_raw(path)
    log(f"[1/9] Loaded {len(raw)} rows from {path.name}")
    data_report = inspect_dataset(raw, path)

    # 3-5. clean, remove invalid rows, check duplicates
    df, prep_report = prepare_dataframe(raw)
    data_report["preparation"] = prep_report
    log(f"[2/9] Prepared dataset: {prep_report}")

    # 6. split
    tr, va, te = grouped_stratified_split(df)
    split_info = {"strategy": "StratifiedGroupKFold (groups = identical cleaned text), 70/15/15",
                  "random_state": RANDOM_STATE, "leakage_check": "passed (no text group appears in two splits)"}
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        sub = df.iloc[idx]
        split_info[name] = {"rows": int(len(sub)), "unique_text_groups": int(sub["text_group"].nunique()),
                            "class_distribution": {k: int((sub["sentiment"] == k).sum()) for k in LABELS}}
    data_report["split"] = split_info
    log(f"[3/9] Split: " + ", ".join(f"{k}={split_info[k]['rows']}" for k in ("train", "validation", "test")))

    df["split"] = "train"
    df.loc[df.index[va], "split"] = "validation"
    df.loc[df.index[te], "split"] = "test"
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["text_group"]).to_csv(config.DATA_PROCESSED / "prepared_dataset.csv", index=False)

    X_tr, y_tr = df.iloc[tr]["text"].tolist(), df.iloc[tr]["sentiment"].values
    g_tr = df.iloc[tr]["text_group"].values
    X_va, y_va = df.iloc[va]["text"].tolist(), df.iloc[va]["sentiment"].values
    X_te, y_te = df.iloc[te]["text"].tolist(), df.iloc[te]["sentiment"].values

    # 7-8. train & tune each candidate with group-aware cross-validation, then evaluate
    results, fitted = [], {}
    cv = GroupKFold(n_splits=5)
    for name, (pipe, grid) in candidate_models().items():
        t1 = time.time()
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gs.fit(X_tr, y_tr, groups=g_tr)
            best, best_params, cv_score = gs.best_estimator_, _params_to_json(gs.best_params_), float(gs.best_score_)
            cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
        else:
            best = pipe.fit(X_tr, y_tr)
            best_params, cv_score, cv_std = {}, None, None
        fitted[name] = best
        m_tr = compute_metrics(y_tr, best.predict(X_tr))
        m_va = compute_metrics(y_va, best.predict(X_va))
        m_te = compute_metrics(y_te, best.predict(X_te))
        results.append({
            "model": name, "best_params": best_params,
            "cv_f1_macro_mean": cv_score, "cv_f1_macro_std": cv_std,
            "train": m_tr, "validation": m_va, "test": m_te,
            "train_seconds": round(time.time() - t1, 2),
        })
        log(f"[4/9] {name:28s} val macroF1={m_va['f1_macro']:.4f}  test macroF1={m_te['f1_macro']:.4f}  acc={m_te['accuracy']:.4f}")

    # 9. select by VALIDATION macro-F1 among the three required models (baseline excluded)
    contenders = [r for r in results if not r["model"].startswith("Baseline")]
    selected = max(contenders, key=lambda r: r["validation"]["f1_macro"])
    final_pipe = fitted[selected["model"]]
    log(f"[5/9] Selected: {selected['model']} (validation macro-F1 = {selected['validation']['f1_macro']:.4f})")

    base = next(r for r in results if r["model"].startswith("Baseline"))
    test_m = selected["test"]
    gap = selected["train"]["f1_macro"] - test_m["f1_macro"]
    # error analysis on the held-out test set
    test_df = df.iloc[te].copy()
    test_df["pred"] = final_pipe.predict(X_te)
    wrong = test_df[test_df["pred"] != test_df["sentiment"]]
    samples = wrong.sample(min(12, len(wrong)), random_state=RANDOM_STATE)[["text", "category", "sentiment", "pred"]]
    ceiling = data_report["label_consistency"]["max_possible_text_only_accuracy_pct"]
    evaluation = {
        "selection_rule": "highest macro-F1 on the VALIDATION split among Multinomial NB, Logistic Regression and Linear SVM",
        "selected_model": selected["model"],
        "models": results,
        "final_test_metrics": test_m,
        "baseline_test_accuracy": base["test"]["accuracy"],
        "baseline_test_f1_macro": base["test"]["f1_macro"],
        "lift_over_baseline_f1_macro": test_m["f1_macro"] - base["test"]["f1_macro"],
        "overfitting_check": {
            "train_f1_macro": selected["train"]["f1_macro"],
            "validation_f1_macro": selected["validation"]["f1_macro"],
            "test_f1_macro": test_m["f1_macro"],
            "train_minus_test_gap": gap,
            "verdict": ("Large train/test gap: the model is overfitting." if gap > 0.15 else
                        "Moderate gap: mild overfitting." if gap > 0.07 else
                        "Small gap: no strong sign of overfitting."),
            "underfitting_note": ("Even train macro-F1 is low, so most remaining error comes from noisy/inconsistent "
                                  "labels rather than model capacity." if selected["train"]["f1_macro"] < 0.75 else ""),
        },
        "success_criteria": {
            "criterion_1": "Final model beats the majority-class baseline on test macro-F1",
            "criterion_1_met": bool(test_m["f1_macro"] > base["test"]["f1_macro"]),
            "criterion_2": "Test accuracy stays below the label-consistency ceiling (sanity check against leakage)",
            "label_consistency_ceiling_accuracy_pct": ceiling,
            "criterion_2_met": bool(100 * test_m["accuracy"] <= ceiling + 5),
        },
        "top_confusions": top_confusions(test_m["confusion_matrix"]),
        "error_examples": samples.to_dict(orient="records"),
    }

    # save
    joblib.dump(final_pipe, config.PIPELINE_PATH, compress=3)
    config.LABEL_MAP_PATH.write_text(json.dumps({"labels": LABELS, "label_to_id": {l: i for i, l in enumerate(LABELS)}}, indent=2), encoding="utf-8")
    metadata = {
        "model_name": selected["model"],
        "best_params": selected["best_params"],
        "selection_metric": "validation macro-F1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "labels": LABELS,
        "input": "raw comment text (cleaning is the first step inside the saved pipeline)",
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
        "dataset": {"file": path.name, "rows_used": int(len(df)), "split": split_info},
        "test_metrics": {k: test_m[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                "precision_weighted", "recall_weighted", "f1_weighted")},
        "has_predict_proba": hasattr(final_pipe, "predict_proba"),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config.DATA_REPORT_PATH.write_text(json.dumps(data_report, indent=2, default=str), encoding="utf-8")
    config.MODEL_COMPARISON_PATH.write_text(json.dumps(
        [{"model": r["model"], "best_params": r["best_params"], "cv_f1_macro": r["cv_f1_macro_mean"],
          "validation_f1_macro": r["validation"]["f1_macro"], "test_accuracy": r["test"]["accuracy"],
          "test_f1_macro": r["test"]["f1_macro"]} for r in results], indent=2), encoding="utf-8")
    config.EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")
    log(f"[6/9] Saved model + metadata + reports in {time.time() - t0:.1f}s")

    # figures
    try:
        from src.analysis.figures import make_all_figures
        make_all_figures(df, data_report, evaluation)
        log("[7/9] Saved figures to reports/figures/")
    except Exception as exc:  # figures must never break training
        log(f"[7/9] Figure generation skipped: {exc}")
    return {"metadata": metadata, "evaluation": evaluation, "data_report": data_report}


if __name__ == "__main__":
    run_training()
