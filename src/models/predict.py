"""Load the saved pipeline and make predictions (used by the FastAPI backend)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.data.preprocess import clean_text


class ModelNotAvailable(Exception):
    pass


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SentimentPredictor:
    def __init__(self, pipeline_path: Path | None = None, metadata_path: Path | None = None):
        pipeline_path = Path(pipeline_path or config.PIPELINE_PATH)
        metadata_path = Path(metadata_path or config.METADATA_PATH)
        if not pipeline_path.is_file():
            raise ModelNotAvailable(
                "Trained model not found. Run `python scripts/train_model.py` to create models/sentiment_pipeline.pkl."
            )
        self.pipeline = joblib.load(pipeline_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        self.labels = [str(c) for c in self.pipeline.classes_]
        self.model_name = self.metadata.get("model_name", type(self.pipeline[-1]).__name__)

    # ---- probabilities -------------------------------------------------
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(texts)
        # LinearSVM has no probabilities: use a softmax over decision scores (relative confidence).
        return _softmax(self.pipeline.decision_function(texts))

    def predict_many(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        proba = self.predict_proba(texts)
        out = []
        for t, p in zip(texts, proba):
            k = int(np.argmax(p))
            out.append({
                "sentiment": self.labels[k],
                "confidence": float(p[k]),
                "probabilities": {lab: float(p[i]) for i, lab in enumerate(self.labels)},
            })
        return out

    def predict_one(self, text: str) -> dict:
        res = self.predict_many([text])[0]
        res["text"] = text
        res["cleaned_text"] = clean_text(text)
        res["model"] = self.model_name
        res["explanation"] = explanation_sentence(res["sentiment"], res["confidence"])
        return res

    # ---- explanation ---------------------------------------------------
    def important_words(self, text: str, top_k: int = 6) -> list[dict]:
        """Words that pushed the prediction toward the predicted class."""
        pipe = self.pipeline
        clf = pipe[-1]
        X = pipe[:-1].transform([text])
        if X.nnz == 0:
            return []
        names = pipe.named_steps["tfidf"].get_feature_names_out()
        k = int(np.argmax(self.predict_proba([text])[0]))
        if hasattr(clf, "coef_"):
            w = clf.coef_[k] if clf.coef_.shape[0] > 1 else (clf.coef_[0] if k == 1 else -clf.coef_[0])
        elif hasattr(clf, "feature_log_prob_"):
            lp = clf.feature_log_prob_
            others = np.delete(lp, k, axis=0).mean(axis=0)
            w = lp[k] - others
        else:
            return []
        row = X.tocoo()
        contrib = [(names[j], float(v * w[j])) for j, v in zip(row.col, row.data)]
        contrib.sort(key=lambda t: -abs(t[1]))
        return [{"word": n, "weight": round(c, 4), "direction": "supports" if c > 0 else "opposes"}
                for n, c in contrib[:top_k]]

    def known_word_count(self, text: str) -> int:
        X = self.pipeline[:-1].transform([text])
        return int(X.nnz)


def explanation_sentence(label: str, confidence: float) -> str:
    base = f"{label.capitalize()} sentiment detected with {round(confidence * 100)}% confidence."
    if confidence < config.CONF_MEDIUM:
        base += " Confidence is low, so treat this prediction as uncertain."
    return base
