"""Loads the saved model once and serves predictions."""
from __future__ import annotations

import threading

from backend.app.database.database import log_prediction
from src import config
from src.models.predict import ModelNotAvailable, SentimentPredictor

_lock = threading.Lock()
_predictor: SentimentPredictor | None = None
_load_error: str | None = None


def get_predictor() -> SentimentPredictor:
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    with _lock:
        if _predictor is None:
            try:
                _predictor = SentimentPredictor()
                _load_error = None
            except ModelNotAvailable as exc:
                _load_error = str(exc)
                raise
            except Exception as exc:  # corrupt pickle / version mismatch
                _load_error = f"Model could not be loaded: {exc}"
                raise ModelNotAvailable(_load_error) from exc
    return _predictor


def model_status() -> dict:
    try:
        p = get_predictor()
        return {"loaded": True, "name": p.model_name, "labels": p.labels}
    except ModelNotAvailable as exc:
        return {"loaded": False, "error": str(exc)}


def confidence_level(conf: float) -> str:
    if conf >= config.CONF_HIGH:
        return "high"
    if conf >= config.CONF_MEDIUM:
        return "medium"
    return "low"


def predict(text: str, source: str = "api") -> dict:
    res = get_predictor().predict_one(text)
    log_prediction(source, text, res["sentiment"], res["confidence"])
    return res


def analyze(text: str) -> dict:
    p = get_predictor()
    res = predict(text, source="analyze")
    res["confidence_level"] = confidence_level(res["confidence"])
    res["important_words"] = p.important_words(text)
    res["known_features"] = p.known_word_count(text)
    res["warning"] = None
    if res["known_features"] == 0:
        res["warning"] = ("None of the words in this comment were seen during training, so the prediction "
                          "falls back to the overall class frequencies.")
    return res
