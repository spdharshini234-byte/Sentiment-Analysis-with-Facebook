"""Read-only analytics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.database import database as db
from backend.app.services import analytics_service as svc
from backend.app.services import prediction_service
from src import config

router = APIRouter(prefix="/api", tags=["analytics"])


def _guard(fn, *args):
    try:
        return fn(*args)
    except svc.ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/health")
def health():
    model = prediction_service.model_status()
    ready = db.database_ready()
    return {
        "status": "ok" if (model["loaded"] and ready) else "degraded",
        "model_loaded": model["loaded"], "model_name": model.get("name"), "model_error": model.get("error"),
        "database_ready": ready,
        "dataset_present": config.find_dataset() is not None,
    }


@router.get("/dashboard")
def dashboard():
    return _guard(svc.dashboard)


@router.get("/sentiment-distribution")
def sentiment_distribution():
    return _guard(svc.sentiment_distribution)


@router.get("/sentiment-trend")
def sentiment_trend(granularity: str = Query("month", pattern="^(day|week|month)$"),
                    start: str | None = Query(None, description="YYYY-MM-DD"),
                    end: str | None = Query(None, description="YYYY-MM-DD")):
    return _guard(svc.sentiment_trend, granularity, start, end)


@router.get("/common-words")
def common_words(limit: int = Query(15, ge=1, le=50),
                 sentiment: str | None = Query(None, pattern="^(negative|neutral|positive)$")):
    return _guard(svc.common_words, limit, sentiment)


@router.get("/keyword-sentiment")
def keyword_sentiment(top_n: int = Query(15, ge=1, le=40), min_count: int = Query(30, ge=1, le=1000)):
    return _guard(svc.keyword_sentiment, top_n, min_count)


@router.get("/negative-feedback")
def negative_feedback(limit: int = Query(20, ge=1, le=100)):
    return _guard(svc.negative_feedback, limit)


@router.get("/positive-feedback")
def positive_feedback(limit: int = Query(20, ge=1, le=100)):
    return _guard(svc.positive_feedback, limit)


@router.get("/model-performance")
def model_performance():
    return _guard(svc.model_performance)


@router.get("/dataset-info")
def dataset_info():
    return _guard(svc.dataset_info)
