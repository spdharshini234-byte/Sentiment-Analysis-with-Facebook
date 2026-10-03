from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.prediction import AnalyzeResponse, PredictRequest
from backend.app.services import prediction_service
from src.models.predict import ModelNotAvailable

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: PredictRequest):
    """Prediction plus the words that influenced it (comment analyzer)."""
    try:
        return prediction_service.analyze(req.text)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
