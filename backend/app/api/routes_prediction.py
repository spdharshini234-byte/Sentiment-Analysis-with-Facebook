from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.prediction import PredictRequest, PredictResponse
from backend.app.services import prediction_service
from src.models.predict import ModelNotAvailable

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        res = prediction_service.predict(req.text)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return res
