from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.database.database import log_prediction
from backend.app.schemas.prediction import BatchResponse
from backend.app.services import prediction_service
from src.config import LABELS
from src.data.preprocess import clean_text
from src.models.predict import ModelNotAvailable

router = APIRouter(prefix="/api", tags=["batch"])

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "5000"))
NAME_HINTS = ["text", "comment", "comments", "message", "content", "review", "post", "body", "tweet", "sentence"]


def detect_text_column(df: pd.DataFrame) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for hint in NAME_HINTS:
        if hint in lower:
            return lower[hint]
    best, best_len = None, 0.0
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            s = df[c].dropna().astype(str)
            if len(s) and s.str.len().mean() > best_len:
                best, best_len = c, float(s.str.len().mean())
    return best if best_len >= 8 else None


def _read_csv(raw: bytes) -> pd.DataFrame:
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as exc:
            last = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise HTTPException(400, f"The file is not a valid CSV: {exc}")
    raise HTTPException(400, f"Could not decode the CSV file: {last}")


@router.post("/batch-predict", response_model=BatchResponse)
async def batch_predict(file: UploadFile = File(...), text_column: str | None = Form(None)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File too large. The limit is {MAX_BYTES // (1024 * 1024)} MB.")
    df = _read_csv(raw)
    if df.empty or len(df.columns) == 0:
        raise HTTPException(400, "The CSV has no rows.")
    col = text_column if text_column in df.columns else detect_text_column(df)
    if col is None:
        raise HTTPException(400, f"Could not find a text column. Columns found: {', '.join(map(str, df.columns))}. "
                                 "Rename the comment column to 'text' or 'comment' and try again.")
    truncated = len(df) > MAX_ROWS
    df = df.head(MAX_ROWS).copy()
    total_rows = int(len(df))
    texts = df[col].fillna("").astype(str)
    valid = texts.str.strip().str.len() > 0
    try:
        predictor = prediction_service.get_predictor()
    except ModelNotAvailable as exc:
        raise HTTPException(503, str(exc))
    preds = predictor.predict_many(texts[valid].tolist())
    out = df.copy()
    out["cleaned_text"] = [clean_text(t) for t in texts]
    out["predicted_sentiment"] = None
    out["confidence"] = np.nan
    for lab in LABELS:
        out[f"prob_{lab}"] = np.nan
    idx = out.index[valid]
    out.loc[idx, "predicted_sentiment"] = [p["sentiment"] for p in preds]
    out.loc[idx, "confidence"] = [round(p["confidence"], 4) for p in preds]
    for lab in LABELS:
        out.loc[idx, f"prob_{lab}"] = [round(p["probabilities"][lab], 4) for p in preds]
    counts = {k: int(sum(1 for p in preds if p["sentiment"] == k)) for k in LABELS}
    n_ok = int(valid.sum())
    if n_ok:
        log_prediction("batch", f"{file.filename} ({n_ok} rows)", max(counts, key=counts.get), float(np.mean([p["confidence"] for p in preds])))
    out = out.astype(object).where(out.notna(), None)
    return {
        "detected_text_column": str(col), "total_rows": total_rows, "analyzed_rows": n_ok,
        "skipped_rows": total_rows - n_ok, "truncated": truncated,
        "summary": {"counts": counts,
                    "percentages": {k: round(100 * v / n_ok, 2) if n_ok else 0.0 for k, v in counts.items()},
                    "mean_confidence": round(float(np.mean([p["confidence"] for p in preds])), 4) if n_ok else None},
        "columns": [str(c) for c in out.columns], "rows": out.to_dict(orient="records"),
    }
