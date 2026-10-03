"""Pydantic request/response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., description="A Facebook-style comment", max_length=5000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Comment text must not be empty.")
        return v


class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    explanation: str
    model: str


class WordInfluence(BaseModel):
    word: str
    weight: float
    direction: str


class AnalyzeResponse(PredictResponse):
    confidence_level: str
    important_words: list[WordInfluence]
    known_features: int
    warning: str | None = None


class BatchResponse(BaseModel):
    detected_text_column: str
    total_rows: int
    analyzed_rows: int
    skipped_rows: int
    truncated: bool
    summary: dict[str, Any]
    columns: list[str]
    rows: list[dict[str, Any]]
