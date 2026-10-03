"""FastAPI application entry point.

Run from the project root:  uvicorn backend.app.main:app --reload
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from backend.app.api import routes_analysis, routes_batch, routes_dashboard, routes_prediction  # noqa: E402
from backend.app.database import database as db  # noqa: E402
from backend.app.database import init_db  # noqa: E402
from backend.app.services import analytics_service, prediction_service  # noqa: E402
from src import config  # noqa: E402

log = logging.getLogger("fsi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure the analytics database exists (build it from the prepared data if it was not committed).
    if not db.database_ready():
        try:
            n = init_db.build_database()
            analytics_service.reset_cache()
            log.info("Database was missing; built it with %s posts.", n)
        except Exception as exc:
            log.warning("Database not available: %s", exc)
    try:
        prediction_service.get_predictor()
    except Exception as exc:
        log.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Facebook Sentiment Intelligence API",
    description="Facebook comment sentiment analysis using a real-world Facebook comment dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(_: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    msg = str(first.get("msg", "Invalid request")).replace("Value error, ", "")
    return JSONResponse(status_code=422, content={"error": msg, "field": ".".join(str(p) for p in first.get("loc", []))})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)  # stack trace goes to the server log only
    return JSONResponse(status_code=500, content={"error": "Something went wrong on the server. Please try again."})


for r in (routes_dashboard.router, routes_prediction.router, routes_analysis.router, routes_batch.router):
    app.include_router(r)

# Serve the built React UI (frontend/dist) from the same server when it exists.
FRONTEND_DIST = ROOT / "frontend" / "dist"
if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"name": app.title, "docs": "/docs", "health": "/api/health",
                "note": "Frontend not built. Run the Vite dev server (npm run dev) or build it (npm run build)."}
