#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
