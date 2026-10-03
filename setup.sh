#!/usr/bin/env bash
# One-time setup for Linux / macOS / GitHub Codespaces.
set -e
cd "$(dirname "$0")"
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" || { echo "Python 3.11 or newer is required."; exit 1; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
[ -f models/sentiment_pipeline.pkl ] || .venv/bin/python scripts/train_model.py
[ -f data/processed/sentiment.db ] || .venv/bin/python scripts/build_database.py
echo "Setup finished. Run ./run.sh to start the application."
