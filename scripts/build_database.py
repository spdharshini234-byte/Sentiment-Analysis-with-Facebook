"""Build the SQLite database from the prepared dataset:  python scripts/build_database.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.app.database.init_db import build_database  # noqa: E402
from src.models.predict import ModelNotAvailable  # noqa: E402

if __name__ == "__main__":
    try:
        n = build_database()
        print(f"Database built with {n} posts.")
    except (FileNotFoundError, ModelNotAvailable) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
