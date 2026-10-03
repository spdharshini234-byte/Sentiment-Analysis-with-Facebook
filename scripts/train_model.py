"""Single command to train:  python scripts/train_model.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.load_data import DatasetError  # noqa: E402
from src.models.train import run_training  # noqa: E402

if __name__ == "__main__":
    try:
        run_training()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
