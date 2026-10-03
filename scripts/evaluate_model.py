"""Print the saved evaluation summary:  python scripts/evaluate_model.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config  # noqa: E402

if __name__ == "__main__":
    if not config.EVALUATION_PATH.is_file():
        print("ERROR: reports/evaluation.json not found. Run `python scripts/train_model.py` first.")
        sys.exit(1)
    ev = json.loads(config.EVALUATION_PATH.read_text(encoding="utf-8"))
    print(f"Selected model: {ev['selected_model']}  ({ev['selection_rule']})")
    print(f"{'model':30s} {'val F1m':>8s} {'test F1m':>9s} {'test acc':>9s}")
    for r in ev["models"]:
        print(f"{r['model']:30s} {r['validation']['f1_macro']:8.3f} {r['test']['f1_macro']:9.3f} {r['test']['accuracy']:9.3f}")
    print("Overfitting check:", ev["overfitting_check"]["verdict"])
