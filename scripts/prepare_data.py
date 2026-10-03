"""Inspect and clean the raw dataset, writing data/processed/prepared_dataset.csv and reports/data_report.json.

(`train_model.py` runs the same steps; this script lets you run Phase 3-4 on their own.)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config  # noqa: E402
from src.data.load_data import DatasetError, inspect_dataset, load_raw  # noqa: E402
from src.data.preprocess import prepare_dataframe  # noqa: E402

if __name__ == "__main__":
    try:
        path = config.find_dataset()
        raw = load_raw(path)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    report = inspect_dataset(raw, path)
    clean, prep = prepare_dataframe(raw)
    report["preparation"] = prep
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    clean.drop(columns=["text_group"]).to_csv(config.DATA_PROCESSED / "cleaned_dataset.csv", index=False)
    (config.REPORTS_DIR / "data_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Prepared:", prep)
