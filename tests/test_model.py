import json

import pytest

from src import config
from src.models.predict import SentimentPredictor


@pytest.fixture(scope="module")
def predictor():
    return SentimentPredictor()


def test_artifacts_exist():
    for p in (config.PIPELINE_PATH, config.METADATA_PATH, config.LABEL_MAP_PATH):
        assert p.is_file(), f"{p} missing - run scripts/train_model.py"


def test_label_mapping_matches_model(predictor):
    mapping = json.loads(config.LABEL_MAP_PATH.read_text(encoding="utf-8"))["labels"]
    assert sorted(predictor.labels) == sorted(mapping) == sorted(config.LABELS)


def test_prediction_shape(predictor):
    r = predictor.predict_one("Having trouble logging into my account, any fix?")
    assert r["sentiment"] in config.LABELS
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert 0 <= r["confidence"] <= 1
    assert r["cleaned_text"]


def test_batch_prediction_length(predictor):
    out = predictor.predict_many(["good", "bad", "okay then"])
    assert len(out) == 3


def test_important_words_for_known_text(predictor):
    words = predictor.important_words("Having trouble logging into my account")
    assert isinstance(words, list)


def test_metrics_are_saved_and_sane():
    ev = json.loads(config.EVALUATION_PATH.read_text(encoding="utf-8"))
    m = ev["final_test_metrics"]
    assert 0 <= m["f1_macro"] <= 1 and m["n"] > 0
    assert m["f1_macro"] > ev["baseline_test_f1_macro"]
    assert len(ev["models"]) == 4
