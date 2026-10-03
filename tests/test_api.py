import io

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True and body["database_ready"] is True


@pytest.mark.parametrize("path", [
    "/api/dashboard", "/api/sentiment-distribution", "/api/sentiment-trend", "/api/common-words",
    "/api/negative-feedback", "/api/model-performance", "/api/dataset-info", "/api/keyword-sentiment",
    "/api/positive-feedback",
])
def test_get_endpoints_return_json(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_dashboard_numbers_come_from_data(client):
    d = client.get("/api/dashboard").json()
    t = d["totals"]
    assert t["total"] == sum(t["counts"].values())
    dist = client.get("/api/sentiment-distribution").json()
    assert dist["counts"] == t["counts"]
    assert 0 <= d["health"]["score"] <= 100


def test_trend_params(client):
    r = client.get("/api/sentiment-trend?granularity=week&start=2024-01-01&end=2024-03-31").json()
    assert r["available"] and r["granularity"] == "week"
    assert client.get("/api/sentiment-trend?granularity=year").status_code == 422


def test_predict_ok(client):
    r = client.post("/api/predict", json={"text": "Thanks for resolving my issue fast!"})
    assert r.status_code == 200
    b = r.json()
    assert b["sentiment"] in ("negative", "neutral", "positive")
    assert set(b["probabilities"]) == {"negative", "neutral", "positive"}


def test_predict_rejects_empty(client):
    r = client.post("/api/predict", json={"text": "   "})
    assert r.status_code == 422 and "error" in r.json()
    assert "Traceback" not in r.text


def test_analyze_has_explanation(client):
    b = client.post("/api/analyze", json={"text": "Having trouble logging into my account"}).json()
    assert "confidence" in b["explanation"] and isinstance(b["important_words"], list)


def test_batch_detects_column(client):
    csv = "id,comment\n1,Having trouble logging in\n2,Great sale today!\n3,\n"
    r = client.post("/api/batch-predict", files={"file": ("x.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    b = r.json()
    assert b["detected_text_column"] == "comment" and b["analyzed_rows"] == 2 and b["skipped_rows"] == 1
    assert "predicted_sentiment" in b["columns"] and "confidence" in b["columns"]


def test_batch_errors(client):
    r = client.post("/api/batch-predict", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
    assert r.status_code == 400 and "error" in r.json()
    r = client.post("/api/batch-predict", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")})
    assert r.status_code == 400 and "text column" in r.json()["error"]
    r = client.post("/api/batch-predict", files={"file": ("x.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 400
