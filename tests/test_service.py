import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import Settings
from supportsense.classifier import Classifier, KeywordClassifier


@pytest.fixture
def client():
    settings = Settings(model_backend="keyword", max_batch_size=8, max_wait_ms=5, max_text_chars=200, max_batch_request=5)
    with TestClient(create_app(settings, classifier=KeywordClassifier())) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["backend"] == "keyword"


def test_classify_single(client):
    r = client.post("/classify", json={"text": "I was charged twice, please refund"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"billing", "refund_return"}
    assert 0 <= body["urgency"] <= 1 and len(body["scores"]) == 8


def test_classify_batch_returns_one_result_per_text_in_order(client):
    r = client.post("/classify/batch", json={"texts": ["password locked out", "app crash error", "store hours"]})
    assert r.status_code == 200
    assert [x["label"] for x in r.json()["results"]] == ["login_access", "bug_report", "general_inquiry"]


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/classify", {"text": "   "}),
        ("/classify", {}),
        ("/classify", {"text": "x" * 201}),
        ("/classify/batch", {"texts": []}),
        ("/classify/batch", {"texts": ["ok", " "]}),
        ("/classify/batch", {"texts": ["a"] * 6}),
        ("/classify/batch", {"texts": ["x" * 201]}),
    ],
)
def test_validation_errors(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


def test_metrics_endpoint_counts_requests_and_batches(client):
    client.post("/classify", json={"text": "refund"})
    client.post("/classify/batch", json={"texts": ["a", "b"]})
    text = client.get("/metrics").text
    assert 'supportsense_requests_total{endpoint="/classify",status="200"} 1.0' in text
    assert "supportsense_batch_size_count" in text
    assert "supportsense_inference_seconds_sum" in text


class Boom(Classifier):
    name = "boom"

    def predict_batch(self, texts):
        raise RuntimeError("no model")


def test_inference_failure_returns_500_and_is_counted():
    settings = Settings(model_backend="keyword", max_wait_ms=1)
    with TestClient(create_app(settings, classifier=Boom())) as c:
        r = c.post("/classify", json={"text": "hello"})
        assert r.status_code == 500
        assert 'status="500"' in c.get("/metrics").text
        assert c.post("/classify/batch", json={"texts": ["a"]}).status_code == 500


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("MAX_WAIT_MS", "2.5")
    s = Settings()
    assert s.max_batch_size == 3 and s.max_wait_ms == 2.5
