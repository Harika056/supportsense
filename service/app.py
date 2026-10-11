"""Inference service: /classify, /classify/batch, /health, /metrics."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from supportsense.classifier import Classifier, build_classifier
from service.batcher import MicroBatcher
from service.config import Settings


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Ticket text")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)

    @field_validator("texts")
    @classmethod
    def not_blank(cls, v: list[str]) -> list[str]:
        if any(not t.strip() for t in v):
            raise ValueError("texts must not contain blank entries")
        return v


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    urgency: float
    scores: dict[str, float]


class BatchResponse(BaseModel):
    results: list[ClassifyResponse]


def create_app(settings: Settings | None = None, classifier: Classifier | None = None) -> FastAPI:
    settings = settings or Settings()
    registry = CollectorRegistry()  # per-app registry: safe to create many apps in tests
    requests_total = Counter("supportsense_requests_total", "HTTP requests", ["endpoint", "status"], registry=registry)
    batch_size_h = Histogram("supportsense_batch_size", "Items per model batch",
                             buckets=(1, 2, 4, 8, 16, 32, 64, 128), registry=registry)
    latency_h = Histogram("supportsense_inference_seconds", "Model inference latency per batch (seconds)",
                          buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5), registry=registry)

    def on_batch(size: int, seconds: float) -> None:
        batch_size_h.observe(size)
        latency_h.observe(seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clf = classifier or build_classifier(settings.model_backend, settings.model_dir, settings.max_length)
        batcher = MicroBatcher(clf.predict_batch, settings.max_batch_size, settings.max_wait_ms, on_batch=on_batch)
        await batcher.start()
        app.state.classifier, app.state.batcher = clf, batcher
        yield
        await batcher.stop()

    app = FastAPI(title="SupportSense inference service", lifespan=lifespan)
    app.state.settings = settings

    def check_len(texts: list[str]) -> None:
        for t in texts:
            if len(t) > settings.max_text_chars:
                raise HTTPException(422, f"text exceeds MAX_TEXT_CHARS={settings.max_text_chars}")

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest):
        check_len([req.text])
        try:
            pred = await app.state.batcher.submit(req.text)
        except Exception as exc:
            requests_total.labels("/classify", "500").inc()
            raise HTTPException(500, f"inference failed: {exc}") from exc
        requests_total.labels("/classify", "200").inc()
        return pred.to_dict()

    @app.post("/classify/batch", response_model=BatchResponse)
    async def classify_batch(req: BatchRequest):
        if len(req.texts) > settings.max_batch_request:
            raise HTTPException(422, f"at most {settings.max_batch_request} texts per request")
        check_len(req.texts)
        try:
            preds = await app.state.batcher.submit_many(req.texts)
        except Exception as exc:
            requests_total.labels("/classify/batch", "500").inc()
            raise HTTPException(500, f"inference failed: {exc}") from exc
        requests_total.labels("/classify/batch", "200").inc()
        return {"results": [p.to_dict() for p in preds]}

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": app.state.classifier.name,
                "max_batch_size": settings.max_batch_size, "max_wait_ms": settings.max_wait_ms}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


