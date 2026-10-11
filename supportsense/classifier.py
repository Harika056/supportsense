"""Classifier interface and backends.

Backends
- TransformersClassifier: fine-tuned DistilBERT with two heads (8-way category
  classification + sigmoid urgency regression). This is the real model.
- KeywordClassifier: tiny deterministic keyword/rule backend. FOR TESTS AND
  OFFLINE DEMOS ONLY. It makes no accuracy claims.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from .labels import CATEGORIES, ID2LABEL


@dataclass
class Prediction:
    label: str
    confidence: float
    urgency: float
    scores: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class Classifier(ABC):
    name: str = "base"

    @abstractmethod
    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Classify a list of texts. Must return one Prediction per input, in order."""

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]


# --------------------------------------------------------------------------- keyword
_KEYWORDS: dict[str, list[str]] = {
    "billing": ["charge", "charged", "invoice", "billed", "billing", "payment", "credit card", "statement"],
    "login_access": ["log in", "login", "sign in", "password", "locked", "two-factor", "2fa", "username"],
    "bug_report": ["crash", "bug", "error", "freeze", "freezes", "blank screen", "broken"],
    "feature_request": ["feature request", "would be great", "please consider", "i wish", "could you add", "add a"],
    "shipping_delivery": ["shipping", "ship", "delivered", "delivery", "package", "tracking", "courier", "arrived"],
    "refund_return": ["refund", "return", "send back", "money back"],
    "account_settings": ["change my", "update my", "profile", "notifications", "delete my account", "settings", "preferences", "change the email"],
    "general_inquiry": ["hours", "contact", "discount", "store", "partnership", "tell me more", "what are"],
}
_URGENT_WORDS = ["urgent", "immediately", "asap", "critical", "emergency", "blocked", "down", "right now"]
_CALM_WORDS = ["no rush", "whenever", "just curious", "not urgent"]


class KeywordClassifier(Classifier):
    """Deterministic keyword counter. NOT a real model; for tests/offline demos only."""

    name = "keyword"

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> Prediction:
        low = text.lower()
        hits = {c: sum(low.count(k) for k in kws) for c, kws in _KEYWORDS.items()}
        total = sum(hits.values())
        if total == 0:
            hits["general_inquiry"] = 1  # fallback bucket
            total = 1
        # Deterministic tie-break: earlier category in CATEGORIES wins.
        best = max(CATEGORIES, key=lambda c: (hits[c], -CATEGORIES.index(c)))
        scores = {c: round(hits[c] / total, 4) for c in CATEGORIES}
        urgency = 0.5
        if any(w in low for w in _CALM_WORDS):
            urgency = 0.1
        elif any(re.search(rf"\b{re.escape(w)}\b", low) for w in _URGENT_WORDS):
            urgency = 0.9
        if "!" in text:
            urgency = min(1.0, urgency + 0.05)
        return Prediction(best, scores[best], round(urgency, 3), scores)


# --------------------------------------------------------------------------- transformers
class TransformersClassifier(Classifier):
    """DistilBERT encoder + category head + urgency head.

    Loads a directory written by training/train.py (encoder + tokenizer +
    heads.pt + labels.json), or falls back to the base `distilbert-base-uncased`
    with freshly initialised (untrained) heads, which is only useful for plumbing.
    """

    name = "transformers"

    def __init__(self, model_dir: str | None = None, max_length: int = 128, device: str | None = None):
        import torch  # imported lazily: keyword backend must not need torch

        from training.model import MultiTaskDistilBert, load_multitask

        self.torch = torch
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.tokenizer = load_multitask(model_dir or "distilbert-base-uncased")
        self.model.to(self.device).eval()
        self.labels = self.model.labels

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        torch = self.torch
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits, urg = self.model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            urg = urg.cpu().tolist()
        out = []
        for p, u in zip(probs, urg):
            scores = {self.labels[i]: round(v, 4) for i, v in enumerate(p)}
            best = max(scores, key=scores.get)
            out.append(Prediction(best, scores[best], round(float(u), 4), scores))
        return out


def build_classifier(backend: str | None = None, model_dir: str | None = None, max_length: int = 128) -> Classifier:
    backend = (backend or os.getenv("MODEL_BACKEND", "transformers")).lower()
    if backend == "keyword":
        return KeywordClassifier()
    if backend == "transformers":
        return TransformersClassifier(model_dir or os.getenv("MODEL_DIR") or None, max_length)
    raise ValueError(f"Unknown MODEL_BACKEND {backend!r} (use 'keyword' or 'transformers')")
