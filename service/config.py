"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name, default): return int(os.getenv(name, default))
def _float(name, default): return float(os.getenv(name, default))


@dataclass
class Settings:
    model_backend: str = field(default_factory=lambda: os.getenv("MODEL_BACKEND", "transformers"))
    model_dir: str | None = field(default_factory=lambda: os.getenv("MODEL_DIR") or None)
    max_length: int = field(default_factory=lambda: _int("MAX_LENGTH", 128))
    max_batch_size: int = field(default_factory=lambda: _int("MAX_BATCH_SIZE", 16))
    max_wait_ms: float = field(default_factory=lambda: _float("MAX_WAIT_MS", 10))
    max_text_chars: int = field(default_factory=lambda: _int("MAX_TEXT_CHARS", 5000))
    max_batch_request: int = field(default_factory=lambda: _int("MAX_BATCH_REQUEST", 256))
