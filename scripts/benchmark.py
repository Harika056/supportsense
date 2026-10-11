"""Measure what micro-batching buys on YOUR machine.

Sends the same tickets through the classifier twice: one at a time (batch size 1) and with
micro-batching, and prints tickets/second plus p50/p95 per-ticket latency.

    set MODEL_BACKEND=transformers
    set MODEL_DIR=artifacts\\model
    python -m scripts.benchmark --tickets 512 --concurrency 64 --batch-size 16

Only numbers printed by a run with the real model on your machine belong on a resume.
The keyword backend is not a model; its numbers say nothing about batching a neural net.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from service.batcher import MicroBatcher
from service.config import Settings
from supportsense.classifier import build_classifier
from supportsense.synthetic import generate_tickets


async def measure(clf, texts: list[str], batch_size: int, max_wait_ms: float, concurrency: int) -> dict:
    batcher = MicroBatcher(clf.predict_batch, batch_size, max_wait_ms)
    await batcher.start()
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def one(text: str) -> None:
        async with sem:
            t0 = time.perf_counter()
            await batcher.submit(text)
            latencies.append((time.perf_counter() - t0) * 1000)

    await asyncio.gather(*(one(t) for t in texts[:concurrency]))  # warm-up, not measured
    latencies.clear()
    t0 = time.perf_counter()
    await asyncio.gather(*(one(t) for t in texts))
    elapsed = time.perf_counter() - t0
    sizes = batcher.batch_sizes[-max(1, len(texts) // max(1, batch_size)):]
    await batcher.stop()
    lat = sorted(latencies)
    return {
        "tickets_per_sec": len(texts) / elapsed,
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[max(0, int(len(lat) * 0.95) - 1)],
        "avg_batch": sum(sizes) / len(sizes),
    }


async def main_async(a) -> None:
    s = Settings()
    clf = build_classifier(s.model_backend, s.model_dir, s.max_length)
    texts = [t["text"] for t in generate_tickets(a.tickets, seed=1)]
    print(f"backend: {clf.name}   tickets: {a.tickets}   concurrency: {a.concurrency}")
    single = await measure(clf, texts, 1, 0.0, a.concurrency)
    batched = await measure(clf, texts, a.batch_size, a.max_wait_ms, a.concurrency)
    print(f"{'mode':<22}{'tickets/s':>11}{'p50 ms':>10}{'p95 ms':>10}{'avg batch':>11}")
    for name, r in (("one at a time", single), (f"micro-batch (max {a.batch_size})", batched)):
        print(f"{name:<22}{r['tickets_per_sec']:>11.1f}{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}{r['avg_batch']:>11.1f}")
    print(f"speed-up: {batched['tickets_per_sec'] / single['tickets_per_sec']:.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-wait-ms", type=float, default=10.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
