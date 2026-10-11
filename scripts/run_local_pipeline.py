"""Run the full pipeline WITHOUT Kafka: synthetic tickets -> micro-batched classifier ->
category queues -> live dashboard. Uses the same process_ticket() as the Kafka consumer.

Start the dashboard first:  uvicorn dashboard.app:create_app --factory --port 8001
Then:                       python -m scripts.run_local_pipeline --rate 5 --count 200
Open http://localhost:8001 to watch tickets arrive.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import httpx

from service.batcher import MicroBatcher
from service.config import Settings
from streaming.consumer import process_ticket
from streaming.router import CategoryRouter
from supportsense.classifier import build_classifier
from supportsense.synthetic import stream_tickets


async def run(dashboard_url: str, rate: float, count: int, seed: int | None) -> None:
    s = Settings()
    clf = build_classifier(s.model_backend, s.model_dir, s.max_length)
    batcher = MicroBatcher(clf.predict_batch, s.max_batch_size, s.max_wait_ms)
    await batcher.start()
    router = CategoryRouter()
    stats = {"posted": 0, "failed": 0}

    async def publish(event: dict) -> None:  # stands in for the Kafka "classified" topic
        return None

    async with httpx.AsyncClient(timeout=2.0) as http:
        async def push(event: dict) -> None:
            try:
                await http.post(f"{dashboard_url}/ingest", json={
                    k: event[k] for k in ("id", "text", "label", "confidence", "urgency")})
                stats["posted"] += 1
            except Exception:
                stats["failed"] += 1
                raise

        tasks, t0, n = [], time.perf_counter(), 0
        for t in stream_tickets(seed):
            n += 1
            ticket = {"id": f"local-{n}", "text": t["text"]}
            tasks.append(asyncio.create_task(process_ticket(ticket, batcher, router, publish, push)))
            if n >= count:
                break
            if rate > 0:
                await asyncio.sleep(1.0 / rate)
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0
    await batcher.stop()
    print(f"backend={clf.name} classified {n} tickets in {elapsed:.1f}s; "
          f"dashboard accepted {stats['posted']}, failed {stats['failed']}")
    print("queue sizes:", router.sizes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dashboard-url", default=os.getenv("DASHBOARD_URL", "http://localhost:8001"))
    ap.add_argument("--rate", type=float, default=5.0, help="tickets per second (0 = as fast as possible)")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(run(a.dashboard_url, a.rate, a.count, a.seed))


if __name__ == "__main__":
    main()
