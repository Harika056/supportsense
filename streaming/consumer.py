"""Kafka consumer: tickets.raw -> classify (in-process micro-batcher) -> tickets.classified,
per-category routing, and push to the dashboard.

Records are pulled in chunks with getmany() and classified concurrently, so the
micro-batcher sees many in-flight requests and forms real batches.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Awaitable, Callable

import httpx

from service.batcher import MicroBatcher
from service.config import Settings
from streaming.router import CategoryRouter
from supportsense.classifier import build_classifier

log = logging.getLogger("consumer")


async def process_ticket(
    ticket: dict,
    batcher: MicroBatcher,
    router: CategoryRouter,
    publish: Callable[[dict], Awaitable[None]],
    push_dashboard: Callable[[dict], Awaitable[None]],
) -> dict:
    """Classify one raw ticket {id, text}; returns the enriched event. Kafka-free so it is unit-testable."""
    pred = await batcher.submit(ticket["text"])
    event = {"id": ticket.get("id"), "text": ticket["text"], **pred.to_dict()}
    router.route(event)
    await publish(event)
    try:
        await push_dashboard(event)
    except Exception as exc:  # dashboard being down must not stop the pipeline
        log.warning("dashboard push failed: %s", exc)
    return event


async def run(args) -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    s = Settings()
    clf = build_classifier(s.model_backend, s.model_dir, s.max_length)
    batcher = MicroBatcher(clf.predict_batch, s.max_batch_size, s.max_wait_ms)
    await batcher.start()
    router = CategoryRouter()

    consumer = AIOKafkaConsumer(args.raw_topic, bootstrap_servers=args.bootstrap, group_id=args.group,
                                auto_offset_reset="earliest", enable_auto_commit=True)
    producer = AIOKafkaProducer(bootstrap_servers=args.bootstrap)
    await consumer.start()
    await producer.start()
    http = httpx.AsyncClient(timeout=2.0)

    async def publish(event: dict) -> None:
        await producer.send(args.out_topic, json.dumps(event).encode(), key=(event["label"]).encode())

    async def push(event: dict) -> None:
        await http.post(f"{args.dashboard_url}/ingest", json={
            k: event[k] for k in ("id", "text", "label", "confidence", "urgency")})

    done = 0
    try:
        while True:
            batches = await consumer.getmany(timeout_ms=200, max_records=256)
            raw = [json.loads(m.value) for msgs in batches.values() for m in msgs]
            if not raw:
                continue
            await asyncio.gather(*(process_ticket(t, batcher, router, publish, push) for t in raw))
            done += len(raw)
            log.info("classified %d total (last chunk %d, batch sizes tail %s) queues=%s",
                     done, len(raw), batcher.batch_sizes[-3:], router.sizes())
    finally:
        await consumer.stop(); await producer.stop(); await http.aclose(); await batcher.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--raw-topic", default=os.getenv("RAW_TOPIC", "tickets.raw"))
    ap.add_argument("--out-topic", default=os.getenv("CLASSIFIED_TOPIC", "tickets.classified"))
    ap.add_argument("--group", default=os.getenv("CONSUMER_GROUP", "supportsense"))
    ap.add_argument("--dashboard-url", default=os.getenv("DASHBOARD_URL", "http://localhost:8001"))
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
