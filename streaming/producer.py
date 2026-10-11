"""Send SYNTHETIC tickets to Kafka topic tickets.raw at a configurable rate.

    python -m streaming.producer --rate 5 --count 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid

from supportsense.synthetic import stream_tickets


async def run(bootstrap: str, topic: str, rate: float, count: int | None, seed: int | None) -> None:
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    interval = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    try:
        for t in stream_tickets(seed):
            msg = {"id": uuid.uuid4().hex[:12], "text": t["text"]}  # true label deliberately not sent
            await producer.send_and_wait(topic, json.dumps(msg).encode())
            sent += 1
            if sent % 20 == 0:
                print(f"sent {sent}")
            if count and sent >= count:
                break
            if interval:
                await asyncio.sleep(interval)
    finally:
        await producer.stop()
    print(f"done, sent {sent} tickets to {topic}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--topic", default=os.getenv("RAW_TOPIC", "tickets.raw"))
    ap.add_argument("--rate", type=float, default=5.0, help="tickets per second (0 = as fast as possible)")
    ap.add_argument("--count", type=int, default=None, help="stop after N tickets (default: run forever)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(run(a.bootstrap, a.topic, a.rate, a.count, a.seed))


if __name__ == "__main__":
    main()
