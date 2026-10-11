import asyncio

import pytest
from fastapi.testclient import TestClient

from dashboard.app import Hub, create_app
from service.batcher import MicroBatcher
from streaming.consumer import process_ticket
from streaming.router import CategoryRouter
from supportsense.classifier import KeywordClassifier
from supportsense.labels import URGENCY_THRESHOLD

EVENT = {"id": "t1", "text": "I was charged twice", "label": "billing", "confidence": 0.9, "urgency": 0.2}


def test_ingest_updates_stats_and_flags_urgent():
    with TestClient(create_app()) as c:
        assert c.post("/ingest", json=EVENT).json()["ok"] is True
        assert c.post("/ingest", json={**EVENT, "id": "t2", "urgency": URGENCY_THRESHOLD}).status_code == 200
        stats = c.get("/stats").json()
        assert stats["counts"]["billing"] == 2
        assert stats["urgent"] == 1
        assert [t["urgent"] for t in stats["recent"]] == [False, True]


def test_ingest_validation():
    with TestClient(create_app()) as c:
        assert c.post("/ingest", json={"text": "no label"}).status_code == 422


def test_websocket_gets_snapshot_then_live_tickets():
    with TestClient(create_app()) as c:
        c.post("/ingest", json=EVENT)
        with c.websocket_connect("/ws") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot" and snap["counts"]["billing"] == 1
            c.post("/ingest", json={**EVENT, "id": "t2", "label": "bug_report"})
            msg = ws.receive_json()
            assert msg["type"] == "ticket" and msg["ticket"]["label"] == "bug_report"
            assert msg["counts"]["bug_report"] == 1


def test_index_page_is_served():
    with TestClient(create_app()) as c:
        r = c.get("/")
        assert r.status_code == 200 and "SupportSense" in r.text


async def test_hub_drops_dead_clients():
    class Dead:
        async def send_json(self, _):
            raise RuntimeError("closed")

    hub = Hub()
    hub.clients.add(Dead())
    assert await hub.publish(dict(EVENT)) == 0
    assert hub.clients == set()


async def test_hub_history_is_bounded():
    hub = Hub(history=3)
    for i in range(5):
        await hub.publish({**EVENT, "id": str(i)})
    assert [t["id"] for t in hub.recent] == ["2", "3", "4"]


async def _run_process(push):
    batcher = MicroBatcher(KeywordClassifier().predict_batch, 8, 5)
    await batcher.start()
    router, published = CategoryRouter(), []

    async def publish(ev):
        published.append(ev)

    try:
        event = await process_ticket({"id": "x", "text": "refund my order"}, batcher, router, publish, push)
    finally:
        await batcher.stop()
    return event, router, published


async def test_process_ticket_classifies_routes_publishes_and_pushes():
    pushed = []

    async def push(ev):
        pushed.append(ev)

    event, router, published = await _run_process(push)
    assert event["label"] == "refund_return" and event["id"] == "x"
    assert router.sizes()["refund_return"] == 1
    assert published == [event] == pushed


async def test_dashboard_failure_does_not_stop_the_pipeline():
    async def push(ev):
        raise ConnectionError("dashboard down")

    event, router, published = await _run_process(push)
    assert published == [event]
    assert router.sizes()["refund_return"] == 1
