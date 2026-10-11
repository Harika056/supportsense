"""Dashboard service: POST /ingest (from the consumer) -> broadcast over WebSocket /ws.

State is in memory (recent tickets + per-category counts); restart clears it.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from supportsense.labels import CATEGORIES, URGENCY_THRESHOLD

STATIC = Path(__file__).parent / "static"


class TicketEvent(BaseModel):
    id: str | None = None
    text: str
    label: str
    confidence: float
    urgency: float
    ts: float | None = None


class Hub:
    def __init__(self, history: int = 100):
        self.clients: set[WebSocket] = set()
        self.recent: deque[dict] = deque(maxlen=history)
        self.counts: dict[str, int] = {c: 0 for c in CATEGORIES}
        self.urgent = 0

    def snapshot(self) -> dict:
        return {"type": "snapshot", "counts": self.counts, "urgent": self.urgent,
                "recent": list(self.recent), "urgency_threshold": URGENCY_THRESHOLD}

    async def publish(self, event: dict) -> int:
        event["ts"] = event.get("ts") or time.time()
        event["urgent"] = event["urgency"] >= URGENCY_THRESHOLD
        self.counts[event["label"]] = self.counts.get(event["label"], 0) + 1
        self.urgent += int(event["urgent"])
        self.recent.append(event)
        msg = {"type": "ticket", "ticket": event, "counts": self.counts, "urgent": self.urgent}
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
        return len(self.clients)


def create_app() -> FastAPI:
    app = FastAPI(title="SupportSense dashboard")
    hub = Hub()
    app.state.hub = hub

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stats")
    async def stats():
        return hub.snapshot()

    @app.post("/ingest")
    async def ingest(ev: TicketEvent):
        n = await hub.publish(ev.model_dump())
        return {"ok": True, "clients": n}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.add(websocket)
        try:
            await websocket.send_json(hub.snapshot())
            while True:
                await websocket.receive_text()  # keep-alive; client messages ignored
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    return app
