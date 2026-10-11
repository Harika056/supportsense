"""In-memory per-category queues (bounded deques).

Deliberately simple: this stands in for "route to the right team's queue".
A production system would use Redis lists/streams or per-category Kafka topics;
see README. State is lost on restart.
"""
from __future__ import annotations

from collections import defaultdict, deque

from supportsense.labels import CATEGORIES


class CategoryRouter:
    def __init__(self, maxlen: int = 1000):
        self.queues: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        for c in CATEGORIES:
            self.queues[c]  # pre-create

    def route(self, ticket: dict) -> str:
        self.queues[ticket["label"]].append(ticket)
        return ticket["label"]

    def sizes(self) -> dict[str, int]:
        return {c: len(q) for c, q in self.queues.items()}

    def pop(self, category: str):
        q = self.queues[category]
        return q.popleft() if q else None
