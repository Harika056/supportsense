"""Dynamic micro-batcher.

Callers `await batcher.submit(text)`. A single worker task collects queued items
and flushes a batch when EITHER `max_batch_size` items are waiting OR
`max_wait_ms` has elapsed since the first item of the batch arrived. The (blocking)
model call runs in a thread executor so the event loop stays responsive.
Each item carries its own Future, so results always return to the right caller;
if the batch function raises, every caller in that batch gets the exception.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence


class MicroBatcher:
    def __init__(
        self,
        batch_fn: Callable[[list[str]], Sequence[Any]],
        max_batch_size: int = 16,
        max_wait_ms: float = 10.0,
        on_batch: Optional[Callable[[int, float], None]] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.on_batch = on_batch  # callback(batch_size, inference_seconds), used for metrics
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._own_executor = executor is None
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.batch_sizes: list[int] = []  # history, handy for tests and debugging

    async def start(self) -> None:
        if self._task is None:
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            # fail anything still waiting rather than leaving callers hanging
            while self._queue is not None and not self._queue.empty():
                _, fut = self._queue.get_nowait()
                if not fut.done():
                    fut.set_exception(RuntimeError("batcher stopped"))
            self._task = None
        if self._own_executor:
            self._executor.shutdown(wait=False)

    async def submit(self, text: str) -> Any:
        if self._task is None or self._queue is None:
            raise RuntimeError("MicroBatcher not started")
        fut = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((text, fut))
        return await fut

    async def submit_many(self, texts: list[str]) -> list[Any]:
        return list(await asyncio.gather(*(self.submit(t) for t in texts)))

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            batch = [first]
            try:
                deadline = loop.time() + self.max_wait
                while len(batch) < self.max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        # still drain anything already queued without waiting
                        while len(batch) < self.max_batch_size and not self._queue.empty():
                            batch.append(self._queue.get_nowait())
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break
                await self._flush(batch, loop)
            except asyncio.CancelledError:
                # Shutting down mid-batch: callers already holding a future must not hang forever.
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(RuntimeError("batcher stopped"))
                raise

    async def _flush(self, batch, loop) -> None:
        texts = [t for t, _ in batch]
        futs = [f for _, f in batch]
        self.batch_sizes.append(len(batch))
        t0 = time.perf_counter()
        try:
            results = await loop.run_in_executor(self._executor, self.batch_fn, texts)
            if len(results) != len(texts):
                raise RuntimeError(f"batch_fn returned {len(results)} results for {len(texts)} inputs")
        except Exception as exc:  # propagate to every caller in this batch
            for f in futs:
                if not f.done():
                    f.set_exception(exc)
            return
        finally:
            if self.on_batch:
                self.on_batch(len(batch), time.perf_counter() - t0)
        for f, r in zip(futs, results):
            if not f.done():  # caller may have been cancelled
                f.set_result(r)
