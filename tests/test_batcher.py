import asyncio
import threading
import time

import pytest

from service.batcher import MicroBatcher


def upper_batch(texts):
    return [t.upper() for t in texts]


async def test_results_return_to_the_right_callers():
    b = MicroBatcher(upper_batch, max_batch_size=4, max_wait_ms=30)
    await b.start()
    try:
        words = [f"item{i}" for i in range(11)]
        results = await asyncio.gather(*(b.submit(w) for w in words))
        assert results == [w.upper() for w in words]
    finally:
        await b.stop()


async def test_batches_never_exceed_max_size_and_cover_all_items():
    b = MicroBatcher(upper_batch, max_batch_size=4, max_wait_ms=50)
    await b.start()
    try:
        await asyncio.gather(*(b.submit(str(i)) for i in range(10)))
        assert max(b.batch_sizes) <= 4
        assert sum(b.batch_sizes) == 10
        assert max(b.batch_sizes) > 1  # requests really were grouped
    finally:
        await b.stop()


async def test_single_request_flushes_after_max_wait():
    b = MicroBatcher(upper_batch, max_batch_size=64, max_wait_ms=20)
    await b.start()
    try:
        t0 = time.perf_counter()
        assert await b.submit("solo") == "SOLO"
        assert time.perf_counter() - t0 < 1.0
        assert b.batch_sizes == [1]
    finally:
        await b.stop()


async def test_batch_size_one_means_no_grouping():
    b = MicroBatcher(upper_batch, max_batch_size=1, max_wait_ms=0)
    await b.start()
    try:
        await asyncio.gather(*(b.submit(str(i)) for i in range(5)))
        assert b.batch_sizes == [1] * 5
    finally:
        await b.stop()


async def test_error_reaches_every_caller_in_the_batch_and_batcher_recovers():
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("model exploded")
        return [t.upper() for t in texts]

    b = MicroBatcher(flaky, max_batch_size=8, max_wait_ms=30)
    await b.start()
    try:
        first = await asyncio.gather(*(b.submit(str(i)) for i in range(3)), return_exceptions=True)
        assert all(isinstance(r, ValueError) for r in first)
        assert await b.submit("ok") == "OK"
    finally:
        await b.stop()


async def test_wrong_result_count_is_reported_as_error():
    b = MicroBatcher(lambda texts: ["only one"], max_batch_size=4, max_wait_ms=20)
    await b.start()
    try:
        results = await asyncio.gather(b.submit("a"), b.submit("b"), return_exceptions=True)
        assert all(isinstance(r, RuntimeError) for r in results)
    finally:
        await b.stop()


async def test_submit_before_start_raises():
    b = MicroBatcher(upper_batch)
    with pytest.raises(RuntimeError):
        await b.submit("x")


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        MicroBatcher(upper_batch, max_batch_size=0)


async def test_cancelled_caller_does_not_break_the_batch():
    b = MicroBatcher(upper_batch, max_batch_size=4, max_wait_ms=50)
    await b.start()
    try:
        doomed = asyncio.create_task(b.submit("gone"))
        await asyncio.sleep(0)  # let it enqueue
        doomed.cancel()
        assert await b.submit("alive") == "ALIVE"
    finally:
        await b.stop()


async def test_on_batch_callback_gets_size_and_duration():
    seen = []
    b = MicroBatcher(upper_batch, max_batch_size=4, max_wait_ms=30, on_batch=lambda n, s: seen.append((n, s)))
    await b.start()
    try:
        await asyncio.gather(b.submit("a"), b.submit("b"))
    finally:
        await b.stop()
    assert sum(n for n, _ in seen) == 2
    assert all(s >= 0 for _, s in seen)


async def test_submit_many_preserves_order():
    b = MicroBatcher(upper_batch, max_batch_size=3, max_wait_ms=20)
    await b.start()
    try:
        assert await b.submit_many(["a", "b", "c", "d", "e"]) == ["A", "B", "C", "D", "E"]
    finally:
        await b.stop()


async def test_stop_does_not_leave_in_flight_or_queued_callers_hanging():
    release = threading.Event()

    def slow(texts):
        release.wait(timeout=5)
        return [t.upper() for t in texts]

    b = MicroBatcher(slow, max_batch_size=2, max_wait_ms=5)
    await b.start()
    in_flight = [asyncio.create_task(b.submit("a")), asyncio.create_task(b.submit("b"))]
    await asyncio.sleep(0.1)  # first batch is now inside the blocked model call
    queued = asyncio.create_task(b.submit("c"))
    await asyncio.sleep(0.01)
    stopper = asyncio.create_task(b.stop())
    results = await asyncio.wait_for(
        asyncio.gather(*in_flight, queued, return_exceptions=True), timeout=3
    )
    release.set()
    await stopper
    assert all(isinstance(r, RuntimeError) for r in results)
