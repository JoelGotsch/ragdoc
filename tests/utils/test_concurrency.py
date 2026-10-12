"""Tests for ``ragdoc.utils.concurrency`` — the one bounded fan-out helper, ``run_ordered``.

``run_ordered`` takes zero-argument coroutine *factories* (not coroutine objects), so a failure
or a cancellation never leaves already-created coroutines un-awaited.
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from collections.abc import Awaitable, Callable
from functools import partial

import pytest

import ragdoc.utils
from ragdoc.utils.concurrency import resolve_semaphore, run_ordered


class _Tracker:
    """Records how many tracked coroutines run at the same time."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def work(self, value: int) -> int:
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0)
        self.running -= 1
        return value


async def _collect_unawaited(run: Callable[[], Awaitable[object]], expected: type[BaseException]) -> list[str]:
    """Await ``run()`` expecting *expected*; return the "never awaited" warnings it left behind."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # No pytest.raises here: its ExceptionInfo would keep the traceback — and with it the
        # frames holding any un-awaited coroutine — alive past gc.collect().
        try:
            await run()
        except expected:
            pass
        else:
            pytest.fail(f"expected {expected.__name__}")
        gc.collect()
    return [str(w.message) for w in caught if "never awaited" in str(w.message)]


def test_resolve_semaphore_int_creates_new():
    assert isinstance(resolve_semaphore(3), asyncio.Semaphore)


def test_resolve_semaphore_passthrough():
    original = asyncio.Semaphore(2)
    assert resolve_semaphore(original) is original


def test_fan_out_is_gone():
    """One helper only: the eager-coroutine ``fan_out`` was replaced by ``run_ordered``."""
    assert not hasattr(ragdoc.utils, "fan_out")
    assert not hasattr(ragdoc.utils.concurrency, "fan_out")
    assert ragdoc.utils.run_ordered is run_ordered


@pytest.mark.anyio
@pytest.mark.parametrize("concurrency", [1, 2, 10])
async def test_run_ordered_returns_results_in_input_order(concurrency: int):
    async def slow_then_fast(n: int) -> int:
        await asyncio.sleep(0.01 * (5 - n))  # later items finish first when run concurrently
        return n

    assert await run_ordered([partial(slow_then_fast, n) for n in range(5)], concurrency) == [0, 1, 2, 3, 4]


@pytest.mark.anyio
async def test_run_ordered_is_generic_over_the_result_type():
    async def label(n: int) -> tuple[int, str]:
        return n, f"item-{n}"

    assert await run_ordered([partial(label, n) for n in range(2)], 2) == [(0, "item-0"), (1, "item-1")]


@pytest.mark.anyio
async def test_run_ordered_empty():
    assert await run_ordered([], 1) == []
    assert await run_ordered([], 4) == []


@pytest.mark.anyio
async def test_run_ordered_sequential_runs_one_at_a_time():
    tracker = _Tracker()
    await run_ordered([partial(tracker.work, n) for n in range(4)], 1)
    assert tracker.peak == 1


@pytest.mark.anyio
async def test_run_ordered_respects_the_cap():
    tracker = _Tracker()
    await run_ordered([partial(tracker.work, n) for n in range(6)], 2)
    assert tracker.peak == 2


@pytest.mark.anyio
async def test_run_ordered_shared_semaphore_bounds_separate_calls():
    shared = asyncio.Semaphore(1)
    tracker = _Tracker()
    await asyncio.gather(
        run_ordered([partial(tracker.work, n) for n in range(3)], shared),
        run_ordered([partial(tracker.work, n) for n in range(3)], shared),
    )
    assert tracker.peak == 1


@pytest.mark.anyio
async def test_run_ordered_sequential_failure_never_creates_the_remaining_coroutines():
    called: list[int] = []

    async def work(n: int) -> int:
        called.append(n)
        if n == 1:
            raise RuntimeError("boom")
        return n

    unawaited = await _collect_unawaited(lambda: run_ordered([partial(work, n) for n in range(4)], 1), RuntimeError)

    assert called == [0, 1]  # fail-fast: nothing after the failure is started
    assert unawaited == []


@pytest.mark.anyio
async def test_run_ordered_concurrent_failure_cancels_running_siblings():
    cancelled: list[int] = []
    finished: list[int] = []

    async def work(n: int) -> int:
        if n == 0:
            await asyncio.sleep(0)
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(n)
            raise
        finished.append(n)
        return n

    unawaited = await _collect_unawaited(lambda: run_ordered([partial(work, n) for n in range(3)], 3), RuntimeError)

    assert sorted(cancelled) == [1, 2]
    assert finished == []
    assert unawaited == []


@pytest.mark.anyio
async def test_run_ordered_concurrent_failure_never_starts_queued_work():
    started: list[int] = []

    async def work(n: int) -> int:
        started.append(n)
        await asyncio.sleep(0)
        if n == 0:
            raise RuntimeError("boom")
        await asyncio.sleep(30)
        return n

    unawaited = await _collect_unawaited(lambda: run_ordered([partial(work, n) for n in range(5)], 2), RuntimeError)

    # Items still waiting for a slot when the cleanup runs are cancelled before their factory is
    # called (the slot freed by the failing item may admit one more before the cleanup).
    assert started[:2] == [0, 1]
    assert 3 not in started
    assert 4 not in started
    assert unawaited == []


@pytest.mark.anyio
async def test_run_ordered_cancellation_of_the_caller_cancels_the_work():
    cancelled: list[int] = []

    async def work(n: int) -> int:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(n)
            raise
        return n

    task = asyncio.ensure_future(run_ordered([partial(work, n) for n in range(2)], 2))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(cancelled) == [0, 1]


@pytest.mark.parametrize("concurrency", [0, -1])
def test_resolve_semaphore_rejects_concurrency_below_one(concurrency: int):
    # Semaphore(0) is valid asyncio but grants no permits: every waiter would block forever.
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        resolve_semaphore(concurrency)


@pytest.mark.anyio
@pytest.mark.parametrize("concurrency", [0, -1])
async def test_run_ordered_rejects_concurrency_below_one_instead_of_hanging(concurrency: int):
    tracker = _Tracker()
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        await asyncio.wait_for(run_ordered([partial(tracker.work, 1)], concurrency), timeout=1)
    assert tracker.peak == 0  # rejected before any factory was called
