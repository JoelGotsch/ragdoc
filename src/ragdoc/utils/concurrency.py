"""Bounded-concurrency helpers for async processors and pipelines.

Public home (Phase 8): moved from ``ragdoc.processing._concurrency`` so that
cross-package consumers (sync pipelines, extraction) import a public name.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def resolve_semaphore(concurrency: int | asyncio.Semaphore) -> asyncio.Semaphore:
    """Return an ``asyncio.Semaphore`` from an int, or pass through an existing one.

    Raises:
        ValueError: If *concurrency* is an int below 1. ``Semaphore(0)`` is valid asyncio but
            grants no permits, so every waiter would block forever.
    """
    if isinstance(concurrency, asyncio.Semaphore):
        return concurrency
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    return asyncio.Semaphore(concurrency)


async def run_ordered(
    factories: Sequence[Callable[[], Awaitable[T]]],
    concurrency: int | asyncio.Semaphore,
) -> list[T]:
    """Run coroutine *factories* with bounded concurrency; results come back in input order.

    This is the library's one fan-out helper (element-level processors, the document
    summarizer's map step, per-split extraction).

    Takes **factories**, not coroutine objects: a coroutine is only created at the moment it is
    going to be awaited, so a failure or a cancellation never leaves already-created coroutines
    un-awaited (``RuntimeWarning: coroutine ... was never awaited``). Bind arguments with
    :func:`functools.partial`::

        results = await run_ordered([partial(summarize, image) for image in images], concurrency=4)

    Fail-fast: the first failure propagates. With ``concurrency == 1`` (int) the work runs
    strictly sequentially in the calling task and the remaining factories are simply never
    called; otherwise every still-running sibling task is cancelled and awaited before the error
    is re-raised, so no LLM call keeps running in the background. The same cleanup runs when the
    caller itself is cancelled. Work that must survive a sibling's failure has to contain its
    own errors inside the factory's coroutine.

    Args:
        factories: Zero-argument callables, each returning the awaitable for one result.
        concurrency: Maximum simultaneous awaits — an ``int`` (private semaphore) or a shared
            ``asyncio.Semaphore`` (a budget across several calls). The int ``1`` runs strictly
            sequentially without a semaphore.

    Returns:
        One result per factory, in the order of *factories*.

    Raises:
        ValueError: If *concurrency* is an int below 1 (checked before any factory is called).
    """
    if isinstance(concurrency, int) and concurrency == 1:
        return [await factory() for factory in factories]

    sem = resolve_semaphore(concurrency)

    async def _guarded(factory: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await factory()

    tasks = [asyncio.ensure_future(_guarded(factory)) for factory in factories]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
