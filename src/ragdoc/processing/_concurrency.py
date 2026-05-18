"""Bounded-concurrency helpers for async processors and pipelines."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable


def _resolve_semaphore(concurrency: int | asyncio.Semaphore) -> asyncio.Semaphore:
    """Return an ``asyncio.Semaphore`` from an int, or pass through an existing one."""
    if isinstance(concurrency, asyncio.Semaphore):
        return concurrency
    return asyncio.Semaphore(concurrency)


async def _fan_out(
    coros: list[Awaitable[None]],
    concurrency: int | asyncio.Semaphore,
) -> None:
    """Run *coros* with bounded concurrency.

    When *concurrency* is the integer ``1`` (the default for all processors),
    takes a direct ``for`` loop fast-path: no semaphore overhead, no
    ``asyncio.gather``, and sequential ordering is guaranteed.

    For any other value — an int ``> 1`` or a pre-built
    ``asyncio.Semaphore`` — wraps each coroutine with the semaphore and
    fans out via :func:`asyncio.gather`.

    Args:
        coros: Coroutines to run.  Each must return ``None``.
        concurrency: Maximum number of coroutines that may run at the same
            time.  Pass a shared ``asyncio.Semaphore`` to enforce a
            cross-processor budget.
    """
    if isinstance(concurrency, int) and concurrency == 1:
        for coro in coros:
            await coro
        return

    sem = _resolve_semaphore(concurrency)

    async def _guarded(coro: Awaitable[None]) -> None:
        async with sem:
            await coro

    await asyncio.gather(*[_guarded(c) for c in coros])
