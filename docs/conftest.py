"""pytest configuration for the notebook test layer (``uv run pytest docs/notebooks/``).

marimo exposes each ``test_*`` cell as a module-level callable that runs the cell's
upstream cells and then the cell itself. When any upstream cell is ``async`` (most
ragdoc notebooks — the library is async-only) that callable returns an *awaitable*,
which plain pytest refuses to run ("async def functions are not natively supported").
The callable is a ``marimo.Cell`` object rather than a coroutine function, so async
pytest plugins (anyio, pytest-asyncio) do not recognise it either. The hook below
closes the gap: it calls the test cell and, if the result is awaitable, drives it to
completion on a fresh asyncio event loop.

This file lives in ``docs/`` (not ``docs/notebooks/``) on purpose: every ``*.py`` in
``docs/notebooks/`` is treated as a marimo notebook by ``make html`` and ``marimo check``.
"""

import asyncio
import inspect
from collections.abc import Awaitable

import pytest


async def _await(awaitable: Awaitable[object]) -> object:
    return await awaitable


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    """Run a notebook test cell, awaiting it when its upstream cells are async."""
    argnames = pyfuncitem._fixtureinfo.argnames  # pyright: ignore[reportPrivateUsage]
    result = pyfuncitem.obj(**{name: pyfuncitem.funcargs[name] for name in argnames})
    if inspect.isawaitable(result):
        asyncio.run(_await(result))
    return True
