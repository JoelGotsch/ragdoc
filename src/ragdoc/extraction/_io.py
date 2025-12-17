"""Private async file-IO helpers shared by the local extraction stores.

One home for the aiofiles read/write pattern used by :class:`~ragdoc.extraction.entity.LocalEntityStore`
and :class:`~ragdoc.extraction.graph_store.LocalGraphStore` — the library is async-only, so local
stores must never block the event loop on file IO.
"""

from __future__ import annotations

from pathlib import Path

import aiofiles


async def read_text(path: Path) -> str:
    """Read *path* via aiofiles (UTF-8) without blocking the event loop."""
    async with aiofiles.open(path, encoding="utf-8") as f:
        return await f.read()


async def write_text(path: Path, content: str) -> None:
    """Write *content* to *path* via aiofiles (UTF-8) without blocking the event loop."""
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)
