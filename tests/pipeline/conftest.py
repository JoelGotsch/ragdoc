"""Shared fixtures for pipeline tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.chunking import Chunk
from ragdoc.document import Document, Heading, Paragraph
from ragdoc.parsing.parser import Parser


# ---------------------------------------------------------------------------
# Document factories
# ---------------------------------------------------------------------------


def make_document(title: str = "Doc", body: str = "Content.") -> Document:
    """Create a minimal Document with one heading and one paragraph."""
    return Document(
        title=title,
        elements=[
            Heading(innerhtml=title, level=1),
            Paragraph(html_content=f"<p>{body}</p>"),
        ],
    )


# ---------------------------------------------------------------------------
# Parser fixtures (async — the library is async-only)
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_parser() -> Parser:
    """A parser that ignores the path and returns a fixed Document."""

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body="Parsed content.")

    return _parse


@pytest.fixture
def error_parser() -> Parser:
    """A parser that always raises ValueError."""

    async def _parse(path: Path) -> Document:
        raise ValueError(f"Cannot parse {path.name}")

    return _parse


# ---------------------------------------------------------------------------
# In-memory VectorStore (shared across test modules)
# ---------------------------------------------------------------------------


class MemoryVectorStore:
    """In-memory VectorStore for testing.

    Implements the typed VectorStore protocol using chunk.source_id and
    chunk.source_hash (first-class fields, not chunk.metadata).
    """

    def __init__(self) -> None:
        self.stored: dict[str, Chunk] = {}
        self.upsert_calls: list[list[Chunk]] = []
        self.delete_calls: list[list[str]] = []

    async def upsert(self, chunks: list[Chunk]) -> list[str]:
        self.upsert_calls.append(list(chunks))
        for c in chunks:
            self.stored[c.id] = c
        return [c.id for c in chunks]

    async def delete(self, ids: list[str]) -> None:
        self.delete_calls.append(list(ids))
        for id_ in ids:
            self.stored.pop(id_, None)

    async def get_source_hash(self, source_id: str) -> str | None:
        for chunk in self.stored.values():
            if chunk.source_id == source_id:
                return chunk.source_hash
        return None

    async def delete_by_source(self, source_id: str) -> None:
        to_delete = [cid for cid, chunk in self.stored.items() if chunk.source_id == source_id]
        for cid in to_delete:
            del self.stored[cid]

    async def list_source_ids(self) -> set[str]:
        return {chunk.source_id for chunk in self.stored.values() if chunk.source_id is not None}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_paths() -> list[Path]:
    """Three fake file paths (no real files needed; used with mock parsers)."""
    return [Path(f"doc_{i}.html") for i in range(3)]
