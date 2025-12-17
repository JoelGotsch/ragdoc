"""Shared fixtures for pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.chunking import Chunk
from ragdoc.document import Document, Heading, Paragraph
from ragdoc.parsing.parser import Parser
from ragdoc.pipeline.stores import SourceState

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

    async def list_source_state(self) -> dict[str, SourceState]:
        state: dict[str, SourceState] = {}
        for chunk in self.stored.values():
            if chunk.source_id is not None:
                state[chunk.source_id] = SourceState(
                    source_hash=chunk.source_hash,
                    content_hash=chunk.content_hash,
                )
        return state


class MemoryDocumentStore:
    """In-memory DocumentStore for testing.

    Stores one Document per source_id and computes content_hash on read for
    list_source_state (cheap enough in tests).
    """

    def __init__(self) -> None:
        self.stored: dict[str, Document] = {}

    async def upsert(self, documents: list[Document]) -> list[str]:
        written: list[str] = []
        for doc in documents:
            if not doc.source_id:
                raise ValueError("MemoryDocumentStore.upsert requires document.source_id")
            self.stored[doc.source_id] = doc
            written.append(doc.source_id)
        return written

    async def delete_by_source(self, source_id: str) -> None:
        self.stored.pop(source_id, None)

    async def get_document(self, source_id: str) -> Document | None:
        return self.stored.get(source_id)

    async def list_source_ids(self) -> set[str]:
        return set(self.stored.keys())

    async def list_source_state(self) -> dict[str, SourceState]:
        return {
            sid: SourceState(source_hash=doc.source_hash or "", content_hash=doc.content_hash())
            for sid, doc in self.stored.items()
        }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_paths() -> list[Path]:
    """Three fake file paths (no real files needed; used with mock parsers)."""
    return [Path(f"doc_{i}.html") for i in range(3)]
