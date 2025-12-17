"""Unit tests for store source-state and the DocumentStore implementations (Phase 2).

Covers:
- VectorStore.list_source_state() (via MemoryVectorStore) — source_id -> SourceState.
- LocalDocumentStore: upsert/get_document round-trip, list_source_state (cheap, no full
  doc load), list_source_ids, delete_by_source, path-separator source_ids, re-upsert.

LocalDocumentStore does real file I/O against tmp_path — deterministic, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.chunking import Chunk
from ragdoc.document import Document
from ragdoc.pipeline.stores import SourceState

from .conftest import MemoryVectorStore, make_document

# ---------------------------------------------------------------------------
# VectorStore.list_source_state
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_vectorstore_list_source_state_reports_both_hashes():
    store = MemoryVectorStore()
    await store.upsert(
        [
            Chunk(
                prompt_content="p",
                embedding_content="e",
                source_id="a.pdf",
                source_hash="hash-a",
                content_hash="content-a",
            )
        ]
    )
    state = await store.list_source_state()
    assert state == {"a.pdf": SourceState(source_hash="hash-a", content_hash="content-a")}


@pytest.mark.anyio
async def test_vectorstore_list_source_state_empty():
    assert await MemoryVectorStore().list_source_state() == {}


# ---------------------------------------------------------------------------
# LocalDocumentStore
# ---------------------------------------------------------------------------


def _doc(source_id: str, source_hash: str, body: str = "Body.") -> Document:
    doc = make_document(title="T", body=body)
    doc.source_id = source_id
    doc.source_hash = source_hash
    return doc


@pytest.mark.anyio
async def test_local_document_store_round_trips_document(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    doc = _doc("report.pdf", "h1", body="Distinctive content here.")
    await store.upsert([doc])

    loaded = await store.get_document("report.pdf")
    assert loaded is not None
    assert loaded.source_id == "report.pdf"
    assert loaded.source_hash == "h1"
    assert loaded.content_hash() == doc.content_hash()
    assert [type(e).__name__ for e in loaded.elements] == [type(e).__name__ for e in doc.elements]


@pytest.mark.anyio
async def test_local_document_store_get_unknown_returns_none(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    assert await LocalDocumentStore(tmp_path).get_document("missing") is None


@pytest.mark.anyio
async def test_local_document_store_list_source_state(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    d1 = _doc("a.pdf", "ha")
    d2 = _doc("b.pdf", "hb")
    await store.upsert([d1, d2])

    state = await store.list_source_state()
    assert set(state) == {"a.pdf", "b.pdf"}
    assert state["a.pdf"] == SourceState(source_hash="ha", content_hash=d1.content_hash())
    assert state["b.pdf"] == SourceState(source_hash="hb", content_hash=d2.content_hash())


@pytest.mark.anyio
async def test_local_document_store_list_source_ids(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    await store.upsert([_doc("a.pdf", "ha"), _doc("b.pdf", "hb")])
    assert await store.list_source_ids() == {"a.pdf", "b.pdf"}


@pytest.mark.anyio
async def test_local_document_store_delete_by_source(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    await store.upsert([_doc("a.pdf", "ha"), _doc("b.pdf", "hb")])
    await store.delete_by_source("a.pdf")

    assert await store.get_document("a.pdf") is None
    assert await store.list_source_ids() == {"b.pdf"}
    assert "a.pdf" not in await store.list_source_state()


@pytest.mark.anyio
async def test_local_document_store_handles_path_separator_source_ids(tmp_path: Path):
    """source_id may be a relative path (e.g. multi-dir strategy)."""
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    sid = "sub/dir/report.pdf"
    await store.upsert([_doc(sid, "h")])
    loaded = await store.get_document(sid)
    assert loaded is not None and loaded.source_id == sid


@pytest.mark.anyio
async def test_local_document_store_reupsert_updates(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    await store.upsert([_doc("a.pdf", "ha", body="Original.")])
    await store.upsert([_doc("a.pdf", "hb", body="Updated content.")])

    loaded = await store.get_document("a.pdf")
    assert loaded is not None and loaded.source_hash == "hb"
    state = await store.list_source_state()
    assert state["a.pdf"].source_hash == "hb"
    assert state["a.pdf"].content_hash == loaded.content_hash()
    # still a single source
    assert await store.list_source_ids() == {"a.pdf"}


@pytest.mark.anyio
async def test_local_document_store_upsert_requires_source_id(tmp_path: Path):
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    doc = make_document()  # no source_id
    with pytest.raises(ValueError):
        await LocalDocumentStore(tmp_path).upsert([doc])


@pytest.mark.anyio
async def test_local_document_store_detects_manual_file_edit(tmp_path: Path):
    """A human editing the stored JSON changes content_hash (no stale cached index)."""
    from ragdoc.pipeline.local_document_store import LocalDocumentStore

    store = LocalDocumentStore(tmp_path)
    doc = _doc("a.pdf", "h", body="Original body.")
    await store.upsert([doc])
    before = (await store.list_source_state())["a.pdf"].content_hash

    # Simulate a manual edit: load, change content, write back to the same file.
    loaded = await store.get_document("a.pdf")
    assert loaded is not None
    loaded.elements[-1].html = "<p>Edited body, totally different.</p>"
    store._doc_path("a.pdf").write_text(loaded.model_dump_json(indent=2), encoding="utf-8")

    after = (await store.list_source_state())["a.pdf"].content_hash
    assert after != before
