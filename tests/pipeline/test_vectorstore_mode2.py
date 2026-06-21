"""Tests for VectorStorePipeline Mode 2 (DocumentStore -> VectorStore, Boundary 2).

Covers:
- run(source_ids=None) chunks every document in the doc store into the vector store.
- unchanged document (same content_hash) is skipped on re-run.
- editing a document in the store (new content_hash) triggers re-chunk.
- source_ids subset limits which documents are processed.
- orphan = source_id in vec store but absent from doc store (only with delete_orphans=True).
- chunks carry content_hash (Boundary-2 change token) and the doc's source_id.
"""

from __future__ import annotations

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline

from .conftest import MemoryDocumentStore, MemoryVectorStore, make_document


def make_mode2_pipeline(doc_store, vec_store) -> VectorStorePipeline:
    # No parser needed — Mode 2 calls pipeline.chunk_document(doc). processors=[] (docs are
    # already processed at Boundary 1).
    return VectorStorePipeline(
        pipeline=DocumentPipeline(processors=[]),
        vector_store=vec_store,
        document_store=doc_store,
    )


async def _seed(doc_store: MemoryDocumentStore, sid: str, body: str) -> Document:
    doc = make_document(title=sid, body=body)
    doc.source_id = sid
    doc.source_hash = f"filehash::{sid}"
    await doc_store.upsert([doc])
    return doc


@pytest.fixture
def vec() -> MemoryVectorStore:
    return MemoryVectorStore()


@pytest.fixture
def docs() -> MemoryDocumentStore:
    return MemoryDocumentStore()


@pytest.mark.anyio
async def test_run_none_chunks_all_documents(docs, vec):
    await _seed(docs, "a.pdf", "Alpha body")
    await _seed(docs, "b.pdf", "Beta body")

    result = await make_mode2_pipeline(docs, vec).run()  # source_ids=None → all
    assert set(result.processed) == {"a.pdf", "b.pdf"}
    assert {c.source_id for c in vec.stored.values()} == {"a.pdf", "b.pdf"}


@pytest.mark.anyio
async def test_chunks_carry_content_hash_and_source_id(docs, vec):
    doc = await _seed(docs, "a.pdf", "Some content")
    await make_mode2_pipeline(docs, vec).run()
    for chunk in vec.stored.values():
        assert chunk.source_id == "a.pdf"
        assert chunk.content_hash == doc.content_hash()
        assert chunk.source_hash == "filehash::a.pdf"  # original file hash carried from B1


@pytest.mark.anyio
async def test_unchanged_document_skipped_on_rerun(docs, vec):
    await _seed(docs, "a.pdf", "Stable")
    pipeline = make_mode2_pipeline(docs, vec)
    await pipeline.run()
    result = await pipeline.run()
    assert result.skipped == ["a.pdf"] and result.processed == []


@pytest.mark.anyio
async def test_edited_document_is_rechunked(docs, vec):
    await _seed(docs, "a.pdf", "Original content")
    pipeline = make_mode2_pipeline(docs, vec)
    await pipeline.run()

    await _seed(docs, "a.pdf", "Edited, entirely different content")  # new content_hash
    result = await pipeline.run()
    assert result.processed == ["a.pdf"] and result.skipped == []


@pytest.mark.anyio
async def test_source_ids_subset_limits_processing(docs, vec):
    await _seed(docs, "a.pdf", "A")
    await _seed(docs, "b.pdf", "B")
    result = await make_mode2_pipeline(docs, vec).run(["a.pdf"])
    assert result.processed == ["a.pdf"]
    assert {c.source_id for c in vec.stored.values()} == {"a.pdf"}


@pytest.mark.anyio
async def test_orphan_deleted_when_doc_removed_from_doc_store(docs, vec):
    await _seed(docs, "a.pdf", "A")
    await _seed(docs, "b.pdf", "B")
    pipeline = make_mode2_pipeline(docs, vec)
    await pipeline.run()

    await docs.delete_by_source("b.pdf")  # b removed upstream
    result = await pipeline.run(delete_orphans=True)
    assert "b.pdf" in result.deleted
    assert "b.pdf" not in await vec.list_source_ids()


@pytest.mark.anyio
async def test_orphan_kept_by_default(docs, vec):
    await _seed(docs, "a.pdf", "A")
    await _seed(docs, "b.pdf", "B")
    pipeline = make_mode2_pipeline(docs, vec)
    await pipeline.run()

    await docs.delete_by_source("b.pdf")
    result = await pipeline.run()  # delete_orphans=False
    assert result.deleted == []
    assert "b.pdf" in await vec.list_source_ids()
