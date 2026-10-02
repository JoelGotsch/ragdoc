"""Integration tests for VectorStorePipeline: stateless incremental updates via vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline

from .conftest import MemoryVectorStore, make_document


@pytest.mark.anyio
async def test_skip_persists_across_pipeline_instances(tmp_path: Path):
    """A new VectorStorePipeline instance (same vector store) skips unchanged files."""
    doc_path = tmp_path / "doc.html"
    doc_path.write_text("Stable content", encoding="utf-8")

    vstore = MemoryVectorStore()

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    p1 = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    r1 = await p1.run([doc_path])
    assert r1.processed == [doc_path.name]

    # New pipeline instance, same vector store
    p2 = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    r2 = await p2.run([doc_path])
    assert r2.skipped == [doc_path.name]
    assert r2.processed == []


@pytest.mark.anyio
async def test_delete_works_across_pipeline_instances(tmp_path: Path):
    """A new VectorStorePipeline instance can clean up sources added by another."""
    a_path = tmp_path / "a.html"
    b_path = tmp_path / "b.html"
    a_path.write_text("A content", encoding="utf-8")
    b_path.write_text("B content", encoding="utf-8")

    vstore = MemoryVectorStore()

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    p1 = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    await p1.run([a_path, b_path])

    # New pipeline instance removes b by running only a — orphan deletion is opt-in.
    p2 = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    r2 = await p2.run([a_path], delete_orphans=True)
    assert b_path.name in r2.deleted

    # b's chunks should be gone from the store
    source_ids = await vstore.list_source_ids()
    assert b_path.name not in source_ids
