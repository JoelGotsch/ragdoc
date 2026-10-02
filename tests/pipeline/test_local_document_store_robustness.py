"""LocalDocumentStore robustness: atomic writes and per-file isolation of unreadable documents.

- a failed / interrupted write never leaves a truncated ``*.doc.json`` (temp file + rename);
- one unreadable stored file (bad hand edit, truncated JSON) does not abort
  ``list_source_state``: it is reported as "always changed", stays listed (never an orphan),
  surfaces in ``UpdateResult.errors`` at Boundary 2, and is overwritten by a Boundary-1 re-ingest.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragdoc.document import Document, Paragraph
from ragdoc.pipeline import ChunkPipeline, DocumentStorePipeline, IngestPipeline, VectorStorePipeline
from ragdoc.pipeline.local_document_store import LocalDocumentStore, write_text_atomic
from ragdoc.pipeline.sync import SourceReplaceStore

from .conftest import MemoryVectorStore

TRUNCATED = '{"elements": [ {"oops": '


def make_doc(source_id: str, body: str, source_hash: str = "h") -> Document:
    doc = Document(elements=[Paragraph(html=f"<p>{body}</p>")])
    doc.source_id = source_id
    doc.source_hash = source_hash
    return doc


def body_of(doc: Document | None) -> str:
    assert doc is not None
    return doc.elements[0].text


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upsert_leaves_no_temp_files_and_lists_only_documents(tmp_path: Path):
    store = LocalDocumentStore(tmp_path)
    await store.upsert([make_doc("a", "one"), make_doc("b", "two")])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.doc.json", "b.doc.json"]
    assert await store.list_source_ids() == {"a", "b"}


@pytest.mark.anyio
async def test_write_interrupted_before_the_rename_keeps_the_previous_document(tmp_path: Path, monkeypatch):
    """A crash mid-write (here: the final rename never happens) must not damage the stored file."""
    store = LocalDocumentStore(tmp_path)
    await store.upsert([make_doc("a", "version one")])

    def _crash(src: object, dst: object) -> None:
        raise OSError("disk went away mid-write")

    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError, match="disk went away"):
        await store.upsert([make_doc("a", "version two")])
    monkeypatch.undo()

    assert body_of(await store.get_document("a")) == "version one"
    assert [p.name for p in tmp_path.iterdir()] == ["a.doc.json"], "the temp file must be cleaned up"
    assert set(await store.list_source_state()) == {"a"}


@pytest.mark.anyio
async def test_leftover_temp_file_from_a_killed_process_is_not_a_source(tmp_path: Path):
    """A hard kill can leave the temp file behind; it must never be read as a document."""
    store = LocalDocumentStore(tmp_path)
    await store.upsert([make_doc("a", "one")])
    (tmp_path / ".a.doc.json.deadbeef.tmp").write_text(TRUNCATED, encoding="utf-8")

    assert await store.list_source_ids() == {"a"}
    assert set(await store.list_source_state()) == {"a"}


@pytest.mark.anyio
async def test_write_text_atomic_replaces_existing_content(tmp_path: Path):
    target = tmp_path / "x.json"
    target.write_text("old", encoding="utf-8")

    await write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


@pytest.mark.anyio
async def test_replace_source_swaps_the_document_in_place(tmp_path: Path):
    store = LocalDocumentStore(tmp_path)
    assert isinstance(store, SourceReplaceStore)
    await store.upsert([make_doc("a", "one")])

    assert await store.replace_source("a", [make_doc("a", "two")]) == ["a"]

    assert body_of(await store.get_document("a")) == "two"
    with pytest.raises(ValueError, match="exactly one document"):
        await store.replace_source("a", [make_doc("b", "other")])


# ---------------------------------------------------------------------------
# an unreadable file does not poison the rest
# ---------------------------------------------------------------------------


@pytest.fixture
async def store_with_corrupt_b(tmp_path: Path) -> LocalDocumentStore:
    store = LocalDocumentStore(tmp_path / "docs")
    await store.upsert([make_doc("a", "healthy a"), make_doc("b", "healthy b")])
    (tmp_path / "docs" / "b.doc.json").write_text(TRUNCATED, encoding="utf-8")
    return store


@pytest.mark.anyio
async def test_list_source_state_isolates_the_unreadable_file(store_with_corrupt_b, caplog):
    with caplog.at_level(logging.ERROR, logger="ragdoc.pipeline.local_document_store"):
        state = await store_with_corrupt_b.list_source_state()

    assert set(state) == {"a", "b"}, "the unreadable file stays listed — it is not an orphan"
    assert state["a"].source_hash == "h" and state["a"].content_hash
    assert state["b"].source_hash == "" and state["b"].content_hash is None  # always changed
    assert any("b.doc.json" in r.message and "unreadable" in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_get_document_still_raises_for_the_unreadable_file(store_with_corrupt_b):
    with pytest.raises(ValidationError):
        await store_with_corrupt_b.get_document("b")


@pytest.mark.anyio
async def test_non_utf8_file_is_isolated_too(tmp_path: Path):
    store = LocalDocumentStore(tmp_path)
    await store.upsert([make_doc("a", "ok")])
    (tmp_path / "bin.doc.json").write_bytes(b"\xff\xfe\x00garbage")

    state = await store.list_source_state()

    assert state["bin"].content_hash is None and state["a"].content_hash


@pytest.mark.anyio
async def test_boundary2_run_syncs_healthy_docs_and_reports_the_unreadable_one(store_with_corrupt_b):
    """The reviewer's repro: previously the whole run raised ValidationError and synced nothing."""
    vectors = MemoryVectorStore()
    vsp = VectorStorePipeline.from_document_store(ChunkPipeline(), vectors, store_with_corrupt_b)

    result = await vsp.run()

    assert result.processed == ["a"]
    assert [sid for sid, _ in result.errors] == ["b"]
    assert isinstance(result.errors[0][1], ValidationError)
    assert await vectors.list_source_ids() == {"a"}

    again = await vsp.run()  # the error keeps surfacing until the file is fixed
    assert again.skipped == ["a"] and [sid for sid, _ in again.errors] == ["b"]


@pytest.mark.anyio
async def test_boundary2_never_orphan_deletes_the_chunks_of_an_unreadable_document(tmp_path: Path):
    store = LocalDocumentStore(tmp_path / "docs")
    await store.upsert([make_doc("a", "healthy a"), make_doc("b", "healthy b")])
    vectors = MemoryVectorStore()
    vsp = VectorStorePipeline.from_document_store(ChunkPipeline(), vectors, store)
    await vsp.run()
    b_chunks = {cid for cid, c in vectors.stored.items() if c.source_id == "b"}
    (tmp_path / "docs" / "b.doc.json").write_text(TRUNCATED, encoding="utf-8")

    result = await vsp.run(delete_orphans=True)

    assert result.deleted == [] and [sid for sid, _ in result.errors] == ["b"]
    assert {cid for cid, c in vectors.stored.items() if c.source_id == "b"} == b_chunks

    plan = await vsp.plan(delete_orphans=True)  # plan() must not raise either
    assert plan.to_delete == [] and plan.to_update == []


@pytest.mark.anyio
async def test_boundary1_reingest_overwrites_the_unreadable_file(tmp_path: Path):
    src = tmp_path / "b.txt"
    src.write_text("fresh from source", encoding="utf-8")

    async def parse(path: Path) -> Document:
        return Document(elements=[Paragraph(html=f"<p>{path.read_text(encoding='utf-8')}</p>")])

    store = LocalDocumentStore(tmp_path / "docs")
    dsp = DocumentStorePipeline(IngestPipeline(parser=parse), store)
    await dsp.run([src])
    assert (await dsp.run([src])).skipped == ["b.txt"]
    (tmp_path / "docs" / "b.txt.doc.json").write_text(TRUNCATED, encoding="utf-8")

    result = await dsp.run([src])  # previously: ValidationError out of list_source_state

    assert result.processed == ["b.txt"] and result.errors == []
    assert body_of(await store.get_document("b.txt")) == "fresh from source"
    assert (await dsp.run([src])).skipped == ["b.txt"]
