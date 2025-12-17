"""Tests for DocumentStorePipeline (Boundary 1: source files -> DocumentStore).

Covers:
- plan(): new -> to_add; changed file (source_hash) -> to_update; unchanged -> skipped;
  plan() does not touch the store; collision raises.
- apply(): documents stored and retrievable via get_document; list_source_state reflects them.
- run() == apply(plan()); re-run skips unchanged.
- delete_orphans True removes documents absent from the run set.
- a document filtered to None by processing is not stored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import IngestPipeline
from ragdoc.pipeline.document_store_pipeline import DocumentStorePipeline
from ragdoc.processing.base import DocumentProcessor

from .conftest import MemoryDocumentStore, make_document


@pytest.fixture
def doc_store() -> MemoryDocumentStore:
    return MemoryDocumentStore()


@pytest.fixture
def make_files(tmp_path: Path):
    def _make(names_and_contents: dict[str, str]) -> list[Path]:
        paths = []
        for name, content in names_and_contents.items():
            p = tmp_path / name
            p.write_text(content, encoding="utf-8")
            paths.append(p)
        return paths

    return _make


def make_doc_pipeline(doc_store, processors=None, source_id_fn=None) -> DocumentStorePipeline:
    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    dp_kwargs = {}
    if source_id_fn is not None:
        dp_kwargs["source_id_fn"] = source_id_fn
    return DocumentStorePipeline(
        ingest=IngestPipeline(parser=_parse, processors=processors, **dp_kwargs),
        document_store=doc_store,
    )


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_new_sources_to_add(make_files, doc_store):
    paths = make_files({"a.html": "A", "b.html": "B"})
    changeset = await make_doc_pipeline(doc_store).plan(paths)
    assert {sc.source_id for sc in changeset.to_add} == {"a.html", "b.html"}
    # each carries exactly one Document
    assert all(len(sc.items) == 1 and isinstance(sc.items[0], Document) for sc in changeset.to_add)


@pytest.mark.anyio
async def test_plan_does_not_touch_store(make_files, doc_store):
    paths = make_files({"a.html": "A"})
    await make_doc_pipeline(doc_store).plan(paths)
    assert doc_store.stored == {}


@pytest.mark.anyio
async def test_plan_changed_file_to_update(tmp_path, doc_store):
    p = tmp_path / "doc.html"
    p.write_text("v1", encoding="utf-8")
    pipeline = make_doc_pipeline(doc_store)
    await pipeline.run([p])

    p.write_text("v2 different", encoding="utf-8")
    changeset = await pipeline.plan([p])
    assert {sc.source_id for sc in changeset.to_update} == {"doc.html"}


@pytest.mark.anyio
async def test_plan_collision_raises(tmp_path, doc_store):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1, p2 = tmp_path / "a" / "s.pdf", tmp_path / "b" / "s.pdf"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")
    with pytest.raises(ValueError, match=r"s\.pdf"):
        await make_doc_pipeline(doc_store).plan([p1, p2])


# ---------------------------------------------------------------------------
# apply() / run()
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_stores_documents_retrievable(make_files, doc_store):
    paths = make_files({"doc.html": "Hello content"})
    result = await make_doc_pipeline(doc_store).run(paths)

    assert result.processed == ["doc.html"]
    stored = await doc_store.get_document("doc.html")
    assert stored is not None
    assert stored.source_id == "doc.html"
    assert stored.source_hash  # file-byte hash set


@pytest.mark.anyio
async def test_run_list_source_state_reflects_stored(make_files, doc_store):
    paths = make_files({"doc.html": "content"})
    await make_doc_pipeline(doc_store).run(paths)
    state = await doc_store.list_source_state()
    assert "doc.html" in state
    assert state["doc.html"].content_hash is not None


@pytest.mark.anyio
async def test_run_second_run_skips_unchanged(make_files, doc_store):
    paths = make_files({"doc.html": "stable"})
    pipeline = make_doc_pipeline(doc_store)
    await pipeline.run(paths)
    result = await pipeline.run(paths)
    assert result.skipped == ["doc.html"] and result.processed == []


@pytest.mark.anyio
async def test_run_update_replaces_document(tmp_path, doc_store):
    from ragdoc.pipeline.sync import file_hash

    p = tmp_path / "doc.html"
    p.write_text("v1", encoding="utf-8")
    pipeline = make_doc_pipeline(doc_store)
    await pipeline.run([p])

    p.write_text("v2 new content", encoding="utf-8")
    await pipeline.run([p])

    stored = await doc_store.get_document("doc.html")
    assert stored is not None
    assert stored.source_hash == file_hash(p)  # replaced with the v2 file's hash
    assert await doc_store.list_source_ids() == {"doc.html"}  # still one source


# ---------------------------------------------------------------------------
# delete_orphans
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_orphans_default_keeps_absent(make_files, doc_store):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_doc_pipeline(doc_store)
    await pipeline.run(paths)
    result = await pipeline.run([paths[0]])
    assert result.deleted == []
    assert "b.html" in await doc_store.list_source_ids()


@pytest.mark.anyio
async def test_delete_orphans_true_removes_absent(make_files, doc_store):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_doc_pipeline(doc_store)
    await pipeline.run(paths)
    result = await pipeline.run([paths[0]], delete_orphans=True)
    assert result.deleted == ["b.html"]
    assert await doc_store.list_source_ids() == {"a.html"}


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


class DropAll(DocumentProcessor):
    async def process(self, document):
        return None


@pytest.mark.anyio
async def test_filtered_document_not_stored(make_files, doc_store):
    """A filtered source yields an empty change: counted processed, nothing stored."""
    paths = make_files({"doc.html": "content"})
    result = await make_doc_pipeline(doc_store, processors=[DropAll()]).run(paths)
    assert result.processed == ["doc.html"]  # the source was evaluated; it just yields nothing
    assert doc_store.stored == {}


@pytest.mark.anyio
async def test_filtered_document_clears_stale_stored_entry(tmp_path, doc_store):
    """A previously-stored source whose new version is filtered has its stale Document deleted."""

    class DropV2(DocumentProcessor):
        async def process(self, document):
            if any("v2" in getattr(el, "html", "") for el in document.elements):
                return None
            return document

    p = tmp_path / "doc.html"
    p.write_text("v1", encoding="utf-8")
    pipeline = make_doc_pipeline(doc_store, processors=[DropV2()])
    await pipeline.run([p])
    assert "doc.html" in doc_store.stored

    p.write_text("v2 now filtered", encoding="utf-8")
    result = await pipeline.run([p])
    assert result.processed == ["doc.html"]
    assert doc_store.stored == {}  # stale entry deleted, not left behind
