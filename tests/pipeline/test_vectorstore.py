"""Tests for VectorStorePipeline (plan / apply, Mode 1 direct path).

Covers:
- _file_hash(): 64-char hex, content-stable.
- plan(): new -> to_add; changed hash -> to_update; unchanged -> skipped (not in ChangeSet);
  plan() does not touch the store; collision raises before processing.
- delete_orphans: default False (no deletion); True -> to_delete.
- apply(): embeds all chunks BEFORE any store write (failed embed leaves store untouched);
  per-source delete-then-upsert; stale chunks removed on update.
- run() == apply(plan()); reports skipped + per-source errors; never aborts on one bad source.
- custom source_id_fn (on the DocumentPipeline) flows to chunks and to skip detection.
- chunk.content_hash is populated on the direct path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline, EmbedderConfig, VectorStorePipeline
from ragdoc.pipeline.embedders import Embedder
from ragdoc.pipeline.vectorstore import UpdateResult, _file_hash

from .conftest import MemoryVectorStore, make_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture
def vstore() -> MemoryVectorStore:
    return MemoryVectorStore()


def make_vs_pipeline(vstore, source_id_fn=None, embedders=None) -> VectorStorePipeline:
    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    dp_kwargs = {}
    if source_id_fn is not None:
        dp_kwargs["source_id_fn"] = source_id_fn
    return VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse, **dp_kwargs),
        vector_store=vstore,
        embedders=embedders,
    )


# ---------------------------------------------------------------------------
# _file_hash
# ---------------------------------------------------------------------------


def test_file_hash_is_64_char_hex(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    h = _file_hash(p)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_file_hash_content_stable(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    assert _file_hash(a) == _file_hash(b)


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_new_sources_go_to_add(make_files, vstore):
    paths = make_files({"a.html": "Alpha", "b.html": "Beta"})
    changeset = await make_vs_pipeline(vstore).plan(paths)
    assert {sc.source_id for sc in changeset.to_add} == {"a.html", "b.html"}
    assert changeset.to_update == [] and changeset.to_delete == []


@pytest.mark.anyio
async def test_plan_does_not_touch_store(make_files, vstore):
    paths = make_files({"a.html": "Alpha"})
    await make_vs_pipeline(vstore).plan(paths)
    assert vstore.stored == {} and vstore.upsert_calls == [] and vstore.delete_calls == []


@pytest.mark.anyio
async def test_plan_unchanged_source_is_skipped_not_in_changeset(make_files, vstore):
    paths = make_files({"doc.html": "Stable"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)  # populate

    changeset = await pipeline.plan(paths)
    assert changeset.to_add == [] and changeset.to_update == []


@pytest.mark.anyio
async def test_plan_changed_source_goes_to_update(tmp_path, vstore):
    p = tmp_path / "doc.html"
    p.write_text("v1", encoding="utf-8")
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run([p])

    p.write_text("v2 different", encoding="utf-8")
    changeset = await pipeline.plan([p])
    assert {sc.source_id for sc in changeset.to_update} == {"doc.html"}
    assert changeset.to_add == []


@pytest.mark.anyio
async def test_plan_collision_raises_before_processing(tmp_path, vstore):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1, p2 = tmp_path / "a" / "s.pdf", tmp_path / "b" / "s.pdf"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")
    with pytest.raises(ValueError, match=r"s\.pdf"):
        await make_vs_pipeline(vstore).plan([p1, p2])
    assert vstore.upsert_calls == []


# ---------------------------------------------------------------------------
# delete_orphans
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_orphans_default_false_keeps_absent_sources(make_files, vstore):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)

    result = await pipeline.run([paths[0]])  # b.html absent, but default keeps it
    assert result.deleted == []
    assert "b.html" in await vstore.list_source_ids()


@pytest.mark.anyio
async def test_delete_orphans_true_removes_absent_sources(make_files, vstore):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)

    result = await pipeline.run([paths[0]], delete_orphans=True)
    assert result.deleted == ["b.html"]
    assert "b.html" not in await vstore.list_source_ids()


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_apply_embeds_before_writing_and_leaves_store_untouched_on_embed_failure(make_files, vstore):
    paths = make_files({"a.html": "A"})
    failing = MagicMock(spec=Embedder)
    failing.embed = AsyncMock(side_effect=RuntimeError("embed boom"))
    pipeline = make_vs_pipeline(vstore, embedders={"dense": EmbedderConfig(failing)})

    changeset = await pipeline.plan(paths)
    result = await pipeline.apply(changeset)

    assert vstore.stored == {} and vstore.upsert_calls == []  # untouched
    assert [sid for sid, _ in result.errors] == ["a.html"]
    assert result.processed == []


@pytest.mark.anyio
async def test_apply_update_deletes_stale_chunks_before_upsert(tmp_path, vstore):
    p = tmp_path / "doc.html"
    p.write_text("v1", encoding="utf-8")
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run([p])
    old_ids = set(vstore.stored.keys())

    p.write_text("v2 entirely new content", encoding="utf-8")
    await pipeline.run([p])
    assert old_ids.isdisjoint(set(vstore.stored.keys()))
    # source_hash updated to the new file's hash
    assert all(c.source_hash == _file_hash(p) for c in vstore.stored.values())


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_new_files_processed(make_files, vstore):
    paths = make_files({"a.html": "A", "b.html": "B"})
    result = await make_vs_pipeline(vstore).run(paths)
    assert set(result.processed) == {"a.html", "b.html"}
    assert result.skipped == [] and result.errors == []
    assert len(vstore.stored) >= 2


@pytest.mark.anyio
async def test_run_second_run_skips_all(make_files, vstore):
    paths = make_files({"doc.html": "Stable"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)
    result = await pipeline.run(paths)
    assert result.skipped == ["doc.html"] and result.processed == []


@pytest.mark.anyio
async def test_run_empty_sources(vstore):
    result = await make_vs_pipeline(vstore).run([])
    assert result == UpdateResult()


@pytest.mark.anyio
async def test_run_chunks_carry_provenance(make_files, vstore):
    paths = make_files({"doc.html": "Content"})
    await make_vs_pipeline(vstore).run(paths)
    expected_hash = _file_hash(paths[0])
    for chunk in vstore.stored.values():
        assert chunk.source_id == "doc.html"
        assert chunk.source_hash == expected_hash
        assert "source_id" not in chunk.metadata and "source_hash" not in chunk.metadata


@pytest.mark.anyio
async def test_run_chunks_have_content_hash(make_files, vstore):
    paths = make_files({"doc.html": "Some content"})
    await make_vs_pipeline(vstore).run(paths)
    expected = make_document(title="doc", body="Some content").content_hash()
    assert all(c.content_hash == expected for c in vstore.stored.values())


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_parse_error_collected_not_raised(tmp_path, vstore):
    p = tmp_path / "bad.html"
    p.write_text("x", encoding="utf-8")

    async def _parse(_: Path) -> Document:
        raise RuntimeError("parse failed")

    pipeline = VectorStorePipeline(pipeline=DocumentPipeline(parser=_parse), vector_store=vstore)
    result = await pipeline.run([p])
    assert [sid for sid, _ in result.errors] == ["bad.html"]
    assert isinstance(result.errors[0][1], RuntimeError)
    assert result.processed == []


@pytest.mark.anyio
async def test_run_one_bad_source_does_not_block_others(tmp_path, vstore):
    good = tmp_path / "good.html"
    bad = tmp_path / "bad.html"
    good.write_text("Good", encoding="utf-8")
    bad.write_text("x", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        if path.name == "bad.html":
            raise ValueError("bad")
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(pipeline=DocumentPipeline(parser=_parse), vector_store=vstore)
    result = await pipeline.run([good, bad])
    assert result.processed == ["good.html"]
    assert [sid for sid, _ in result.errors] == ["bad.html"]


# ---------------------------------------------------------------------------
# custom source_id_fn (on the DocumentPipeline)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_custom_source_id_fn_flows_to_chunks_and_skip(make_files, vstore):
    paths = make_files({"doc.html": "Content"})
    pipeline = make_vs_pipeline(vstore, source_id_fn=lambda p: str(p))
    await pipeline.run(paths)
    assert all(c.source_id == str(paths[0]) for c in vstore.stored.values())

    result = await pipeline.run(paths)
    assert result.skipped == [str(paths[0])] and result.processed == []


# ---------------------------------------------------------------------------
# run_directory
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_directory_discovers_files(tmp_path, vstore):
    (tmp_path / "a.html").write_text("A", encoding="utf-8")
    (tmp_path / "b.html").write_text("B", encoding="utf-8")
    result = await make_vs_pipeline(vstore).run_directory(tmp_path, glob="*.html")
    assert set(result.processed) == {"a.html", "b.html"}
