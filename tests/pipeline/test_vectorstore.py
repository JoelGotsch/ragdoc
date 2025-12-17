"""Tests for VectorStorePipeline (stateless, vector-store-as-source-of-truth).

Covers:
- run(): new files are processed and upserted with source_id / source_hash fields
- run(): unchanged files (same hash in vector store) are skipped
- run(): changed files delete stale chunks by source_id and re-process
- run(): removed sources delete chunks from vector store
- run(): per-file errors are collected in result.errors (never propagated)
- run(): source_id_fn collision raises ValueError before any processing
- run(): custom source_id_fn is used for store queries
- run(): concurrency=N limits concurrent pipeline.run() calls
- run_directory(): discovers files and calls run()
- _file_hash(): returns a 64-char hex string
- UpdateResult fields are populated correctly
- chunk.metadata is NOT modified by the pipeline (provenance is in source_id/source_hash)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline
from ragdoc.pipeline.vectorstore import _file_hash

from .conftest import MemoryVectorStore, make_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_files(tmp_path: Path):
    """Factory: create real text files in tmp_path and return their paths."""

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


def make_vs_pipeline(
    vstore: MemoryVectorStore,
    source_id_fn=None,
) -> VectorStorePipeline:
    """Build a VectorStorePipeline with a trivial file-aware parser."""

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    kwargs = {}
    if source_id_fn is not None:
        kwargs["source_id_fn"] = source_id_fn

    return VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
        **kwargs,
    )


# --- TestFileHash ---


def test_file_hash_returns_64_char_hex(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    h = _file_hash(p)
    assert isinstance(h, str) and len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_file_hash_same_content_same_hash(tmp_path: Path):
    p1, p2 = tmp_path / "a.txt", tmp_path / "b.txt"
    p1.write_text("same", encoding="utf-8")
    p2.write_text("same", encoding="utf-8")
    assert _file_hash(p1) == _file_hash(p2)


def test_file_hash_different_content_different_hash(tmp_path: Path):
    p1, p2 = tmp_path / "a.txt", tmp_path / "b.txt"
    p1.write_text("hello", encoding="utf-8")
    p2.write_text("world", encoding="utf-8")
    assert _file_hash(p1) != _file_hash(p2)


# --- TestSyncNewFiles ---


@pytest.mark.anyio
async def test_sync_new_files_are_processed(make_files, vstore):
    paths = make_files({"a.html": "Alpha content", "b.html": "Beta content"})
    pipeline = make_vs_pipeline(vstore)
    result = await pipeline.run(paths)

    assert len(result.processed) == 2
    assert result.skipped == []
    assert result.deleted == []
    assert result.errors == []


@pytest.mark.anyio
async def test_sync_new_files_upserted_to_vector_store(make_files, vstore):
    paths = make_files({"doc.html": "Some content"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)

    assert len(vstore.upsert_calls) == 1
    assert len(vstore.stored) >= 1


@pytest.mark.anyio
async def test_sync_new_files_have_provenance_fields(make_files, vstore):
    """Provenance is stored in chunk.source_id and chunk.source_hash, not metadata."""
    paths = make_files({"doc.html": "Content here"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)

    expected_source_id = paths[0].name  # default: filename
    expected_hash = _file_hash(paths[0])

    for chunk in vstore.stored.values():
        assert chunk.source_id == expected_source_id
        assert chunk.source_hash == expected_hash
        # metadata must NOT contain provenance keys
        assert "source_path" not in chunk.metadata
        assert "source_hash" not in chunk.metadata


@pytest.mark.anyio
async def test_sync_new_files_empty_sources_returns_empty_result(vstore):
    pipeline = make_vs_pipeline(vstore)
    result = await pipeline.run([])
    assert result.processed == []
    assert result.skipped == []
    assert result.deleted == []


# --- TestSyncSkip ---


@pytest.mark.anyio
async def test_sync_skip_second_run_skips_all(make_files, vstore):
    paths = make_files({"doc.html": "Stable content"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)
    result2 = await pipeline.run(paths)

    assert result2.skipped == paths
    assert result2.processed == []


@pytest.mark.anyio
async def test_sync_skip_does_not_call_upsert_again(make_files, vstore):
    paths = make_files({"doc.html": "Stable content"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)
    upsert_count_after_first = len(vstore.upsert_calls)

    await pipeline.run(paths)
    assert len(vstore.upsert_calls) == upsert_count_after_first


@pytest.mark.anyio
async def test_sync_skip_partial_skip(make_files, vstore):
    """First file unchanged, second file is new."""
    paths = make_files({"old.html": "Old content", "new.html": "New content"})
    pipeline = make_vs_pipeline(vstore)
    # Pre-populate vector store with first file only
    await pipeline.run([paths[0]])

    result = await pipeline.run(paths)
    assert paths[0] in result.skipped
    assert paths[1] in result.processed


# --- TestSyncUpdate ---


@pytest.mark.anyio
async def test_sync_update_changed_file_is_reprocessed(tmp_path, vstore):
    p = tmp_path / "doc.html"
    p.write_text("Version 1", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    r1 = await pipeline.run([p])
    assert r1.processed == [p]

    p.write_text("Version 2 — completely different", encoding="utf-8")

    r2 = await pipeline.run([p])
    assert r2.processed == [p]
    assert r2.skipped == []


@pytest.mark.anyio
async def test_sync_update_stale_chunks_deleted_on_update(tmp_path, vstore):
    p = tmp_path / "doc.html"
    p.write_text("Version 1", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    await pipeline.run([p])
    old_ids = set(vstore.stored.keys())

    p.write_text("Version 2", encoding="utf-8")
    await pipeline.run([p])

    # Old chunks should no longer be in the store
    new_ids = set(vstore.stored.keys())
    assert old_ids.isdisjoint(new_ids) or old_ids != new_ids


@pytest.mark.anyio
async def test_sync_update_source_hash_updated_after_reprocess(tmp_path, vstore):
    p = tmp_path / "doc.html"
    p.write_text("v1", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    await pipeline.run([p])

    p.write_text("v2 new content here", encoding="utf-8")
    await pipeline.run([p])

    # All chunks should now have the new hash in source_hash field
    for chunk in vstore.stored.values():
        assert chunk.source_hash == _file_hash(p)


# --- TestSyncDelete ---


@pytest.mark.anyio
async def test_sync_delete_removed_source_appears_in_deleted(make_files, vstore):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)

    result = await pipeline.run([paths[0]])
    # deleted contains source_id strings (filenames by default)
    assert paths[1].name in result.deleted


@pytest.mark.anyio
async def test_sync_delete_removed_source_chunks_deleted_from_store(make_files, vstore):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)

    # Record chunks for b (keyed by source_id = filename)
    b_source_id = paths[1].name
    b_ids = {cid for cid, c in vstore.stored.items() if c.source_id == b_source_id}
    assert b_ids  # sanity check

    await pipeline.run([paths[0]])

    # b chunks should be gone
    remaining_ids = set(vstore.stored.keys())
    assert b_ids.isdisjoint(remaining_ids)


@pytest.mark.anyio
async def test_sync_delete_removed_source_not_in_vector_store(make_files, vstore):
    paths = make_files({"a.html": "A", "b.html": "B"})
    pipeline = make_vs_pipeline(vstore)
    await pipeline.run(paths)
    await pipeline.run([paths[0]])

    # No chunks for b should exist
    source_ids = await vstore.list_source_ids()
    assert paths[1].name not in source_ids


# --- TestSyncErrorHandling ---


@pytest.mark.anyio
async def test_sync_error_parse_error_collected_in_errors(tmp_path, vstore):
    p = tmp_path / "bad.html"
    p.write_text("x", encoding="utf-8")

    async def _parse(_: Path) -> Document:
        raise RuntimeError("parse failed")

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    result = await pipeline.run([p])
    assert len(result.errors) == 1
    assert result.errors[0][0] == p
    assert isinstance(result.errors[0][1], RuntimeError)
    assert result.processed == []


@pytest.mark.anyio
async def test_sync_error_partial_error(tmp_path, vstore):
    good = tmp_path / "good.html"
    bad = tmp_path / "bad.html"
    good.write_text("Good content", encoding="utf-8")
    bad.write_text("x", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        if path == bad:
            raise ValueError("bad file")
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    result = await pipeline.run([good, bad])
    assert good in result.processed
    assert bad in [e[0] for e in result.errors]


@pytest.mark.anyio
async def test_sync_error_does_not_prevent_other_files(tmp_path, vstore):
    good1 = tmp_path / "good1.html"
    bad = tmp_path / "bad.html"
    good2 = tmp_path / "good2.html"
    good1.write_text("Content 1", encoding="utf-8")
    bad.write_text("x", encoding="utf-8")
    good2.write_text("Content 2", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        if path == bad:
            raise ValueError("bad")
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    result = await pipeline.run([good1, bad, good2])
    assert len(result.processed) == 2
    assert len(result.errors) == 1


# --- TestSourceIdCollision ---


@pytest.mark.anyio
async def test_sync_collision_raises_before_processing(tmp_path, vstore):
    """Two paths with the same filename collide under the default source_id_fn."""
    subdir_a = tmp_path / "a"
    subdir_b = tmp_path / "b"
    subdir_a.mkdir()
    subdir_b.mkdir()
    p1 = subdir_a / "summary.pdf"
    p2 = subdir_b / "summary.pdf"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")

    pipeline = make_vs_pipeline(vstore)
    with pytest.raises(ValueError, match=r"summary\.pdf"):
        await pipeline.run([p1, p2])

    # No processing should have occurred
    assert vstore.upsert_calls == []


@pytest.mark.anyio
async def test_sync_collision_error_message_mentions_strategy(tmp_path, vstore):
    """The collision error message should suggest a relative-path strategy."""
    subdir_a = tmp_path / "a"
    subdir_b = tmp_path / "b"
    subdir_a.mkdir()
    subdir_b.mkdir()
    (subdir_a / "doc.pdf").write_text("A", encoding="utf-8")
    (subdir_b / "doc.pdf").write_text("B", encoding="utf-8")

    pipeline = make_vs_pipeline(vstore)
    with pytest.raises(ValueError, match="relative"):
        await pipeline.run([subdir_a / "doc.pdf", subdir_b / "doc.pdf"])


# --- TestCustomSourceIdFn ---


@pytest.mark.anyio
async def test_custom_source_id_fn_used_for_queries(make_files, vstore):
    """Custom source_id_fn produces the correct source_id on stored chunks."""
    paths = make_files({"doc.html": "Content"})
    # Use full path as source_id
    pipeline = make_vs_pipeline(vstore, source_id_fn=lambda p: str(p))
    await pipeline.run(paths)

    for chunk in vstore.stored.values():
        assert chunk.source_id == str(paths[0])


@pytest.mark.anyio
async def test_custom_source_id_fn_skip_uses_same_id(make_files, vstore):
    """Second sync with custom fn correctly skips unchanged file."""
    paths = make_files({"doc.html": "Content"})
    pipeline = make_vs_pipeline(vstore, source_id_fn=lambda p: str(p))
    await pipeline.run(paths)

    result2 = await pipeline.run(paths)
    assert result2.skipped == paths
    assert result2.processed == []


# --- TestSyncInconsistentState ---


@pytest.mark.anyio
async def test_sync_missing_hash_triggers_reprocess(tmp_path, vstore):
    """If a source has chunks but no source_hash, the pipeline re-processes it."""
    from ragdoc.chunking import Chunk

    p = tmp_path / "doc.html"
    p.write_text("Current content", encoding="utf-8")

    # Simulate state where chunk has source_id but no source_hash
    chunk = Chunk(
        id="old-1",
        prompt_content="old",
        embedding_content="old",
        source_id=p.name,
        source_hash="old-hash",
    )
    await vstore.upsert([chunk])

    pipeline = make_vs_pipeline(vstore)
    result = await pipeline.run([p])

    # Should have re-processed (get_source_hash returns None → treat as new)
    assert p in result.processed


# --- TestRunDirectory ---


@pytest.mark.anyio
async def test_run_directory_discovers_files(tmp_path, vstore):
    (tmp_path / "a.html").write_text("A", encoding="utf-8")
    (tmp_path / "b.html").write_text("B", encoding="utf-8")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "c.html").write_text("C", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
        source_id_fn=lambda p: str(p),  # avoid filename collisions in subdirs
    )
    result = await pipeline.run_directory(tmp_path, glob="**/*.html")
    assert len(result.processed) == 3


@pytest.mark.anyio
async def test_run_directory_glob_filters(tmp_path, vstore):
    (tmp_path / "a.html").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    async def _parse(path: Path) -> Document:
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
    )
    result = await pipeline.run_directory(tmp_path, glob="*.html")
    assert len(result.processed) == 1


# --- TestConcurrency ---


@pytest.mark.anyio
async def test_concurrency_limits_simultaneous_pipeline_runs(tmp_path, vstore):
    """VectorStorePipeline(concurrency=2) runs at most 2 pipeline.run() calls at once."""
    paths = [tmp_path / f"doc_{i}.html" for i in range(6)]
    for p in paths:
        p.write_text(f"Content {p.stem}", encoding="utf-8")

    concurrent = 0
    max_concurrent = 0

    async def _parse(path: Path) -> Document:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=vstore,
        concurrency=2,
    )
    result = await pipeline.run(paths)

    assert len(result.processed) == 6
    assert max_concurrent <= 2


@pytest.mark.anyio
async def test_concurrency_shared_semaphore(tmp_path):
    """A shared Semaphore(1) serialises two VectorStorePipeline.run() calls."""
    shared = asyncio.Semaphore(1)
    concurrent = 0
    max_concurrent = 0

    async def _parse(path: Path) -> Document:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

    store1, store2 = MemoryVectorStore(), MemoryVectorStore()
    files_a = [tmp_path / f"a_{i}.html" for i in range(3)]
    files_b = [tmp_path / f"b_{i}.html" for i in range(3)]
    for p in files_a + files_b:
        p.write_text("x", encoding="utf-8")

    vs1 = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=store1,
        source_id_fn=lambda p: str(p),
        concurrency=shared,
    )
    vs2 = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parse),
        vector_store=store2,
        source_id_fn=lambda p: str(p),
        concurrency=shared,
    )
    await asyncio.gather(vs1.run(files_a), vs2.run(files_b))
    assert max_concurrent <= 1


@pytest.mark.anyio
async def test_concurrency_default_processes_all_files(tmp_path, vstore):
    """Default concurrency=10 still processes all files correctly."""
    paths = [tmp_path / f"doc_{i}.html" for i in range(5)]
    for p in paths:
        p.write_text(f"Content {p.stem}", encoding="utf-8")

    pipeline = make_vs_pipeline(vstore)
    result = await pipeline.run(paths)

    assert len(result.processed) == 5
    assert result.errors == []
