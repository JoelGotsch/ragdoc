"""Unit tests for pure-logic functions: SimpleChunker ID generation and
TokenSplitter splitting behavior.

These tests exercise deterministic behavior of core primitives without running
the full DocumentPipeline. Document.content_hash() itself is covered in
tests/test_content_hash.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.chunking import SimpleChunker
from ragdoc.document import Document, Heading, Paragraph
from ragdoc.pipeline import TokenSplitter

from .conftest import make_document

# --- TestChunkerProvenance ---
# Chunkers resolve provenance via resolve_chunk_provenance; ids are minted pipeline-side.


@pytest.mark.anyio
async def test_simple_chunker_forwards_document_provenance():
    doc = make_document(title="A", body="B")
    doc.source_id = "real-id"
    doc.source_hash = "real-hash"
    chunk = (await SimpleChunker().chunk(doc))[0]
    assert chunk.source_id == "real-id"
    assert chunk.source_hash == "real-hash"
    assert chunk.content_hash == doc.content_hash()


@pytest.mark.anyio
async def test_simple_chunker_source_hash_none_when_document_has_none():
    """No file hash means source_hash is None — never the content hash (no masquerade)."""
    doc = make_document(title="A", body="B")
    doc.source_path = "folder/report.pdf"
    chunk = (await SimpleChunker().chunk(doc))[0]
    assert chunk.source_id == "folder/report.pdf"  # source_path fallback
    assert chunk.source_hash is None
    assert chunk.content_hash == doc.content_hash()


@pytest.mark.anyio
async def test_simple_chunker_falls_back_to_doc_id_when_no_source_path():
    doc = make_document(title="A", body="B")  # source_path defaults to ""
    chunk = (await SimpleChunker().chunk(doc))[0]
    assert chunk.source_id == doc.id  # last-resort fallback, never None


@pytest.mark.anyio
async def test_simple_chunker_metadata_is_copied_per_chunk():
    doc = make_document(title="A", body="B")
    chunk = (await SimpleChunker().chunk(doc))[0]
    chunk.metadata["injected"] = True
    assert "injected" not in doc.metadata


# --- TestChunkPipeline (ChunkPipeline.run: provenance + id minting) ---


@pytest.mark.anyio
async def test_chunk_document_stamps_uniform_provenance():
    from ragdoc.pipeline import ChunkPipeline

    doc = make_document(title="A", body="B")
    doc.source_id = "sync-id"
    doc.source_hash = "file-hash"
    chunks = await ChunkPipeline().run(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.source_id == "sync-id"
        assert chunk.source_hash == "file-hash"
        assert chunk.content_hash == doc.content_hash()


@pytest.mark.anyio
async def test_chunk_document_source_hash_none_passthrough():
    """ChunkPipeline.run never fabricates a source_hash from the content hash."""
    from ragdoc.pipeline import ChunkPipeline

    doc = make_document(title="A", body="B")
    chunks = await ChunkPipeline().run(doc)
    assert chunks
    assert all(c.source_hash is None for c in chunks)


@pytest.mark.anyio
async def test_chunk_document_identical_content_splits_get_distinct_ids():
    """The collision regression: two content-identical splits must yield two distinct ids."""
    from ragdoc.pipeline import ChunkPipeline

    doc = make_document(title="A", body="Same body.")

    def duplicate_splitter(document: Document) -> list[Document]:
        return [document.model_copy(), document.model_copy()]

    pipeline = ChunkPipeline(splitter=duplicate_splitter)  # type: ignore[arg-type]  # Splitter protocol
    chunks = await pipeline.run(doc)
    assert len(chunks) == 2
    assert chunks[0].id != chunks[1].id
    assert chunks[0].content_hash == chunks[1].content_hash


@pytest.mark.anyio
async def test_chunk_document_ids_stable_across_runs():
    from ragdoc.pipeline import ChunkPipeline

    doc1 = make_document(title="A", body="B")
    doc1.source_id = "stable-src"
    doc2 = make_document(title="A", body="B")
    doc2.source_id = "stable-src"
    ids1 = [c.id for c in await ChunkPipeline().run(doc1)]
    ids2 = [c.id for c in await ChunkPipeline().run(doc2)]
    assert ids1 == ids2


@pytest.mark.anyio
async def test_chunk_id_fn_override():
    from ragdoc.pipeline import ChunkPipeline

    doc = make_document(title="A", body="B")
    doc.source_id = "src"
    pipeline = ChunkPipeline(chunk_id_fn=lambda sid, seq, ordinal, ch: f"{sid}:{seq}:{ordinal}")
    chunks = await pipeline.run(doc)
    assert [c.id for c in chunks] == ["src:1:0"]


@pytest.mark.anyio
async def test_process_one_sets_source_id_from_source_id_fn(simple_parser):
    from ragdoc.pipeline import DocumentPipeline

    pipeline = DocumentPipeline(parser=simple_parser, source_id_fn=lambda p: f"custom::{p.name}")
    chunks = await pipeline.run(Path("report.html"))
    assert chunks
    assert all(c.source_id == "custom::report.html" for c in chunks)


# --- TestTokenSplitter ---


def test_token_splitter_small_document_is_not_split():
    doc = make_document(title="Short", body="A brief paragraph.")
    splitter = TokenSplitter(max_tokens=7000)
    splits = splitter(doc)
    assert len(splits) == 1
    # no-split path returns a shallow copy (fresh metadata dict); elements are shared
    assert splits[0].elements == doc.elements
    assert "split_sequence" not in doc.metadata


def test_token_splitter_large_document_is_split():
    body = "Word " * 200  # well over 200 tokens
    doc = Document(
        title="Long doc",
        elements=[
            Heading(html="<h1>Long doc</h1>"),
            Paragraph(html=f"<p>{body}</p>"),
        ],
    )
    splitter = TokenSplitter(max_tokens=200, overlap_tokens=20)
    splits = splitter(doc)
    assert len(splits) >= 2


def test_token_splitter_respects_max_tokens_param():
    body = "Word " * 300
    doc = Document(
        title="T",
        elements=[
            Heading(html="<h1>T</h1>"),
            Paragraph(html=f"<p>{body}</p>"),
        ],
    )
    splitter_tight = TokenSplitter(max_tokens=150, overlap_tokens=20)
    splitter_loose = TokenSplitter(max_tokens=1000, overlap_tokens=20)
    assert len(splitter_tight(doc)) >= len(splitter_loose(doc))


# --- stream() task hygiene (Phase 8, D9) ---


@pytest.mark.anyio
async def test_stream_cancels_pending_tasks_when_consumer_stops_early():
    """Breaking out of stream() must cancel (not abandon) the still-pending parse tasks."""
    import asyncio

    from ragdoc.pipeline import DocumentPipeline

    release = asyncio.Event()
    cancelled: list[str] = []

    async def parser(path: Path) -> Document:
        if path.name == "fast.html":
            doc = make_document(title="fast", body="done")
            doc.source_path = str(path)
            return doc
        try:
            await release.wait()  # blocks forever unless cancelled
        except asyncio.CancelledError:
            cancelled.append(path.name)
            raise
        doc = make_document(title="slow", body="done")
        doc.source_path = str(path)
        return doc

    pipeline = DocumentPipeline(parser=parser, concurrency=3)
    paths = [Path("fast.html"), Path("slow-1.html"), Path("slow-2.html")]

    stream = pipeline.stream(paths)
    async for batch in stream:
        assert batch
        break  # stop consuming after the first yielded batch
    await stream.aclose()  # closing the generator runs its finally: cancel + gather

    assert sorted(cancelled) == ["slow-1.html", "slow-2.html"]
