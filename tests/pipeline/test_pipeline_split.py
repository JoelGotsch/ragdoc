"""Phase-9 unconstructibility tests (spec §9.1).

The boundary split makes illegal pipeline configurations *unexpressible*: a stage that does
not belong to a boundary has no constructor parameter to arrive through, so misconfiguration
is a ``TypeError`` at construction — not a prose ``ValueError`` at runtime.
"""

from __future__ import annotations

import pytest

from ragdoc.chunking import SimpleChunker
from ragdoc.pipeline import (
    ChunkPipeline,
    DocumentPipeline,
    DocumentStorePipeline,
    IngestPipeline,
    TokenSplitter,
    VectorStorePipeline,
)

from .conftest import MemoryDocumentStore, MemoryVectorStore

# ---------------------------------------------------------------------------
# IngestPipeline / ChunkPipeline: wrong-boundary stages are unexpected keywords
# ---------------------------------------------------------------------------


def test_chunk_pipeline_cannot_take_a_parser():
    with pytest.raises(TypeError):
        ChunkPipeline(parser=lambda p: p)  # type: ignore[call-arg]


def test_chunk_pipeline_cannot_take_processors():
    with pytest.raises(TypeError):
        ChunkPipeline(processors=[])  # type: ignore[call-arg]


def test_ingest_pipeline_cannot_take_a_chunker():
    with pytest.raises(TypeError):
        IngestPipeline(chunker=SimpleChunker())  # type: ignore[call-arg]


def test_ingest_pipeline_cannot_take_a_splitter():
    with pytest.raises(TypeError):
        IngestPipeline(splitter=TokenSplitter())  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DocumentPipeline: composition of the two sub-pipelines
# ---------------------------------------------------------------------------


def test_document_pipeline_is_composition():
    dp = DocumentPipeline(ingest=IngestPipeline(), chunk=ChunkPipeline(splitter=TokenSplitter()))
    assert isinstance(dp.ingest, IngestPipeline)
    assert isinstance(dp.chunk, ChunkPipeline)


def test_document_pipeline_flat_kwargs_build_sub_pipelines():
    """Direct-path ergonomics: flat stage kwargs delegate into the two sub-pipelines."""
    dp = DocumentPipeline(splitter=TokenSplitter(), source_id_fn=lambda p: str(p))
    assert isinstance(dp.ingest, IngestPipeline)
    assert isinstance(dp.chunk, ChunkPipeline)


def test_document_pipeline_rejects_ingest_alongside_ingest_kwargs():
    with pytest.raises(TypeError, match="IngestPipeline"):
        DocumentPipeline(ingest=IngestPipeline(), processors=[])


def test_document_pipeline_rejects_chunk_alongside_chunk_kwargs():
    with pytest.raises(TypeError, match="ChunkPipeline"):
        DocumentPipeline(chunk=ChunkPipeline(), splitter=TokenSplitter())


def test_has_predicates_deleted():
    for name in ("has_splitter", "has_custom_chunker", "has_processors", "has_custom_parser"):
        assert not hasattr(DocumentPipeline, name)


def test_boundary_methods_moved_to_sub_pipelines():
    """parse_and_process / chunk_document died with the split — the sub-pipelines' run() is the API."""
    assert not hasattr(DocumentPipeline, "parse_and_process")
    assert not hasattr(DocumentPipeline, "chunk_document")


# ---------------------------------------------------------------------------
# Sync pipelines: boundary-appropriate types only
# ---------------------------------------------------------------------------


def test_document_store_pipeline_takes_ingest_not_document_pipeline():
    """Boundary 1 takes an IngestPipeline; a splitter/chunker has no parameter to arrive through."""
    with pytest.raises(TypeError):
        DocumentStorePipeline(pipeline=DocumentPipeline(), document_store=MemoryDocumentStore())  # type: ignore[call-arg]


def test_document_store_pipeline_constructs_with_ingest():
    dsp = DocumentStorePipeline(ingest=IngestPipeline(), document_store=MemoryDocumentStore())
    assert isinstance(dsp, DocumentStorePipeline)


def test_vector_store_pipeline_init_has_no_document_store_param():
    """Mode 2 is a separate constructor — document_store= on __init__ is a TypeError."""
    with pytest.raises(TypeError):
        VectorStorePipeline(
            pipeline=DocumentPipeline(),
            vector_store=MemoryVectorStore(),
            document_store=MemoryDocumentStore(),  # type: ignore[call-arg]
        )


def test_vector_store_from_document_store_takes_chunk_pipeline():
    vsp = VectorStorePipeline.from_document_store(
        chunk=ChunkPipeline(),
        vector_store=MemoryVectorStore(),
        document_store=MemoryDocumentStore(),
    )
    assert isinstance(vsp, VectorStorePipeline)


def test_vector_store_from_document_store_cannot_take_processors_or_parser():
    """A ChunkPipeline cannot carry processors or a parser, and the classmethod accepts neither."""
    with pytest.raises(TypeError):
        VectorStorePipeline.from_document_store(
            chunk=ChunkPipeline(),
            vector_store=MemoryVectorStore(),
            document_store=MemoryDocumentStore(),
            processors=[],  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        VectorStorePipeline.from_document_store(
            chunk=ChunkPipeline(),
            vector_store=MemoryVectorStore(),
            document_store=MemoryDocumentStore(),
            parser=lambda p: p,  # type: ignore[call-arg]
        )
