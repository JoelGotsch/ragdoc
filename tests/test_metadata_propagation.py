"""Tests that document fields and metadata propagate through the entire pipeline.

``source_path`` is an explicit field on ``Document`` and ``Chunk``.
``metadata`` carries user-defined custom data (including the optional
``metadata["filename"]`` written by parsers).  Both must survive processing,
splitting, and chunking stages.
"""

import pytest

from ragdoc.chunking import SimpleChunker
from ragdoc.chunking.chunk import Chunk
from ragdoc.document import Document, Heading, Paragraph
from ragdoc.processing import ProcessingPipeline
from ragdoc.processing.heading import HeadingLevelProcessor
from ragdoc.splitting.base import split_by_headings
from ragdoc.splitting.hierarchical import split_hierarchical

SOURCE_PATH = "/data/report.pdf"
CUSTOM_METADATA = {"custom_key": "custom_value", "filename": "report.pdf"}


def _make_doc() -> Document:
    return Document(
        title="Report",
        source_path=SOURCE_PATH,
        metadata=CUSTOM_METADATA,
        elements=[
            Heading(innerhtml="Introduction", level=1),
            Paragraph(html_content="<p>Intro text.</p>"),
            Heading(innerhtml="Methods", level=1),
            Paragraph(html_content="<p>Methods text.</p>"),
            Heading(innerhtml="Results", level=1),
            Paragraph(html_content="<p>Results text.</p>"),
        ],
    )


def _assert_doc_provenance(doc: Document) -> None:
    assert doc.source_path == SOURCE_PATH
    assert doc.metadata == CUSTOM_METADATA


def _assert_chunk_provenance(chunk: Chunk) -> None:
    assert chunk.source_path == SOURCE_PATH
    assert chunk.metadata["custom_key"] == "custom_value"
    assert chunk.metadata["filename"] == "report.pdf"


class TestFieldsSurviveProcessing:
    @pytest.mark.anyio
    async def test_processing_preserves_fields(self):
        doc = _make_doc()
        pipeline = ProcessingPipeline(processors=[HeadingLevelProcessor()])
        result = await pipeline.process(doc)
        _assert_doc_provenance(result)

    @pytest.mark.anyio
    async def test_multiple_processors_preserve_fields(self):
        doc = _make_doc()
        pipeline = ProcessingPipeline(processors=[HeadingLevelProcessor(), HeadingLevelProcessor()])
        result = await pipeline.process(doc)
        _assert_doc_provenance(result)


class TestFieldsSurviveSplitting:
    def test_split_by_headings_preserves_fields(self):
        doc = _make_doc()
        splits = split_by_headings(doc)
        assert len(splits) > 1
        for split in splits:
            _assert_doc_provenance(split)

    def test_split_hierarchical_preserves_fields(self):
        doc = _make_doc()
        splits = split_hierarchical(doc)
        assert len(splits) > 1
        for split in splits:
            _assert_doc_provenance(split)


class TestFieldsSurviveChunking:
    @pytest.mark.anyio
    async def test_simple_chunker_preserves_fields(self):
        doc = _make_doc()
        chunker = SimpleChunker()
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        _assert_chunk_provenance(chunks[0])


class TestFieldsEndToEnd:
    @pytest.mark.anyio
    async def test_process_split_chunk_preserves_fields(self):
        """Full pipeline: process -> split -> chunk preserves fields on every chunk."""
        doc = _make_doc()

        # Process
        pipeline = ProcessingPipeline(processors=[HeadingLevelProcessor()])
        processed = await pipeline.process(doc)

        # Split
        splits = split_hierarchical(processed)
        assert len(splits) > 1

        # Chunk each split
        chunker = SimpleChunker()
        all_chunks: list[Chunk] = []
        for split in splits:
            all_chunks.extend(await chunker.chunk(split))

        assert len(all_chunks) > 1
        for chunk in all_chunks:
            _assert_chunk_provenance(chunk)
