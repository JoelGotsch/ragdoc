"""Unit tests for document filter processors."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document, Paragraph
from ragdoc.processing.base import DocumentProcessor, ProcessingPipeline
from ragdoc.processing.filters import EmptyDocumentFilter

# ---------------------------------------------------------------------------
# EmptyDocumentFilter
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_empty_document_filter_returns_none_for_zero_elements():
    doc = Document()
    result = await EmptyDocumentFilter().process(doc)
    assert result is None


@pytest.mark.anyio
async def test_empty_document_filter_passes_non_empty_document():
    doc = Document(elements=[Paragraph(html_content="<p>hello</p>")])
    result = await EmptyDocumentFilter().process(doc)
    assert result is doc


@pytest.mark.anyio
async def test_empty_document_filter_callable():
    """EmptyDocumentFilter works when called directly (via __call__)."""
    doc = Document()
    result = await EmptyDocumentFilter()(doc)
    assert result is None


# ---------------------------------------------------------------------------
# ProcessingPipeline short-circuit on None
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pipeline_short_circuits_on_none():
    """When a processor returns None, subsequent processors must not run."""
    ran = []

    class AlwaysNone(DocumentProcessor):
        async def process(self, document: Document) -> Document | None:
            ran.append("filter")
            return None

    class ShouldNotRun(DocumentProcessor):
        async def process(self, document: Document) -> Document | None:
            ran.append("second")
            return document

    pipeline = ProcessingPipeline([AlwaysNone(), ShouldNotRun()])
    result = await pipeline.process(Document())
    assert result is None
    assert ran == ["filter"], "Second processor must not run after None"


@pytest.mark.anyio
async def test_pipeline_returns_none_when_filter_drops():
    pipeline = ProcessingPipeline([EmptyDocumentFilter()])
    result = await pipeline.process(Document())
    assert result is None


@pytest.mark.anyio
async def test_pipeline_returns_document_when_filter_passes():
    doc = Document(elements=[Paragraph(html_content="<p>text</p>")])
    pipeline = ProcessingPipeline([EmptyDocumentFilter()])
    result = await pipeline.process(doc)
    assert result is doc


@pytest.mark.anyio
async def test_pipeline_callable_returns_none():
    result = await ProcessingPipeline([EmptyDocumentFilter()])(Document())
    assert result is None


# ---------------------------------------------------------------------------
# DocumentPipeline integration: filtered document → empty chunks
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_document_pipeline_returns_empty_chunks_when_document_filtered(
    tmp_path: Path,
):
    """A pipeline that always drops the document must return zero chunks."""
    from ragdoc.pipeline import DocumentPipeline

    fixture = tmp_path / "empty.txt"
    fixture.write_text("")

    async def empty_parser(path: Path) -> Document:
        return Document()  # no elements

    pipeline = DocumentPipeline(
        parser=empty_parser,  # type: ignore[arg-type]
        processors=[EmptyDocumentFilter()],
    )
    chunks = await pipeline.run(fixture)
    assert chunks == []


@pytest.mark.anyio
async def test_document_pipeline_returns_chunks_when_document_passes_filter(
    tmp_path: Path,
):
    """A pipeline whose filter passes must return at least one chunk."""
    from ragdoc.pipeline import DocumentPipeline

    fixture = tmp_path / "doc.txt"
    fixture.write_text("content")

    async def content_parser(path: Path) -> Document:
        return Document(elements=[Paragraph(html_content="<p>Content here.</p>")])

    pipeline = DocumentPipeline(
        parser=content_parser,  # type: ignore[arg-type]
        processors=[EmptyDocumentFilter()],
    )
    chunks = await pipeline.run(fixture)
    assert len(chunks) > 0
