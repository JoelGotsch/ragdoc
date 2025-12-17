"""Tests for the document processing architecture."""

import pytest

from ragdoc.document import Document
from ragdoc.processing.base import (
    DocumentProcessor,
    ProcessingPipeline,
)

# --- TestDocumentProcessor ---


@pytest.mark.anyio
async def test_document_processor_can_be_called_directly():
    """DocumentProcessor can be awaited like a function."""

    class UppercaseTitles(DocumentProcessor):
        async def process(self, document: Document) -> Document:
            if document.title:
                document.title = document.title.upper()
            return document

    doc = Document(title="hello world")
    processor = UppercaseTitles()
    result = await processor(doc)
    assert result.title == "HELLO WORLD"


# --- TestProcessingPipeline ---


@pytest.mark.anyio
async def test_pipeline_single_processor():
    """Pipeline with a single processor works."""

    class AddSuffix(DocumentProcessor):
        async def process(self, document: Document) -> Document:
            document.title = (document.title or "") + " - processed"
            return document

    pipeline = ProcessingPipeline([AddSuffix()])
    doc = Document(title="Original")
    result = await pipeline.process(doc)
    assert result.title == "Original - processed"
