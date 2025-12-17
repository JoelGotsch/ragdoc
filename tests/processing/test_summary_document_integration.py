"""Integration tests for DocumentSummarizerProcessor against a real parsed HTML document.

Uses the real ``split_document`` splitter, real ``Renderer`` and ``GPTTokenizer``, and a small
``max_input_tokens`` so the document genuinely fans out — only the LLM call is mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pydantic_settings", reason="pydantic_settings not installed")

from ragdoc.document import Document
from ragdoc.parsing.html.load import HTML, generate_document
from ragdoc.processing.base import ProcessingPipeline
from ragdoc.processing.summary_base import DocumentSummary
from ragdoc.processing.summary_document import (
    DocumentSummarizerProcessor,
    DocumentSummarizerSettings,
)

_FIXTURE = Path(__file__).parent.parent / "parsing" / "data" / "html" / "tesla-q4-2024-update.html"


@pytest.fixture
def real_document() -> Document:
    doc = generate_document(HTML.from_file(_FIXTURE))
    doc.source_path = str(_FIXTURE)
    doc.metadata["filename"] = _FIXTURE.name
    return doc


def _counting_client() -> MagicMock:
    """Mock client returning a deterministic per-call summary, recording calls."""
    state = {"n": 0}

    async def parse(*args, **kwargs):
        state["n"] += 1
        return MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=DocumentSummary(summary=f"summary-{state['n']}")))]
        )

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(side_effect=parse)
    return client


@pytest.mark.anyio
async def test_summarizes_real_document_with_real_splitter(real_document: Document):
    client = _counting_client()
    # Small ceiling so the real splitter actually fans the ~25k-token document into pieces.
    # Kept comfortably above the splitter's default overlap_tokens (200) so the element-level
    # text fallback has room for content.
    settings = DocumentSummarizerSettings(min_tokens=100, max_input_tokens=1000)
    processor = DocumentSummarizerProcessor(client=client, settings=settings, overwrite=False)

    original_keys = set(real_document.metadata)
    result = await processor.process(real_document)

    # Summary written, non-empty.
    assert result.metadata["summary"]
    assert isinstance(result.metadata["summary"], str)

    # The real splitter fanned out → more than a single LLM call (map + reduce).
    assert client.beta.chat.completions.parse.call_count >= 2

    # Original metadata preserved; no split-provenance leaked onto the document.
    assert original_keys <= set(result.metadata)
    assert result.metadata["filename"] == _FIXTURE.name
    assert "split_sequence" not in result.metadata
    assert "split_total" not in result.metadata


@pytest.mark.anyio
async def test_runs_through_processing_pipeline(real_document: Document):
    client = _counting_client()
    settings = DocumentSummarizerSettings(min_tokens=100, max_input_tokens=1000)
    pipeline = ProcessingPipeline([DocumentSummarizerProcessor(client=client, settings=settings)])

    result = await pipeline.process(real_document)

    assert result is not None
    assert result.metadata["summary"]
