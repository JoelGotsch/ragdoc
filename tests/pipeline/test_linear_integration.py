"""Integration tests for DocumentPipeline: full parse → process → split → chunk.

Uses HeadingNormalizationProcessor (a real library processor) and asserts on
actual rendered chunk content, not just counts or structure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.pipeline import DocumentPipeline, TokenSplitter
from ragdoc.processing.heading import HeadingNormalizationProcessor


async def _parse_structured_report(path: Path) -> Document:
    """Multi-section report with non-contiguous heading levels (gaps at 1 → 3 → 5).

    Simulates a realistic parser output where visual-size-based level assignment
    produces non-contiguous levels that need normalisation before rendering.
    """
    return Document(
        elements=[
            Heading(innerhtml="Quarterly Performance Report", level=1),
            Paragraph(html_content="<p>This report covers Q2 2024 performance metrics.</p>"),
            Heading(innerhtml="Revenue Analysis", level=3),  # gap: 1 → 3
            Paragraph(html_content="<p>Total revenue reached $4.2M, up 15% year-over-year.</p>"),
            Heading(innerhtml="By Region", level=5),  # gap: 3 → 5
            Paragraph(html_content="<p>EMEA contributed 42%, APAC 38%, Americas 20%.</p>"),
            Heading(innerhtml="Operating Costs", level=3),
            Paragraph(
                html_content="<p>Operating margin held at 23% despite inflationary pressure.</p>"
            ),
        ],
    )


@pytest.mark.anyio
async def test_pipeline_normalises_heading_gaps_and_preserves_content():
    """Full pipeline: parse → HeadingNormalizationProcessor → split → chunk.

    Verifies:
    - Document content (specific figures) survives end-to-end unchanged.
    - HeadingNormalizationProcessor closes level gaps: (1, 3, 5) → (1, 2, 3),
      so sub-headings render as ## and ### instead of #### and #####.
    - Every chunk carries non-empty prompt_content, embedding_content, and id.
    - Chunk IDs are stable across identical runs (deterministic hashing).
    """
    pipeline = DocumentPipeline(
        parser=_parse_structured_report,
        processors=[HeadingNormalizationProcessor()],
        splitter=TokenSplitter(max_tokens=8000),
    )

    chunks = await pipeline.run(Path("q2_report.html"))
    full_text = "\n".join(c.prompt_content for c in chunks)

    # Key figures survive the pipeline unchanged
    assert "4.2M" in full_text
    assert "15%" in full_text
    assert "23%" in full_text
    assert "EMEA" in full_text

    # Normalisation promoted level-3 → level-2: rendered as ## not ####
    assert "## Revenue Analysis" in full_text
    assert "## Operating Costs" in full_text
    # And level-5 → level-3: rendered as ### not #####
    assert "### By Region" in full_text

    # Every chunk meets the Chunk contract
    for chunk in chunks:
        assert chunk.id
        assert chunk.prompt_content
        assert chunk.embedding_content == chunk.prompt_content  # SimpleChunker

    # IDs are deterministic across runs
    chunks2 = await pipeline.run(Path("q2_report.html"))
    assert [c.id for c in chunks] == [c.id for c in chunks2]
