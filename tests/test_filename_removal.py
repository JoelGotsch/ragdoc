"""Phase 0 TDD tests: verify filename has been removed from Document and Chunk.

These tests cover cross-cutting concerns after removing the first-class
``filename`` field from :class:`~ragdoc.document.Document` and
:class:`~ragdoc.chunking.chunk.Chunk`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.chunking.chunk import Chunk
from ragdoc.document import Document, Heading, Paragraph
from ragdoc.processing.dump import default_file_namer
from ragdoc.splitting.base import split_by_headings

# ---------------------------------------------------------------------------
# Phase 0a — model fields
# ---------------------------------------------------------------------------


def test_document_has_no_filename_field():
    doc = Document(elements=[])
    assert not hasattr(doc, "filename")


def test_chunk_has_no_filename_field():
    chunk = Chunk(prompt_content="x", embedding_content="x", source_id="s", source_hash="h")
    assert not hasattr(chunk, "filename")


# ---------------------------------------------------------------------------
# Phase 0b — parser writes metadata["filename"]
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_html_parser_sets_metadata_filename(tmp_path: Path):
    from ragdoc.parsing import load

    html_file = tmp_path / "report.html"
    html_file.write_text("<html><body><h1>Hello</h1></body></html>", encoding="utf-8")

    doc = await load(html_file)
    assert "filename" in doc.metadata
    assert doc.metadata["filename"] == "report.html"


# ---------------------------------------------------------------------------
# Phase 0c — metadata["filename"] propagates through splits
# ---------------------------------------------------------------------------


def test_filename_propagates_through_split():
    doc = Document(
        elements=[
            Heading(html="<h1>Section A</h1>"),
            Paragraph(html="<p>Text A.</p>"),
            Heading(html="<h1>Section B</h1>"),
            Paragraph(html="<p>Text B.</p>"),
        ],
        metadata={"filename": "report.pdf"},
        source_path="/x/report.pdf",
    )
    splits = split_by_headings(doc)
    assert len(splits) > 1
    for s in splits:
        assert s.metadata["filename"] == "report.pdf"


# ---------------------------------------------------------------------------
# Phase 0d — dump.py derives stem from source_path
# ---------------------------------------------------------------------------


def test_default_file_namer_uses_source_path():
    doc = Document(
        elements=[],
        source_path="/data/reports/report.pdf",
        metadata={"filename": "report.pdf"},
    )
    name = default_file_namer(doc)
    assert name.startswith("report_")
    assert name.endswith(".ragdoc.json")


def test_default_file_namer_fallback_no_source_path():
    doc = Document(elements=[])
    name = default_file_namer(doc)
    assert name.endswith(".ragdoc.json")
    # Only id prefix — no stem separator
    assert "_" not in name
