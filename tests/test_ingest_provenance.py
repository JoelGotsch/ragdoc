"""IngestPipeline provenance stamping: the explicit-parser path must match parsing.load().

Regression suite for the bug where ``IngestPipeline.run`` with an explicit ``parser=``
bypassed :func:`ragdoc.parsing.registry.stamp_provenance`, leaving ``document.parser``,
``source_path``, and ``metadata["filename"]`` unset — and where ``ChunkPipeline.run``'s
metadata validation papered over it by exempting ``filename`` from the required keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document, Paragraph
from ragdoc.metadata import BaseMetadata
from ragdoc.pipeline import ChunkPipeline, IngestPipeline


async def _bare_parser(path: Path) -> Document:
    """A minimal custom parser that stamps no provenance at all."""
    return Document(elements=[Paragraph(html="<p>content</p>")])


@pytest.mark.anyio
async def test_explicit_parser_path_stamps_provenance(tmp_path: Path):
    source = tmp_path / "report.txt"
    source.write_text("content", encoding="utf-8")

    doc = await IngestPipeline(parser=_bare_parser).run(source)

    assert doc is not None
    assert doc.metadata["filename"] == "report.txt"
    assert doc.source_path == str(source)
    assert doc.parser is not None
    assert doc.source_id == "report.txt"


@pytest.mark.anyio
async def test_explicit_parser_provenance_not_overwritten(tmp_path: Path):
    """Parser-set provenance wins — stamping only fills the gaps (same rule as load())."""

    async def opinionated_parser(path: Path) -> Document:
        doc = Document(elements=[Paragraph(html="<p>x</p>")])
        doc.parser = "custom"
        doc.source_path = "/original/location.txt"
        doc.metadata["filename"] = "original.txt"
        return doc

    source = tmp_path / "renamed.txt"
    source.write_text("x", encoding="utf-8")

    doc = await IngestPipeline(parser=opinionated_parser).run(source)

    assert doc is not None
    assert doc.parser == "custom"
    assert doc.source_path == "/original/location.txt"
    assert doc.metadata["filename"] == "original.txt"


@pytest.mark.anyio
async def test_metadata_validation_requires_filename_without_exemption(tmp_path: Path):
    """ChunkPipeline(metadata_type=BaseMetadata) enforces Required[filename] — no carve-out."""
    doc_missing = Document(elements=[Paragraph(html="<p>x</p>")])
    with pytest.raises(ValueError, match="filename"):
        await ChunkPipeline(metadata_type=BaseMetadata).run(doc_missing)

    doc_ok = Document(elements=[Paragraph(html="<p>x</p>")], metadata={"filename": "a.txt"})
    chunks = await ChunkPipeline(metadata_type=BaseMetadata).run(doc_ok)
    assert len(chunks) >= 1
