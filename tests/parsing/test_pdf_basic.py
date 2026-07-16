"""Tests for the pdf_basic parser (pymupdf text + font-size headings, 'pdf' extra)."""

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ragdoc.document import Heading, Paragraph
from ragdoc.parsing import load
from ragdoc.parsing.pdf_basic.load import parse_pdf_basic

# ---------------------------------------------------------------------------
# Fixture PDF generation
# ---------------------------------------------------------------------------


def _write_two_size_pdf(path: Path) -> None:
    """A PDF with one 24pt title line and several 11pt body lines."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Document Title", fontsize=24)
    y = 150
    for i in range(5):
        page.insert_text((72, y), f"Body paragraph line number {i} with some plain text.", fontsize=11)
        y += 20
    doc.save(path)
    doc.close()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pdf_basic_parses_fixture(pdf_file_path: Path):
    doc = await load(pdf_file_path, parser="pdf_basic")
    assert doc.parser == "pdf_basic"
    assert doc.source_path == str(pdf_file_path)
    assert doc.metadata["filename"] == pdf_file_path.name
    assert any(isinstance(el, Paragraph) for el in doc.elements)


@pytest.mark.anyio
async def test_pdf_basic_detects_headings_by_font_size(tmp_path: Path):
    pdf = tmp_path / "two_sizes.pdf"
    _write_two_size_pdf(pdf)

    doc = await load(pdf, parser="pdf_basic")

    headings = [el for el in doc.elements if isinstance(el, Heading)]
    assert headings, "the 24pt line must become a Heading"
    assert headings[0].level == 1
    assert "Document Title" in headings[0].text

    paragraphs = [el for el in doc.elements if isinstance(el, Paragraph)]
    assert len(paragraphs) == 5
    assert all("Body paragraph" in p.text for p in paragraphs)


def test_pdf_basic_emits_font_size_css(tmp_path: Path):
    """Font sizes are emitted as inline CSS so downstream visual processors keep working."""
    pdf = tmp_path / "css.pdf"
    _write_two_size_pdf(pdf)
    doc = parse_pdf_basic(pdf)
    heading = next(el for el in doc.elements if isinstance(el, Heading))
    assert "font-size:24" in heading.html
    paragraph = next(el for el in doc.elements if isinstance(el, Paragraph))
    assert "font-size:11" in paragraph.html


def test_pdf_basic_empty_pdf_yields_empty_document(tmp_path: Path):
    pdf = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    parsed = parse_pdf_basic(pdf)
    assert parsed.parser == "pdf_basic"
    assert parsed.elements == []


# ---------------------------------------------------------------------------
# Registration / resolution
# ---------------------------------------------------------------------------


def test_pdf_basic_registered_at_priority_10():
    from ragdoc.parsing.registry import get_registered_parsers

    regs = [r for r in get_registered_parsers() if r.name == "pdf_basic"]
    assert regs and regs[0].priority == 10
    assert regs[0].parser.is_available()  # pymupdf is installed in the test env


@pytest.mark.anyio
async def test_pdf_resolution_prefers_higher_fidelity(tmp_path: Path):
    """With azure creds configured (and the azure-di extra), .pdf resolves to azure_di (40 > 10)."""
    pytest.importorskip("azure.ai.documentintelligence")
    from ragdoc.config import RagdocConfig, configure
    from ragdoc.parsing.registry import _resolve_parser

    async with configure(RagdocConfig(azure_key="k", azure_endpoint="https://example.invalid")):
        resolved = _resolve_parser(Path("report.pdf"))
        assert resolved.name == "azure_di"


def test_pdf_resolution_falls_back_to_pdf_basic_without_creds():
    """Without azure credentials, .pdf resolves to pdf_basic at resolve time."""
    from ragdoc.parsing.registry import _resolve_parser

    resolved = _resolve_parser(Path("report.pdf"))
    assert resolved.name == "pdf_basic"
