"""Integration tests: DocumentDumpProcessor → parse_ragdoc_json / load() roundtrip."""

from pathlib import Path

import pytest

from ragdoc.document import Document, Footnote, Heading, Image, Paragraph
from ragdoc.parsing.ragdoc_json import ProvenanceMode, parse_ragdoc_json
from ragdoc.processing.dump import DocumentDumpProcessor, default_file_namer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def complex_document() -> Document:
    return Document(
        elements=[
            Heading(innerhtml="Report Title", level=1, page=1),
            Paragraph(html_content="<p>First paragraph.</p>", page=1),
            Image(
                image="aGVsbG8=",  # base64 "hello"
                image_type="jpeg",
                alt="A diagram",
                page=2,
            ),
            Footnote(
                id="fn-1",
                number=1,
                innerhtml="Footnote text here.",
                page=3,
            ),
        ],
        parser="azure_di",
        source_path="/data/report.pdf",
        metadata={"department": "IT", "year": 2024, "filename": "report.pdf"},
    )


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_roundtrip_original_mode(tmp_path: Path, complex_document: Document):
    """Dump + parse preserves id, metadata, elements, image data, and footnotes."""
    proc = DocumentDumpProcessor(output_dir=tmp_path)
    await proc.process(complex_document)
    json_path = tmp_path / default_file_namer(complex_document)

    result = await parse_ragdoc_json(json_path, ProvenanceMode.ORIGINAL)

    assert result.id == complex_document.id
    assert result.metadata == complex_document.metadata
    assert len(result.elements) == len(complex_document.elements)

    images = [e for e in result.elements if isinstance(e, Image)]
    assert images[0].image == "aGVsbG8="
    assert images[0].alt == "A diagram"

    footnotes = [e for e in result.elements if isinstance(e, Footnote)]
    assert footnotes[0].innerhtml == "Footnote text here."


@pytest.mark.anyio
async def test_roundtrip_via_load(tmp_path: Path, complex_document: Document):
    """End-to-end: dump then load() via the parser registry (ORIGINAL mode)."""
    import ragdoc.parsing  # noqa: F401 — ensures ragdoc_json is registered
    from ragdoc.parsing import load

    proc = DocumentDumpProcessor(output_dir=tmp_path)
    await proc.process(complex_document)
    json_path = tmp_path / default_file_namer(complex_document)

    result = await load(json_path)
    assert result.id == complex_document.id
    assert result.parser == complex_document.parser  # ORIGINAL preserves original parser


@pytest.mark.anyio
async def test_roundtrip_json_file_mode(tmp_path: Path, complex_document: Document):
    """JSON_FILE mode overwrites provenance with the .ragdoc.json file's info."""
    proc = DocumentDumpProcessor(output_dir=tmp_path)
    await proc.process(complex_document)
    json_path = tmp_path / default_file_namer(complex_document)

    result = await parse_ragdoc_json(json_path, ProvenanceMode.JSON_FILE)
    assert result.parser == "ragdoc_json"
    assert result.source_path == str(json_path)
    assert result.metadata["filename"] == json_path.name
