"""Tests for DocumentDumpProcessor and default_file_namer."""

from pathlib import Path

import pytest

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.processing.dump import DocumentDumpProcessor, default_file_namer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    return Document(
        elements=[
            Heading(innerhtml="Introduction", level=1, page=1),
            Paragraph(html_content="<p>Hello world.</p>", page=1),
        ],
        parser="html",
        source_path="/data/report.html",
        metadata={"author": "Alice", "filename": "report.html"},
    )


# ---------------------------------------------------------------------------
# default_file_namer
# ---------------------------------------------------------------------------


def test_default_file_namer(sample_document: Document):
    name = default_file_namer(sample_document)
    assert name.endswith(".ragdoc.json")
    assert name.startswith("report_")
    assert sample_document.id[:8] in name


def test_default_file_namer_empty_source_path():
    doc = Document(elements=[])
    name = default_file_namer(doc)
    assert name.endswith(".ragdoc.json")


# ---------------------------------------------------------------------------
# DocumentDumpProcessor
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dump_processor_creates_output_dir(tmp_path: Path, sample_document: Document):
    output_dir = tmp_path / "subdir" / "nested"
    proc = DocumentDumpProcessor(output_dir=output_dir)
    await proc.process(sample_document)

    assert output_dir.exists()
    assert len(list(output_dir.iterdir())) == 1


@pytest.mark.anyio
async def test_dump_processor_returns_document_unchanged(tmp_path: Path, sample_document: Document):
    proc = DocumentDumpProcessor(output_dir=tmp_path)
    result = await proc.process(sample_document)
    assert result is sample_document


@pytest.mark.anyio
async def test_dump_processor_roundtrip(tmp_path: Path, sample_document: Document):
    proc = DocumentDumpProcessor(output_dir=tmp_path)
    await proc.process(sample_document)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".ragdoc.json")

    doc2 = Document.model_validate_json(files[0].read_text(encoding="utf-8"))
    assert doc2.id == sample_document.id
    assert doc2.source_path == sample_document.source_path
    assert doc2.parser == sample_document.parser
    assert len(doc2.elements) == len(sample_document.elements)


@pytest.mark.anyio
async def test_dump_processor_custom_namer(tmp_path: Path, sample_document: Document):
    def my_namer(doc: Document) -> str:
        return "custom_output.ragdoc.json"

    proc = DocumentDumpProcessor(output_dir=tmp_path, file_namer=my_namer)
    await proc.process(sample_document)
    assert (tmp_path / "custom_output.ragdoc.json").exists()


@pytest.mark.anyio
async def test_dump_processor_multiple_documents(tmp_path: Path):
    docs = [Document(elements=[], source_path=f"/data/doc{i}.txt", parser="test") for i in range(3)]
    proc = DocumentDumpProcessor(output_dir=tmp_path)
    for doc in docs:
        await proc.process(doc)
    assert len(list(tmp_path.iterdir())) == 3
