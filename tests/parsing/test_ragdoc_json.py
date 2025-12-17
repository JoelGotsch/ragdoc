"""Tests for RagdocJsonParser, parse_ragdoc_json, and ProvenanceMode."""
import pytest
from pathlib import Path

from ragdoc.document import Document, Heading
from ragdoc.parsing.ragdoc_json import ProvenanceMode, RagdocJsonParser, parse_ragdoc_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_doc_json(path: Path, doc: Document) -> None:
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")


def _make_doc(**kwargs) -> Document:
    defaults: dict = dict(
        elements=[Heading(innerhtml="Title", level=1, page=1)],
        parser="html",
        source_path="/data/source.html",
        metadata={"filename": "source.html"},
    )
    defaults.update(kwargs)
    return Document(**defaults)


# ---------------------------------------------------------------------------
# parse_ragdoc_json — ORIGINAL mode
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parse_ragdoc_json_original_preserves_provenance(tmp_path: Path):
    doc = _make_doc(source_path="/orig/orig.html", parser="html", metadata={"filename": "orig.html"})
    json_path = tmp_path / "doc.ragdoc.json"
    _write_doc_json(json_path, doc)

    result = await parse_ragdoc_json(json_path, ProvenanceMode.ORIGINAL)
    assert result.metadata["filename"] == "orig.html"
    assert result.source_path == "/orig/orig.html"
    assert result.parser == "html"
    assert result.id == doc.id
    assert len(result.elements) == len(doc.elements)


# ---------------------------------------------------------------------------
# parse_ragdoc_json — JSON_FILE mode
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parse_ragdoc_json_json_file_overwrites_provenance(tmp_path: Path):
    doc = _make_doc(source_path="/orig/orig.html", parser="html", metadata={"filename": "orig.html"})
    json_path = tmp_path / "snapshot.ragdoc.json"
    _write_doc_json(json_path, doc)

    result = await parse_ragdoc_json(json_path, ProvenanceMode.JSON_FILE)
    assert result.metadata["filename"] == "snapshot.ragdoc.json"
    assert result.source_path == str(json_path)
    assert result.parser == "ragdoc_json"


# ---------------------------------------------------------------------------
# RagdocJsonParser.__call__
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ragdoc_json_parser_call_original_mode(tmp_path: Path):
    doc = _make_doc(source_path="/src.html", parser="html", metadata={"filename": "src.html"})
    json_path = tmp_path / "doc.ragdoc.json"
    _write_doc_json(json_path, doc)

    result = await RagdocJsonParser()(json_path)
    assert result.metadata["filename"] == "src.html"
    assert result.parser == "html"


@pytest.mark.anyio
async def test_ragdoc_json_parser_call_json_file_mode(tmp_path: Path):
    doc = _make_doc()
    json_path = tmp_path / "snap.ragdoc.json"
    _write_doc_json(json_path, doc)

    result = await RagdocJsonParser(provenance_mode=ProvenanceMode.JSON_FILE)(json_path)
    assert result.metadata["filename"] == "snap.ragdoc.json"
    assert result.parser == "ragdoc_json"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_ragdoc_json_registered_after_import():
    """Importing ragdoc.parsing should register the ragdoc_json parser."""
    import ragdoc.parsing  # noqa: F401

    from ragdoc.parsing.registry import get_parser

    get_parser("ragdoc_json")  # raises if not registered


def test_ragdoc_json_resolves_by_extension(tmp_path: Path):
    """_resolve_parser should pick ragdoc_json for .ragdoc.json files."""
    import ragdoc.parsing  # noqa: F401

    from ragdoc.parsing.registry import _resolve_parser

    resolved = _resolve_parser(tmp_path / "file.ragdoc.json")
    assert resolved.name == "ragdoc_json"
