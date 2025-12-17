"""Contract test: every registered parser yields provenance-stamped documents via load().

Provenance (``document.parser``, ``document.source_path``, ``metadata["filename"]``) is
stamped centrally by :func:`ragdoc.parsing.load` (``stamp_provenance``); this test pins the
contract for every registered parser on a minimal fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.parsing import get_registered_parsers, load

DATA = Path(__file__).parent.parent / "data"
MINERU_DATA = Path(__file__).parent / "data" / "mineru"

# parser name -> fixture path (None => no offline fixture possible; skipped with reason)
_PARSER_FIXTURES: dict[str, Path | None] = {
    "html": DATA / "test.html",
    "pandoc": DATA / "test.DOCX",
    "xlsx": DATA / "test.xlsx",
    "azure_json": DATA / "test_pdf_azure_di.json",
    "azure_di": None,  # requires Azure credentials + network
    "mineru": MINERU_DATA / "bert-paper_middle.json",
    "ragdoc_json": None,  # fixture built on the fly (needs tmp_path); handled below
}


def test_every_registered_parser_has_a_contract_fixture():
    """Fail loudly when a new parser is registered without a contract-test entry."""
    registered = {reg.name for reg in get_registered_parsers()}
    missing = registered - _PARSER_FIXTURES.keys()
    assert not missing, f"Registered parsers without a provenance contract fixture: {sorted(missing)}"


def _assert_provenance(document: Document, path: Path) -> None:
    assert document.parser, "parser must be set"
    assert document.source_path == str(path)
    assert document.metadata["filename"] == path.name


@pytest.mark.parametrize("parser_name", sorted(n for n, p in _PARSER_FIXTURES.items() if p is not None))
@pytest.mark.anyio
async def test_every_registered_parser_stamps_provenance(parser_name: str):
    if parser_name not in {reg.name for reg in get_registered_parsers()}:
        pytest.skip(f"parser {parser_name!r} not registered (optional extra not installed)")
    fixture = _PARSER_FIXTURES[parser_name]
    assert fixture is not None
    document = await load(fixture, parser=parser_name)
    _assert_provenance(document, fixture)


@pytest.mark.anyio
async def test_ragdoc_json_parser_stamps_provenance(tmp_path: Path):
    """ragdoc_json (ProvenanceMode.ORIGINAL): unset fields in the dump are back-filled by load()."""
    from ragdoc.document import Paragraph

    source = tmp_path / "snapshot.ragdoc.json"
    source.write_text(Document(elements=[Paragraph(html="<p>x</p>")]).model_dump_json(), encoding="utf-8")
    document = await load(source, parser="ragdoc_json")
    _assert_provenance(document, source)


@pytest.mark.anyio
async def test_ragdoc_json_original_mode_preserves_stored_provenance(tmp_path: Path):
    """A dump carrying its original provenance keeps it — stamp_provenance only fills gaps."""
    from ragdoc.document import Paragraph

    original = Document(elements=[Paragraph(html="<p>x</p>")], source_path="/orig/report.html", parser="html")
    original.metadata["filename"] = "report.html"
    source = tmp_path / "snapshot.ragdoc.json"
    source.write_text(original.model_dump_json(), encoding="utf-8")
    document = await load(source, parser="ragdoc_json")
    assert document.parser == "html"
    assert document.source_path == "/orig/report.html"
    assert document.metadata["filename"] == "report.html"
