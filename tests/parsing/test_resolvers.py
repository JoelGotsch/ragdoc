"""Tests for secondary-file resolvers."""

from pathlib import Path

from ragdoc.parsing.resolvers import sibling_resolver


def test_sibling_resolver_found(tmp_path: Path):
    """sibling_resolver returns the sibling path when the file exists."""
    docx = tmp_path / "report.docx"
    pdf = tmp_path / "report.pdf"
    docx.write_text("dummy")
    pdf.write_text("dummy")

    resolve = sibling_resolver(".pdf")
    result = resolve(docx)
    assert result == pdf


def test_sibling_resolver_not_found(tmp_path: Path):
    """sibling_resolver returns None when the sibling file doesn't exist."""
    docx = tmp_path / "report.docx"
    docx.write_text("dummy")

    resolve = sibling_resolver(".pdf")
    result = resolve(docx)
    assert result is None


def test_sibling_resolver_different_suffix(tmp_path: Path):
    """sibling_resolver works for arbitrary suffixes."""
    html = tmp_path / "data.html"
    json_f = tmp_path / "data.json"
    html.write_text("dummy")
    json_f.write_text("dummy")

    resolve = sibling_resolver(".json")
    assert resolve(html) == json_f
