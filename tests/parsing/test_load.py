"""Integration tests for the unified load() entry point."""

from pathlib import Path

import pytest

from ragdoc.parsing import load


@pytest.mark.anyio
async def test_load_html(html_file_path: Path):
    doc = await load(html_file_path)
    assert doc.parser == "html"
    assert len(doc.headings) > 0


@pytest.mark.anyio
async def test_load_html_explicit_parser(html_file_path: Path):
    doc = await load(html_file_path, parser="html")
    assert doc.parser == "html"


@pytest.mark.anyio
async def test_load_xlsx(xlsx_file_path: Path):
    doc = await load(xlsx_file_path)
    assert doc.parser == "xlsx"
    assert len(doc.headings) > 0


@pytest.mark.anyio
async def test_load_unsupported():
    with pytest.raises(ValueError, match="No parser registered"):
        await load("file.unsupported_extension_xyz")


@pytest.mark.anyio
async def test_load_string_path(html_file_path: Path):
    doc = await load(str(html_file_path))
    assert doc.parser == "html"
