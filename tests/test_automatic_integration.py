"""Integration tests for the registry-based load() entry point.

Replaces the old test_automatic_integration.py that tested automatic_load_file().
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ragdoc.parsing import load

try:
    from ragdoc.parsing.azure_di.client import di_available
except ImportError:
    di_available = False

_skip_no_azure = pytest.mark.skipif(not di_available, reason="Azure SDK not installed")


@pytest.mark.anyio
@pytest.mark.parametrize("path_fixture", ["html_file_path", "docx_file_path", "xlsx_file_path"])
async def test_pragmatic_paths(path_fixture, request):
    """load() resolves the correct parser even with many intermediate suffixes."""
    path = request.getfixturevalue(path_fixture)
    name, rest = path.name.split(".")
    new_file = ".".join([name, "many", "fun", "suffixes", rest])
    path = path.parent / new_file
    d = await load(path)
    assert len(d.headings) > 0


@_skip_no_azure
@pytest.mark.anyio
async def test_pdf_pragmatic_paths(pdf_file_path: Path, azure_test_pdf_json_file_path: Path):
    with open(azure_test_pdf_json_file_path, "rb") as f:
        analyze_dict = json.load(f)

    with patch("ragdoc.parsing.azure_di.client.get_analyze_result", return_value=analyze_dict):
        name, rest = pdf_file_path.name.split(".")
        new_file = ".".join([name, "many", "fun", "suffixes", rest])
        path = pdf_file_path.parent / new_file
        d = await load(path)

    assert len(d.headings) > 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "fixture_name, expected_heading_count, expected_first_heading",
    [
        ("html_file_path", 6, "New super amazing Document"),
        ("docx_file_path", 6, "New super amazing Document"),
        ("xlsx_file_path", 4, "Sheet1"),
    ],
)
async def test_load(
    fixture_name: str,
    expected_heading_count: int,
    expected_first_heading: str,
    request: pytest.FixtureRequest,
):
    path = request.getfixturevalue(fixture_name)
    document = await load(path)
    assert len(document.headings) == expected_heading_count
    assert document.headings[0].text == expected_first_heading
    assert document.source_path
    assert document.metadata.get("filename")


@_skip_no_azure
@pytest.mark.anyio
async def test_pdf(pdf_file_path: Path, azure_test_pdf_json_file_path: Path):
    with open(azure_test_pdf_json_file_path, "rb") as f:
        analyze_dict = json.load(f)

    with patch("ragdoc.parsing.azure_di.client.get_analyze_result", return_value=analyze_dict):
        document = await load(pdf_file_path)

    assert len(document.headings) == 5
    assert document.tables and "I am a super merged" in document.tables[0].html
    assert document.metadata.get("filename")


@pytest.mark.anyio
async def test_pdf_no_di(pdf_file_path: Path):
    with patch("ragdoc.parsing.azure_di.client.di_available", False):
        with pytest.raises(ImportError):
            await load(pdf_file_path)
