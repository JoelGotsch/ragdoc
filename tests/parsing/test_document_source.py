"""Tests for DocumentSource types, load_document, and from_path (Phase 4.6)."""
import pytest
from pathlib import Path

from ragdoc.parsing import (
    HTMLSource,
    HTMLFile,
    ExcelSource,
    ExcelPackage,
    PandocFile,
    WordFile,
    AzureJSONFile,
    DocumentSource,
    load_document,
    from_path,
)


# =============================================================================
# Rename aliases
# =============================================================================

def test_htmlfile_is_htmlsource():
    """HTMLFile is an alias for HTMLSource."""
    assert HTMLFile is HTMLSource


def test_excelpackage_is_excelsource():
    """ExcelPackage is an alias for ExcelSource."""
    assert ExcelPackage is ExcelSource


# =============================================================================
# from_path factory
# =============================================================================

def test_from_path_html(html_file_path: Path):
    source = from_path(html_file_path)
    assert isinstance(source, HTMLSource)
    assert Path(source.file_path) == html_file_path


def test_from_path_docx(docx_file_path: Path):
    source = from_path(docx_file_path)
    assert isinstance(source, PandocFile)


def test_from_path_xlsx(xlsx_file_path: Path):
    source = from_path(xlsx_file_path)
    assert isinstance(source, ExcelSource)


def test_from_path_azure_json(azure_test_pdf_json_file_path: Path):
    # Rename to *.azure.json to trigger Azure branch
    azure_path = azure_test_pdf_json_file_path.parent / (
        azure_test_pdf_json_file_path.stem + ".azure.json"
    )
    source = from_path(azure_path)
    assert isinstance(source, AzureJSONFile)


def test_from_path_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        from_path(Path("document.pdf"))


def test_from_path_string_input(html_file_path: Path):
    source = from_path(str(html_file_path))
    assert isinstance(source, HTMLSource)


# =============================================================================
# load_document
# =============================================================================

def test_load_document_html(html_file_path: Path):
    source = from_path(html_file_path)
    document = load_document(source)
    assert document.parser == "html"
    assert len(document.headings) > 0


def test_load_document_xlsx(xlsx_file_path: Path):
    source = from_path(xlsx_file_path)
    document = load_document(source)
    assert document.parser == "xlsx"
    assert len(document.headings) > 0
