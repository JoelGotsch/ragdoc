"""Cross-module pytest fixtures and test session setup."""

import logging
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ragdoc_file_log(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped DEBUG file log in a temp dir (never in the repo/CWD)."""
    logger = logging.getLogger("ragdoc")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(tmp_path_factory.mktemp("logs") / "test.log")
    logger.addHandler(handler)
    yield
    logger.removeHandler(handler)
    handler.close()


@pytest.fixture(scope="session")
def data_path() -> Path:
    """Root directory for cross-module test data."""
    return Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# File fixtures used across multiple test modules (automatic, parsing, rendering)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def html_file_path(data_path: Path) -> Path:
    return data_path / "test.html"


@pytest.fixture(scope="session")
def docx_file_path(data_path: Path) -> Path:
    return data_path / "test.DOCX"


@pytest.fixture(scope="session")
def xlsx_file_path(data_path: Path) -> Path:
    return data_path / "test.xlsx"


@pytest.fixture(scope="session")
def pdf_file_path(data_path: Path) -> Path:
    return data_path / "test.pdf"


@pytest.fixture(scope="session")
def azure_test_pdf_json_file_path(data_path: Path) -> Path:
    return data_path / "test_pdf_azure_di.json"


@pytest.fixture(scope="session")
def azure_test_pdf_json_file_path_no_figures(data_path: Path) -> Path:
    return data_path / "test_pdf_azure_di_no_figures.json"


@pytest.fixture(scope="session")
def description_list_file_path() -> Path:
    return Path(__file__).parent / "parsing" / "data" / "html" / "test_description_list.html"
