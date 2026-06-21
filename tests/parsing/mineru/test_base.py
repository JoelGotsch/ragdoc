"""
Tests for MinerU _middle.json parser.

This module tests parsing of MinerU intermediate output files
by iterating through all *_middle.json files in data/03_mineru_output.
"""

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

import json
import re
from pathlib import Path

import pytest

from ragdoc.parsing.mineru.base import (
    BlockType,
    DiscardedBlockType,
    MinerUMiddleDocument,
    _latex_to_text,
    parse_directory_middle_jsons,
)
from ragdoc.utils.helpers import _normalize_text


def _normalize_text_and_spaces(text: str) -> str:
    """Normalize text and also remove whitespaces."""
    normalized = _normalize_text(_latex_to_text(text))
    return re.sub(r"\s+", "", normalized).strip()


# Path to the mineru output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "mineru"
TEST_CASES_FILE = Path(__file__).parent.parent.parent / "data" / "test_cases.json"


def load_test_cases() -> dict:
    """Load test cases from JSON file."""
    if not TEST_CASES_FILE.exists():
        return {}
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        return json.load(f)


TEST_CASES = load_test_cases()


def get_mineru_file_for_test_case(test_case_name: str) -> Path | None:
    """Get the MinerU middle.json file path for a test case."""
    middle_json = DATA_DIR / f"{test_case_name}_middle.json"
    if middle_json.exists():
        return middle_json
    return None


def get_available_test_cases() -> list[str]:
    """Get test cases that have corresponding MinerU output files."""
    available = []
    for test_case_name in TEST_CASES.keys():
        if get_mineru_file_for_test_case(test_case_name):
            available.append(test_case_name)
    return available


AVAILABLE_TEST_CASES = get_available_test_cases()


def _load_mineru_document(test_case_name: str) -> MinerUMiddleDocument:
    """Load and validate a MinerU document for a test case."""
    json_file = get_mineru_file_for_test_case(test_case_name)
    with open(json_file, encoding="utf-8") as f:
        json_data = json.load(f)
    return MinerUMiddleDocument.model_validate(json_data)


def _extract_block_text(block) -> str:
    """Extract all text content from a block's lines/spans."""
    parts = []
    for line in block.lines:
        for span in line.spans:
            parts.append(span.content)
    return " ".join(parts)


def find_middle_json_files() -> list[Path]:
    """Find all *_middle.json files in the data/03_mineru_output directory."""
    if not DATA_DIR.exists():
        return []

    return list(DATA_DIR.rglob("*_middle.json"))


# Get all middle.json files for parametrization
MIDDLE_JSON_FILES = find_middle_json_files()


@pytest.mark.skipif(
    len(MIDDLE_JSON_FILES) == 0,
    reason="No *_middle.json files found in data/03_mineru_output",
)
@pytest.mark.parametrize(
    "json_file",
    MIDDLE_JSON_FILES,
    ids=[f.name.replace("_middle.json", "") for f in MIDDLE_JSON_FILES],
)
def test_parse_middle_json(json_file: Path) -> None:
    """
    Test that each *_middle.json file can be parsed by MinerUMiddleDocument.

    Args:
        json_file: Path to a *_middle.json file.
    """
    # Load the JSON data
    with open(json_file, encoding="utf-8") as f:
        json_data = json.load(f)

    # Parse with Pydantic model - this will raise ValidationError if invalid
    doc = MinerUMiddleDocument.model_validate(json_data)

    # Basic assertions
    assert doc.pdf_info is not None
    assert isinstance(doc.pdf_info, list)
    assert doc.num_pages >= 0

    # Verify each page has required fields
    for page in doc.pdf_info:
        assert page.page_idx >= 0
        assert isinstance(page.para_blocks, list)
        assert isinstance(page.discarded_blocks, list)


def test_parse_directory_middle_jsons() -> None:
    """
    Test that parse_directory_middle_jsons can parse all *_middle.json files in the directory.

    This tests the batch parsing function and ensures it can handle multiple files.
    """
    documents = parse_directory_middle_jsons(DATA_DIR)
    assert isinstance(documents, list)
    assert len(documents) == len(MIDDLE_JSON_FILES)


@pytest.mark.skipif(
    len(AVAILABLE_TEST_CASES) == 0,
    reason="No test cases with MinerU output files found",
)
@pytest.mark.parametrize("test_case_name", AVAILABLE_TEST_CASES)
def test_middle_document_contains_headings(test_case_name: str) -> None:
    """Test that expected headings exist as TitleBlocks in the raw MinerUMiddleDocument.

    Verifies that the MinerU OCR output already contains the expected heading texts
    before any middleware processing. Uses _normalize_text_and_spaces for lenient matching.
    """
    expected = TEST_CASES[test_case_name]
    expected_headings = expected.get("headings", {})
    if not expected_headings:
        pytest.skip(f"No headings defined for {test_case_name}")

    doc = _load_mineru_document(test_case_name)

    # Collect all title block texts from the raw document
    title_blocks = doc.get_all_blocks_by_type(BlockType.TITLE)
    title_texts = [_normalize_text_and_spaces(_extract_block_text(b)) for b in title_blocks]

    assert len(title_texts) > 0, f"[{test_case_name}] No title blocks found in MinerUMiddleDocument"

    for heading_name in expected_headings:
        needle = _normalize_text_and_spaces(heading_name)
        found = any(needle in t or t in needle for t in title_texts)
        assert found, (
            f"[{test_case_name}] Expected heading not found in raw title blocks: "
            f"'{heading_name}'\n  Title blocks: {title_texts[:15]}"
        )


@pytest.mark.skipif(
    len(AVAILABLE_TEST_CASES) == 0,
    reason="No test cases with MinerU output files found",
)
@pytest.mark.parametrize("test_case_name", AVAILABLE_TEST_CASES)
def test_middle_document_contains_footnotes(test_case_name: str) -> None:
    """Test that expected footnotes exist as PAGE_FOOTNOTE discarded blocks in the raw MinerUMiddleDocument.

    Verifies that the MinerU OCR output already contains the expected footnote texts
    in discarded_blocks before any middleware processing. Uses _normalize_text_and_spaces for
    lenient matching.
    """
    expected = TEST_CASES[test_case_name]
    expected_footnotes = expected.get("footnotes", [])
    if not expected_footnotes:
        pytest.skip(f"No footnotes defined for {test_case_name}")

    doc = _load_mineru_document(test_case_name)

    # Collect all page_footnote texts from discarded blocks
    footnote_texts = []
    for page in doc.pdf_info:
        for block in page.discarded_blocks:
            if block.type == DiscardedBlockType.PAGE_FOOTNOTE or block.type == DiscardedBlockType.FOOTER:
                footnote_texts.append(_normalize_text_and_spaces(_extract_block_text(block)))

    assert len(footnote_texts) > 0, (
        f"[{test_case_name}] No page_footnote discarded blocks found in MinerUMiddleDocument"
    )

    for fn in expected_footnotes:
        needle = _normalize_text_and_spaces(fn["text"])
        found = any(needle in ft or ft in needle for ft in footnote_texts)
        assert found, (
            f"[{test_case_name}] Expected footnote not found in raw discarded blocks: "
            f"'{fn['text'][:80]}...'\n"
            f"  Footnote blocks ({len(footnote_texts)}): "
            f"{[f[:60] for f in footnote_texts]}"
        )
