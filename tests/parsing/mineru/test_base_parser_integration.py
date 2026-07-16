import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

import json
import re
from pathlib import Path

import pytest

from ragdoc.document import Document, Heading, Table
from ragdoc.parsing.mineru.base import MinerUMiddleDocument
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor
from ragdoc.utils.helpers import normalize_text
from tests.latex_text import latex_to_text

# Path to test cases JSON
TEST_CASES_FILE = Path(__file__).parent.parent.parent / "data" / "test_cases.json"
FILES_DIR = Path(__file__).parent.parent / "data" / "mineru"


def normalize_text_and_spaces(text: str) -> str:
    """Normalize text and also remove whitespaces."""
    normalized = normalize_text(latex_to_text(text))
    return re.sub(r"\s+", "", normalized).strip()


def load_test_case_json() -> dict:
    """Load test cases from JSON file."""
    if not TEST_CASES_FILE.exists():
        return {}
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_test_cases() -> dict:
    """Load test cases from JSON file and their parsed _middle.json file., keyed by test case name."""
    data = load_test_case_json()
    for test_case_name in data.keys():
        middle_json = FILES_DIR / f"{test_case_name}_middle.json"
        if not middle_json.exists():
            continue
        try:
            with open(middle_json, encoding="utf-8") as f:
                json_data = json.load(f)
            mineru_doc = MinerUMiddleDocument.model_validate(json_data)
            # Run only CoreExtractor (no heading refiner, no footnote resolution)
            # Module-level asyncio.run() — acceptable: feeds @pytest.mark.parametrize at import time.
            import asyncio

            extractor = MinerUExtractor(use_default_stages=False)
            extractor.use(CoreExtractor())
            document = asyncio.run(extractor.parse(mineru_doc))
            data[test_case_name]["parsed_doc"] = document
        except Exception as e:  # noqa: BLE001 -- fixture loader; parse failures surface in test_mineru_base.py
            # should be handled/ detected by test_mineru_base.py
            print(f"Error loading {middle_json}: {e}")
    return data


TEST_CASES = load_test_cases()

_footnote_params = [
    pytest.param(name, i, fn["text"], id=f"{name}[fn{i}]")
    for name, tc in TEST_CASES.items()
    for i, fn in enumerate(tc.get("footnotes", []))
]

_heading_params = [
    pytest.param(name, heading_name, id=f"{name}[{heading_name[:40]}]")
    for name, tc in TEST_CASES.items()
    for heading_name in tc.get("headings", {})
]

_table_params = [
    pytest.param(name, i, text, id=f"{name}[table{i}]")
    for name, tc in TEST_CASES.items()
    for i, text in enumerate(tc.get("tables", []))
]


# =============================================================================
# Footnote Tests (CoreMiddleware only)
# =============================================================================


@pytest.mark.skipif(not _footnote_params, reason="No footnotes defined in test cases")
@pytest.mark.parametrize("test_case_name,fn_index,fn_text", _footnote_params)
def test_footnote_parsed_by_core_middleware(test_case_name: str, fn_index: int, fn_text: str) -> None:
    """Test that CoreExtractor extracts each expected footnote."""
    document: Document = TEST_CASES[test_case_name].get("parsed_doc")
    parsed_footnotes = [normalize_text_and_spaces(fn.text) for fn in document.footnotes]

    assert len(parsed_footnotes) > 0, f"[{test_case_name}] No footnotes extracted by CoreExtractor"

    needle = normalize_text_and_spaces(fn_text)
    found = any(needle in parsed or parsed in needle for parsed in parsed_footnotes)
    assert found, (
        f"[{test_case_name}] fn{fn_index} not found: '{fn_text[:80]}'\n"
        f"  Parsed footnotes ({len(parsed_footnotes)}): {[f[:60] for f in parsed_footnotes]}"
    )


# =============================================================================
# Heading Detection Tests (presence only)
# =============================================================================


@pytest.mark.skipif(not _heading_params, reason="No headings defined in test cases")
@pytest.mark.parametrize("test_case_name,heading_name", _heading_params)
def test_heading_detected(test_case_name: str, heading_name: str) -> None:
    """Test that each expected heading is present in the parsed document."""
    document = TEST_CASES[test_case_name].get("parsed_doc")
    parsed_heading_texts = [normalize_text_and_spaces(el.text) for el in document.elements if isinstance(el, Heading)]

    assert len(parsed_heading_texts) > 0, f"[{test_case_name}] No headings extracted"

    needle = normalize_text_and_spaces(heading_name)
    found = any(needle in h or h in needle for h in parsed_heading_texts)
    assert found, (
        f"[{test_case_name}] Expected heading not found: '{heading_name}'\n"
        f"  Parsed headings: {parsed_heading_texts[:15]}"
    )


# =============================================================================
# Table Extraction Tests
# =============================================================================


@pytest.mark.skipif(not _table_params, reason="No tables defined in test cases")
@pytest.mark.parametrize("test_case_name,table_index,expected_text", _table_params)
def test_table_parsed_by_core_middleware(test_case_name: str, table_index: int, expected_text: str) -> None:
    """Test that CoreExtractor extracts tables with expected HTML content."""
    document: Document = TEST_CASES[test_case_name].get("parsed_doc")
    tables = [el for el in document.elements if isinstance(el, Table)]

    assert len(tables) > 0, f"[{test_case_name}] No tables extracted by CoreExtractor"

    found = any(expected_text in table.html for table in tables)
    assert found, (
        f"[{test_case_name}] table{table_index}: expected text not found in any table: '{expected_text}'\n"
        f"  Parsed tables ({len(tables)}): {[t.html[:80] for t in tables]}"
    )


# =============================================================================
# Parser Field Tests
# =============================================================================


@pytest.mark.anyio
async def test_parser_field_set_to_mineru():
    """MinerUExtractor.parse() must set document.parser = 'mineru'."""
    doc = MinerUMiddleDocument(pdf_info=[])
    result = await MinerUExtractor().parse(doc)
    assert result.parser == "mineru"


def test_heading_html_contains_css():
    """CoreExtractor must embed CSS font-size on the heading outer tag."""
    # Use first available _middle.json fixture that has at least one heading
    for name, tc in TEST_CASES.items():
        document = tc.get("parsed_doc")
        if document is None:
            continue
        headings = [el for el in document.elements if isinstance(el, Heading)]
        if not headings:
            continue
        # At least some headings should have CSS on the outer tag
        css_headings = [h for h in headings if "style=" in h.html]
        assert len(css_headings) > 0, (
            f"[{name}] No headings have CSS style on outer tag. Sample html: {headings[0].html[:100]}"
        )
        # Verify font-size is present in at least one heading
        font_size_headings = [h for h in css_headings if "font-size" in h.html]
        assert len(font_size_headings) > 0, (
            f"[{name}] No headings have font-size CSS. Sample html: {css_headings[0].html[:100]}"
        )
        return  # Passed for at least one fixture, that's enough
