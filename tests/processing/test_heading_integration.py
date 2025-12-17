"""
Integration tests for HeadingLevelProcessor and TitleDetectionProcessor
applied to the MinerU test fixtures.

Test matrix (via parametrization):
- HeadingLevelProcessor  × document × expected heading
- TitleDetectionProcessor × document
"""

import json
import re

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

from pathlib import Path

from ragdoc.document import Document, Heading
from ragdoc.parsing.mineru.base import MinerUMiddleDocument
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor as MinerUParser
from ragdoc.processing.heading import HeadingLevelProcessor, TitleDetectionProcessor
from ragdoc.utils.helpers import normalize_text
from tests.latex_text import latex_to_text

TEST_CASES_FILE = Path(__file__).parent.parent / "data" / "test_cases.json"
FILES_DIR = Path(__file__).parent.parent / "parsing" / "data" / "mineru"


# =============================================================================
# Helpers
# =============================================================================


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(latex_to_text(text))).strip()


def _load_raw() -> tuple[dict, dict[str, MinerUMiddleDocument]]:
    """Return (test_cases_json, {name: MinerUMiddleDocument})."""
    if not TEST_CASES_FILE.exists():
        return {}, {}
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    middle_docs: dict[str, MinerUMiddleDocument] = {}
    for name in cases:
        path = FILES_DIR / f"{name}_middle.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            middle_docs[name] = MinerUMiddleDocument.model_validate(json.load(f))
    return cases, middle_docs


_TEST_CASES, _MIDDLE_DOCS = _load_raw()


async def _parse_fresh(name: str) -> Document:
    """Return a freshly parsed Document (no processing applied)."""
    parser = MinerUParser(use_default_stages=False)
    parser.use(CoreExtractor())
    return await parser.parse(_MIDDLE_DOCS[name])


# =============================================================================
# Parametrize helpers
# =============================================================================

# (test_case_name, heading_text, expected_level) for every heading in every doc
_heading_params = [
    pytest.param(name, heading_text, expected_level, id=f"{name}[{heading_text[:40]}]")
    for name, tc in _TEST_CASES.items()
    if name in _MIDDLE_DOCS
    for heading_text, expected_level in tc.get("headings", {}).items()
]

# Exact-level expectations HeadingLevelProcessor currently misses — a machine-checked
# bug tracker. strict=True: fixing one of these forces promoting it out of this set.
_KNOWN_EXACT_LEVEL_FAILURES: set[tuple[str, str]] = {
    ("tesla-q4-2024-update", "Cash"),
    ("tesla-q4-2024-update", "Operations"),
    ("tesla-q4-2024-update", "Revenue"),
    ("attention-is-all-you-need", "1 Introduction"),
    ("attention-is-all-you-need", "2 Background"),
    ("attention-is-all-you-need", "3 Model Architecture"),
    ("attention-is-all-you-need", "3.1 Encoder and Decoder Stacks"),
    ("attention-is-all-you-need", "3.2 Attention"),
    ("attention-is-all-you-need", "3.2.1 Scaled Dot-Product Attention"),
    ("attention-is-all-you-need", "3.2.2 Multi-Head Attention"),
    ("bert-paper", "Abstract"),
    ("bert-paper", "1 Introduction"),
    ("bert-paper", "2 Related Work"),
    ("bert-paper", "2.1 Unsupervised Feature-based Approaches"),
    ("bert-paper", "2.2 Unsupervised Fine-tuning Approaches"),
    ("bert-paper", "2.3 Transfer Learning from Supervised Data"),
    ("bert-paper", "3 BERT"),
    ("bert-paper", "3.1 Pre-training BERT"),
    ("bert-paper", "3.2 Fine-tuning BERT"),
    ("bert-paper", "4 Experiments"),
    ("vw-sustainability-2023", "Contents"),
    ("vw-sustainability-2023", "Decarbonization"),
    ("vw-sustainability-2023", "Circular Economy"),
}

_exact_level_params = [
    pytest.param(
        name,
        heading_text,
        expected_level,
        id=f"{name}[{heading_text[:40]}]",
        marks=[
            pytest.mark.xfail(
                reason="HeadingLevelProcessor does not assign the exact expected level for this heading",
                strict=True,
            )
        ]
        if (name, heading_text) in _KNOWN_EXACT_LEVEL_FAILURES
        else [],
    )
    for name, tc in _TEST_CASES.items()
    if name in _MIDDLE_DOCS
    for heading_text, expected_level in tc.get("headings", {}).items()
]

# (test_case_name, expected_title) for every doc that declares a title
_title_params = [
    pytest.param(name, tc["title"], id=name)
    for name, tc in _TEST_CASES.items()
    if name in _MIDDLE_DOCS and "title" in tc
]


# =============================================================================
# HeadingLevelProcessor tests
# =============================================================================


@pytest.mark.anyio
@pytest.mark.skipif(not _heading_params, reason="No headings defined in test cases")
@pytest.mark.parametrize("test_case_name,heading_text,expected_level", _heading_params)
async def test_heading_level_in_valid_range(test_case_name: str, heading_text: str, expected_level: int) -> None:
    """After HeadingLevelProcessor every heading level must be in 1–6."""
    doc = await _parse_fresh(test_case_name)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)

    headings = [el for el in doc.elements if isinstance(el, Heading)]
    assert headings, f"[{test_case_name}] No headings remain after HeadingLevelProcessor"
    for h in headings:
        assert 1 <= h.level <= 6, f"[{test_case_name}] '{h.text[:50]}' has out-of-range level {h.level}"


@pytest.mark.anyio
@pytest.mark.skipif(not _heading_params, reason="No headings defined in test cases")
@pytest.mark.parametrize("test_case_name,heading_text,expected_level", _heading_params)
async def test_heading_present_after_level_processor(
    test_case_name: str, heading_text: str, expected_level: int
) -> None:
    """HeadingLevelProcessor must not discard any expected heading."""
    doc = await _parse_fresh(test_case_name)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)

    parsed = [_norm(el.text) for el in doc.elements if isinstance(el, Heading)]
    needle = _norm(heading_text)
    assert any(needle in h or h in needle for h in parsed), (
        f"[{test_case_name}] Expected heading not found after HeadingLevelProcessor: "
        f"'{heading_text}'\n  Parsed headings (first 10): {parsed[:10]}"
    )


@pytest.mark.anyio
@pytest.mark.skipif(not _exact_level_params, reason="No headings defined in test cases")
@pytest.mark.parametrize("test_case_name,heading_text,expected_level", _exact_level_params)
async def test_heading_exact_level(test_case_name: str, heading_text: str, expected_level: int) -> None:
    """HeadingLevelProcessor must assign each heading exactly the expected level."""
    doc = await _parse_fresh(test_case_name)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)

    needle = _norm(heading_text)
    target = next(
        (
            el
            for el in doc.elements
            if isinstance(el, Heading) and (needle in _norm(el.text) or _norm(el.text) in needle)
        ),
        None,
    )
    if target is None:
        pytest.skip(f"Heading '{heading_text}' not found — covered by presence test")

    assert target.level == expected_level, (
        f"[{test_case_name}] '{heading_text}': expected level {expected_level}, got {target.level}"
    )


@pytest.mark.anyio
@pytest.mark.skipif(not _heading_params, reason="No headings defined in test cases")
@pytest.mark.parametrize("test_case_name,heading_text,expected_level", _heading_params)
async def test_heading_relative_ordering_preserved(test_case_name: str, heading_text: str, expected_level: int) -> None:
    """
    For headings with expected_level > 1, their assigned level must be >=
    the assigned level of any expected_level-1 heading in the same document.
    This verifies that HeadingLevelProcessor respects the visual hierarchy.
    """
    if expected_level == 1:
        pytest.skip("Ordering only verified for sub-headings (expected_level > 1)")

    doc = await _parse_fresh(test_case_name)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)

    needle = _norm(heading_text)
    target = next(
        (
            el
            for el in doc.elements
            if isinstance(el, Heading) and (needle in _norm(el.text) or _norm(el.text) in needle)
        ),
        None,
    )
    if target is None:
        pytest.skip(f"Heading '{heading_text}' not found — covered by presence test")

    # Collect assigned levels for all expected level-1 headings
    top_level_norms = {_norm(ht) for ht, lvl in _TEST_CASES[test_case_name].get("headings", {}).items() if lvl == 1}
    top_assigned = [
        el.level
        for el in doc.elements
        if isinstance(el, Heading) and any(_norm(el.text) in n or n in _norm(el.text) for n in top_level_norms)
    ]
    if not top_assigned:
        pytest.skip("No level-1 reference headings resolved — cannot check ordering")

    min_top = min(top_assigned)
    assert target.level >= min_top, (
        f"[{test_case_name}] Sub-heading '{heading_text}' (expected level {expected_level}) "
        f"was assigned level {target.level}, below top-level headings (min assigned={min_top})"
    )


# =============================================================================
# TitleDetectionProcessor tests
# =============================================================================


@pytest.mark.anyio
@pytest.mark.skipif(not _title_params, reason="No titles defined in test cases")
@pytest.mark.parametrize("test_case_name,expected_title", _title_params)
async def test_title_detected(test_case_name: str, expected_title: str) -> None:
    """
    After HeadingLevelProcessor + TitleDetectionProcessor, document.title
    must match the expected title from test_cases.json.
    """
    doc = await _parse_fresh(test_case_name)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)
    doc = await TitleDetectionProcessor(remove_title_from_elements=False).process(doc)

    assert doc.title is not None, f"[{test_case_name}] TitleDetectionProcessor did not set document.title"
    assert _norm(doc.title) == _norm(expected_title), (
        f"[{test_case_name}] Wrong title.\n  Expected: '{expected_title}'\n  Got:      '{doc.title}'"
    )


@pytest.mark.anyio
@pytest.mark.skipif(not _title_params, reason="No titles defined in test cases")
@pytest.mark.parametrize("test_case_name,expected_title", _title_params)
async def test_title_removed_from_elements(test_case_name: str, expected_title: str) -> None:
    """With remove_title_from_elements=True (default), the title heading must not remain as a Heading element."""
    doc = await _parse_fresh(test_case_name)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)
    doc = await TitleDetectionProcessor(remove_title_from_elements=True).process(doc)

    if doc.title is None:
        pytest.skip(f"[{test_case_name}] Title not detected — covered by test_title_detected")

    heading_texts = [_norm(el.text) for el in doc.elements if isinstance(el, Heading)]
    title_norm = _norm(doc.title)
    # Exact match: the title heading is gone. Substring fuzziness produces
    # false positives when the title is referenced elsewhere in the doc
    # (e.g. "Appendix for «<title>»" in academic papers).
    assert title_norm not in heading_texts, (
        f"[{test_case_name}] Title '{doc.title}' still present as a Heading element after removal"
    )
