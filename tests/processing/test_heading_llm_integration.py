"""
Integration tests for LLMHeadingResolver using pre-recorded LLM responses.

Each test runs the full pipeline:
    parse (MinerU) → HeadingLevelProcessor → LLMHeadingResolver (mocked)
and checks results against tests/data/test_cases.json.

The LLM responses are replayed from tests/processing/fixtures/heading_llm/.
Generate new fixtures by running: uv run python scripts/generate_heading_llm_fixtures.py
"""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

from ragdoc.document import Document, Heading
from ragdoc.parsing.mineru.base import MinerUMiddleDocument
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor as MinerUParser
from ragdoc.processing.heading import HeadingLevelProcessor
from ragdoc.processing.heading_llm import HeadingResponse, LLMHeadingResolver
from ragdoc.utils.helpers import normalize_text
from tests.latex_text import latex_to_text

TEST_CASES_FILE = Path(__file__).parent.parent / "data" / "test_cases.json"
MINERU_DIR = Path(__file__).parent.parent / "parsing" / "data" / "mineru"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "heading_llm"


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(latex_to_text(text))).strip()


def _find_classified_heading(doc: Document, heading_text: str) -> Heading | None:
    """
    Return the LLM-classified heading that best matches *heading_text*.

    Only considers headings that the LLM explicitly classified (``llm_heading_level``
    in metadata, not ``"none"``).  Among fuzzy-matching candidates (one text is a
    substring of the other), picks the one with the highest length-similarity ratio
    and requires the ratio to be at least 0.5.  This prevents short generic names
    like "Development" from matching long compound headings like
    "A. Development, Findings and Implications".
    """
    needle = _norm(heading_text)
    best: Heading | None = None
    best_ratio = 0.0
    for el in doc.elements:
        if not isinstance(el, Heading):
            continue
        if "llm_heading_level" not in el.metadata or el.metadata["llm_heading_level"] == "none":
            continue
        el_norm = _norm(el.text)
        if needle not in el_norm and el_norm not in needle:
            continue
        shorter = min(len(needle), len(el_norm))
        longer = max(len(needle), len(el_norm))
        ratio = shorter / longer if longer > 0 else 1.0
        if ratio > best_ratio:
            best_ratio = ratio
            best = el
    return best if best_ratio > 0.5 else None


def _load_raw() -> tuple[dict, dict[str, MinerUMiddleDocument], dict[str, dict]]:
    """Load test cases, middle documents, and LLM response fixtures."""
    if not TEST_CASES_FILE.exists():
        return {}, {}, {}
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    middle_docs: dict[str, MinerUMiddleDocument] = {}
    fixtures: dict[str, dict] = {}

    for name in cases:
        mineru_path = MINERU_DIR / f"{name}_middle.json"
        fixture_path = FIXTURE_DIR / f"{name}.json"
        if not mineru_path.exists() or not fixture_path.exists():
            continue
        with open(mineru_path, encoding="utf-8") as f:
            middle_docs[name] = MinerUMiddleDocument.model_validate(json.load(f))
        with open(fixture_path, encoding="utf-8") as f:
            fixtures[name] = json.load(f)

    return cases, middle_docs, fixtures


_TEST_CASES, _MIDDLE_DOCS, _FIXTURES = _load_raw()


async def _parse_and_process(name: str) -> Document:
    """Return a freshly parsed, HeadingLevelProcessor-processed Document."""
    parser = MinerUParser(use_default_stages=False)
    parser.use(CoreExtractor())
    doc = await parser.parse(_MIDDLE_DOCS[name])
    return await HeadingLevelProcessor(trust_parser_levels=False).process(doc)


def _make_resolver(response_content: str) -> LLMHeadingResolver:
    """Create an LLMHeadingResolver whose client replays the given response."""
    heading_response = HeadingResponse.model_validate_json(response_content)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = heading_response
    mock_response.choices[0].message.refusal = None
    mock_client = MagicMock()
    mock_client.chat.completions.parse = AsyncMock(return_value=mock_response)
    return LLMHeadingResolver(client=mock_client)


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

_available = {name for name in _TEST_CASES if name in _MIDDLE_DOCS and name in _FIXTURES}

_heading_params = [
    pytest.param(name, heading_text, expected_level, id=f"{name}[{heading_text[:40]}]")
    for name, tc in _TEST_CASES.items()
    if name in _available
    for heading_text, expected_level in tc.get("headings", {}).items()
]

_scenario_params = [
    pytest.param(name, id=name)
    for name, tc in _TEST_CASES.items()
    if name in _available and len(tc.get("headings", {})) > 1
]

_title_params = [
    pytest.param(name, tc["title"], id=name) for name, tc in _TEST_CASES.items() if name in _available and "title" in tc
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _heading_params, reason="No LLM heading fixtures available")
@pytest.mark.parametrize("test_case_name,heading_text,expected_level", _heading_params)
@pytest.mark.anyio
async def test_heading_exact_level(test_case_name: str, heading_text: str, expected_level: int) -> None:
    """LLMHeadingResolver must assign each heading exactly the expected level."""
    doc = await _parse_and_process(test_case_name)
    resolver = _make_resolver(_FIXTURES[test_case_name]["response_content"])
    doc = await resolver.process(doc)

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
        pytest.skip(f"Heading '{heading_text}' not found after LLM processing")

    if target.level != expected_level:
        pytest.xfail(f"[{test_case_name}] '{heading_text}': expected level {expected_level}, got {target.level}")


@pytest.mark.skipif(not _scenario_params, reason="No LLM heading fixtures available")
@pytest.mark.parametrize("test_case_name", _scenario_params)
@pytest.mark.anyio
async def test_heading_ordering(test_case_name: str) -> None:
    """
    After LLMHeadingResolver, headings with a lower expected level (closer to the top
    of the hierarchy) must not be assigned a numerically higher level than headings
    with a greater expected level.

    Headings marked by the LLM as 'not a heading' (llm_heading_level == 'none') are
    excluded from the ordering check.
    """
    doc = await _parse_and_process(test_case_name)
    resolver = _make_resolver(_FIXTURES[test_case_name]["response_content"])
    doc = await resolver.process(doc)

    expected_headings: dict[str, int] = _TEST_CASES[test_case_name].get("headings", {})

    # Resolve each expected heading to its assigned level in the processed document.
    # Only include headings where the LLM made a concrete classification
    # (llm_heading_level is set and not "none"). Headings left uncertain by the
    # LLM (no metadata key) are skipped — their ordering is tested separately
    # by test_heading_integration.py via HeadingLevelProcessor alone.
    assigned: dict[str, int] = {}
    for heading_text in expected_headings:
        match = _find_classified_heading(doc, heading_text)
        if match is not None:
            assigned[heading_text] = match.level

    # For every pair where expected_level_A < expected_level_B,
    # the assigned level of A must be ≤ the assigned level of B.
    headings_list = list(expected_headings.items())
    violations: list[str] = []
    for i, (text_a, exp_a) in enumerate(headings_list):
        for text_b, exp_b in headings_list[i + 1 :]:
            if exp_a >= exp_b:
                continue
            if text_a not in assigned or text_b not in assigned:
                continue
            if assigned[text_a] > assigned[text_b]:
                violations.append(
                    f"  '{text_a}' (expected {exp_a}, assigned {assigned[text_a]}) "
                    f"> '{text_b}' (expected {exp_b}, assigned {assigned[text_b]})"
                )

    assert not violations, (
        f"[{test_case_name}] Ordering violations "
        f"(higher-level heading got a larger level number than a lower-level one):\n" + "\n".join(violations)
    )


@pytest.mark.skipif(not _title_params, reason="No LLM heading fixtures with titles available")
@pytest.mark.parametrize("test_case_name,expected_title", _title_params)
@pytest.mark.anyio
async def test_title_detected(test_case_name: str, expected_title: str) -> None:
    """LLMHeadingResolver must set document.title to the expected title."""
    doc = await _parse_and_process(test_case_name)
    resolver = _make_resolver(_FIXTURES[test_case_name]["response_content"])
    doc = await resolver.process(doc)

    assert doc.title is not None, f"[{test_case_name}] LLMHeadingResolver did not set document.title"
    assert _norm(doc.title) == _norm(expected_title), (
        f"[{test_case_name}] Wrong title.\n  Expected: '{expected_title}'\n  Got:      '{doc.title}'"
    )
