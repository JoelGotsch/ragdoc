"""
Integration tests for FootnoteProcessor.

The first section contains MinerU-based fixture tests (require pylatexenc).
The second section contains general integration tests (no extra deps).
"""

import copy
import json
import re
from pathlib import Path

import pytest

from ragdoc.document import Document, Footnote
from ragdoc.processing.footnote import (
    FootnoteProcessor,
    SimpleFootnoteResolver,
    build_footnote_pattern,
    find_footnote_candidates,
    score_footnote_candidates,
)

# MinerU-specific imports — guarded so the module still loads without pylatexenc.
_HAS_MINERU = True
try:
    from ragdoc.parsing.mineru.base import MinerUMiddleDocument
    from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor
    from ragdoc.utils.helpers import normalize_text
    from tests.latex_text import latex_to_text
except ImportError:
    _HAS_MINERU = False

_requires_mineru = pytest.mark.skipif(not _HAS_MINERU, reason="pdf_mineru extra not installed")

TEST_CASES_FILE = Path(__file__).parent.parent / "data" / "test_cases.json"
FILES_DIR = Path(__file__).parent.parent / "parsing" / "data" / "mineru"


# =============================================================================
# MinerU Helpers (only used when _HAS_MINERU is True)
# =============================================================================


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(latex_to_text(text))).strip()


def _contains(haystack: str, needle: str) -> bool:
    hn, nn = _norm(haystack), _norm(needle)
    return nn in hn or hn in nn


def _load_raw() -> tuple[dict, dict]:
    if not _HAS_MINERU or not TEST_CASES_FILE.exists():
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

# Parse all documents once at module load (sync context — no running event loop).
# Each test deep-copies from here so processor mutations don't bleed across tests.
# Module-level asyncio.run() — acceptable: feeds @pytest.mark.parametrize at import time.
import asyncio as _asyncio

_PARSED_DOCS: dict[str, Document] = {}
if _HAS_MINERU:
    for _name, _middle in _MIDDLE_DOCS.items():
        _p = MinerUExtractor(use_default_stages=False)
        _p.use(CoreExtractor())
        _PARSED_DOCS[_name] = _asyncio.run(_p.parse(_middle))


def _fresh_doc(name: str) -> Document:
    """Return a deep-copied Document so processor mutations don't leak between tests."""
    return copy.deepcopy(_PARSED_DOCS[name])


def _make_processor() -> FootnoteProcessor:
    return FootnoteProcessor(resolver=SimpleFootnoteResolver())


# =============================================================================
# Parametrize helpers
# =============================================================================

# (test_case_name, fn_index, fn_text, insertion_point | None)
_footnote_params = [
    pytest.param(
        name,
        i,
        fn["text"],
        fn.get("insertion_point"),
        id=f"{name}[fn{i}]",
    )
    for name, tc in _TEST_CASES.items()
    if name in _MIDDLE_DOCS
    for i, fn in enumerate(tc.get("footnotes", []))
]


# =============================================================================
# Footnote InlineRef tests
# =============================================================================


@_requires_mineru
@pytest.mark.anyio
@pytest.mark.skipif(not _footnote_params, reason="No footnotes defined in test cases")
@pytest.mark.parametrize("test_case_name,fn_index,fn_text,insertion_point", _footnote_params)
async def test_footnote_inline_ref_added(
    test_case_name: str,
    fn_index: int,
    fn_text: str,
    insertion_point: str | None,
) -> None:
    """
    After processing, every expected footnote must be referenced by at least
    one element via an InlineRef(rel_type='footnote').
    """
    doc = _fresh_doc(test_case_name)

    fn_norm = _norm(fn_text)
    footnote: Footnote | None = next(
        (f for f in doc.footnotes if fn_norm in _norm(f.text) or _norm(f.text) in fn_norm),
        None,
    )
    if footnote is None:
        pytest.skip(
            f"[{test_case_name}] fn{fn_index} not found in parsed document — covered by test_mineru_base_parser.py"
        )

    print(f"\n  fn{fn_index}  text: {fn_text[:80]}")
    if insertion_point:
        print(f"         insertion_point: {insertion_point[:80]}")
    print(f"         footnote: number={footnote.number}  page={footnote.page}")

    # Diagnose candidates before running the processor so we can explain failures.
    candidates = find_footnote_candidates(
        document=doc,
        footnote=footnote,
        context_chars=80,
        same_page_only=True,
    )
    print(f"         candidates (same_page_only=True): {len(candidates)}")
    if not candidates:
        # Try without the page restriction to see if that's the cause
        candidates_all = find_footnote_candidates(
            document=doc,
            footnote=footnote,
            context_chars=80,
            same_page_only=False,
        )
        print(f"         candidates (same_page_only=False): {len(candidates_all)}")
        if candidates_all:
            print("         [INFO] candidates exist on other pages — page restriction is filtering them out")
            for c in candidates_all[:5]:
                el = next((e for e in doc.elements if e.id == c.element_id), None)
                snippet = c.full_context.replace("\n", " ")[:70]
                print(f"                p{c.page}  type={type(el).__name__ if el else '?'}  ...{snippet}...")
        else:
            # No candidates at all — show elements on the same page for context
            pat = build_footnote_pattern(footnote.number)
            same_page = [el for el in doc.elements if el.page == footnote.page and not isinstance(el, Footnote)]
            matching = [el for el in same_page if pat.search(el.text)]
            print(
                f"         [INFO] pattern {pat.pattern!r} matched {len(matching)}/{len(same_page)} same-page elements"
            )
            for el in matching[:3]:
                print(f"                idx={doc.elements.index(el)}  type={type(el).__name__}  text: {el.text[:100]}")

    doc = await _make_processor().process(doc)

    resolved_elements = [
        el
        for el in doc.elements
        if any(ref.rel_type == "footnote" and ref.target_id == footnote.id for ref in el.inline_refs)
    ]

    if not resolved_elements:
        el_by_id = {el.id: el for el in doc.elements}
        if not candidates:
            print("         [FAIL] no candidates found — resolver had nothing to pick from")
        else:
            # Show scored candidates (without min_element_idx, so scores are optimistic)
            scored = score_footnote_candidates(candidates)
            print(
                f"         [FAIL] {len(candidates)} candidate(s) — resolver did not resolve (or apply_ref_patches failed)"
            )
            print(
                "         Note: scores shown WITHOUT sequential-ordering penalty (processor applies -10 for out-of-order)"
            )
            print(f"         {'idx':>3}  {'score':>5}  {'page':>4}  {'has_ref_in_html':<17}  context")
            ref_marker = f'id="{footnote.id}"'
            for i, (c, score) in enumerate(scored):
                el = el_by_id.get(c.element_id)
                has_ref = ref_marker in (el.html if el else "")
                snippet = c.full_context.replace("\n", " ")[:60]
                print(f"         [{i:>2}]  {score:>5}  p{c.page:<3}  {has_ref!s:<17}  ...{snippet}...")
            # Check if ref appears in ANY element
            any_el_with_ref = next((el for el in doc.elements if ref_marker in el.html), None)
            if any_el_with_ref:
                print(
                    f"         [INFO] ref IS present in element idx={doc.elements.index(any_el_with_ref)} HTML — inline_refs parse may have failed"
                )
                print(f"         [INFO] element.html snippet: {any_el_with_ref.html[:200]}")
            else:
                print(
                    "         [INFO] ref NOT present in any element HTML — apply_ref_patches failed or resolver returned wrong pick"
                )

    assert resolved_elements, (
        f"[{test_case_name}] fn{fn_index} ('{fn_text[:60]}') was not resolved — "
        f"no element has InlineRef(footnote, {footnote.id}) — see stdout for details"
    )


@_requires_mineru
@pytest.mark.anyio
@pytest.mark.skipif(not _footnote_params, reason="No footnotes defined in test cases")
@pytest.mark.parametrize("test_case_name,fn_index,fn_text,insertion_point", _footnote_params)
async def test_footnote_resolved_near_insertion_point(
    test_case_name: str,
    fn_index: int,
    fn_text: str,
    insertion_point: str | None,
) -> None:
    """
    When test_cases.json provides an insertion_point, the processor must resolve
    the footnote into the element whose text contains that insertion_point.

    We identify the expected element *before* processing (using the raw element
    text), then run the full FootnoteProcessor pipeline and verify the result.
    """
    if not insertion_point or len(insertion_point.strip()) < 8:
        pytest.skip("No meaningful insertion_point defined for this footnote")

    doc = _fresh_doc(test_case_name)

    fn_norm = _norm(fn_text)
    footnote: Footnote | None = next(
        (f for f in doc.footnotes if fn_norm in _norm(f.text) or _norm(f.text) in fn_norm),
        None,
    )
    if footnote is None:
        pytest.skip(
            f"[{test_case_name}] fn{fn_index} not found in parsed document — covered by test_mineru_base_parser.py"
        )

    # Identify the expected element before processing mutates any HTML.
    expected_el = next(
        (el for el in doc.elements if _contains(el.text, insertion_point)),
        None,
    )
    if expected_el is None:
        pytest.skip(
            f"[{test_case_name}] fn{fn_index}: insertion_point '{insertion_point[:60]}' "
            "not found in any element — fixture may be wrong or text is pre-processed away"
        )

    expected_el_id = expected_el.id

    print(f"\n  fn{fn_index}  text: {fn_text[:80]}")
    print(f"         insertion_point: {insertion_point[:80]}")
    print(f"         footnote: number={footnote.number}  page={footnote.page}")
    print(
        f"         expected element: idx={doc.elements.index(expected_el)}"
        f"  page={expected_el.page}  type={type(expected_el).__name__}"
    )

    # Run the full processor pipeline
    doc = await _make_processor().process(doc)

    # After processing, the expected element must have an InlineRef to this footnote.
    resolved_el = next((el for el in doc.elements if el.id == expected_el_id), None)
    assert resolved_el is not None, (
        f"[{test_case_name}] fn{fn_index}: expected element {expected_el_id} disappeared after processing"
    )

    has_ref = any(ref.rel_type == "footnote" and ref.target_id == footnote.id for ref in resolved_el.inline_refs)

    if has_ref:
        print("         [OK] Footnote resolved into expected element")
        return

    # On failure, show which element (if any) got the ref instead
    actual_elements = [
        el
        for el in doc.elements
        if any(r.rel_type == "footnote" and r.target_id == footnote.id for r in el.inline_refs)
    ]
    if actual_elements:
        for el in actual_elements:
            print(
                f"         [FAIL] ref landed in element id={el.id} "
                f"idx={doc.elements.index(el)} type={type(el).__name__} "
                f"text: {el.text[:100]}"
            )
    else:
        print("         [FAIL] footnote was not resolved into any element")

    print(f"         expected element text: {resolved_el.text[:150]}")

    pytest.fail(
        f"[{test_case_name}] fn{fn_index}: footnote was not resolved into the element "
        f"containing insertion_point '{insertion_point[:60]}' — see stdout for details"
    )


# =============================================================================
# Tests moved from tests/processing/test_footnote.py
# =============================================================================

from unittest.mock import AsyncMock, MagicMock

from ragdoc.document import Paragraph

# --- TestFootnoteProcessingIntegration ---


@pytest.mark.anyio
async def test_integration_full_pipeline_with_simple_resolver():
    """Test complete footnote processing pipeline with simple resolver."""
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html="<p>Research shows important findings 1 for the field.</p>",
                page=1,
            ),
            Paragraph(
                id="para-2",
                html="<p>Method 2 was used in the study.</p>",
                page=1,
            ),
            Footnote(
                id="fn-1",
                number=1,
                innerhtml="Smith, J. (2023). Important Research Paper.",
                page=1,
            ),
            Footnote(
                id="fn-2",
                number=2,
                innerhtml="Standard laboratory method.",
                page=1,
            ),
        ]
    )

    processor = FootnoteProcessor(
        resolver=SimpleFootnoteResolver(),
        update_html=True,
    )

    result = await processor.process(document)

    # Both footnotes should be resolved
    para1 = next(e for e in result.elements if e.id == "para-1")
    para2 = next(e for e in result.elements if e.id == "para-2")

    assert any(r.target_id == "fn-1" for r in para1.inline_refs)
    assert any(r.target_id == "fn-2" for r in para2.inline_refs)


@pytest.mark.anyio
async def test_integration_document_get_element_after_processing():
    """Test that document.get_element works with processed elements."""
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html="<p>See reference 1 for details.</p>",
                page=1,
            ),
            Footnote(
                id="fn-1",
                number=1,
                innerhtml="Reference details here.",
                page=1,
            ),
        ]
    )

    processor = FootnoteProcessor(update_html=True)
    result = await processor.process(document)

    # Should be able to resolve the footnote via inline_refs derived from HTML
    para = result.get_element("para-1")
    assert para is not None

    assert para.inline_refs, "Expected at least one <ref> tag in HTML after processing"
    ref = para.inline_refs[0]
    footnote = result.get_element(ref.target_id)
    assert footnote is not None
    assert isinstance(footnote, Footnote)


# --- TestSequentialFootnotesIntegration ---


@pytest.mark.anyio
async def test_sequential_footnotes_two_adjacent_refs_in_same_paragraph():
    """FootnoteProcessor correctly inserts two ref tags adjacent in one element.

    Derived from tests/data/test_cases.json, document
    "tesla-q4-2024-update", insertion_point
    "of the Joint Operating Procedures (the Service Agreement1)."

    The paragraph contains the substring "Agreement1).2": footnote 1 is
    glued to "Agreement" and footnote 2 follows immediately after ")."
    Without the two-phase design the second replacement would target a
    stale offset and produce garbled HTML.
    """
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html=(
                    "<p>executed under the Joint Operating Procedures "
                    "(the Service Agreement1).2 entered into force.</p>"
                ),
                page=1,
            ),
            Footnote(
                id="fn-1",
                number=1,
                innerhtml="Memo MR-214.",
                page=1,
            ),
            Footnote(
                id="fn-2",
                number=2,
                innerhtml=(
                    "The initial report to the Audit Committee on this "
                    "specific matter was provided by the Director of Operations."
                ),
                page=1,
            ),
        ]
    )

    processor = FootnoteProcessor(
        resolver=SimpleFootnoteResolver(),
        update_html=True,
        same_page_only=True,
    )
    result = await processor.process(document)

    para = result.get_element("para-1")
    assert para is not None

    # Both footnote refs must appear in the final HTML
    assert '<ref id="fn-1" rel="footnote"/>' in para.html
    assert '<ref id="fn-2" rel="footnote"/>' in para.html

    # InlineRefs must be present too
    ref_target_ids = {r.target_id for r in para.inline_refs}
    assert "fn-1" in ref_target_ids
    assert "fn-2" in ref_target_ids


# --- TestFootnoteIsolationAndIdempotency ---


@pytest.mark.anyio
async def test_isolation_unpicked_candidate_html_unchanged():
    """When the resolver picks para-1, para-2 (also containing the number) is not patched."""
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html="<p>Smith study 1 confirms this finding.</p>",
                page=1,
            ),
            Paragraph(
                id="para-2",
                html="<p>See also result 1 elsewhere in the document.</p>",
                page=1,
            ),
            Footnote(id="fn-1", number=1, innerhtml="Smith study original.", page=1),
        ]
    )
    original_para2 = document.get_element("para-2").html

    # Resolver always picks the first candidate (para-1 because it appears first)
    processor = FootnoteProcessor(
        resolver=SimpleFootnoteResolver(),
        update_html=True,
        same_page_only=True,
    )
    await processor.process(document)

    assert document.get_element("para-2").html == original_para2


@pytest.mark.anyio
async def test_isolation_resolver_returns_none_leaves_html_unchanged():
    """When the resolver returns None, no HTML is modified."""
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html="<p>See 1 for details.</p>",
                page=1,
            ),
            Footnote(id="fn-1", number=1, innerhtml="Smith.", page=1),
        ]
    )
    original_html = document.get_element("para-1").html

    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=None)
    await FootnoteProcessor(resolver=resolver, update_html=True).process(document)

    assert document.get_element("para-1").html == original_html


@pytest.mark.anyio
async def test_idempotency_second_pass_does_not_corrupt_html():
    """Processing a document twice must not nest ref tags inside already-inserted refs."""
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html="<p>See reference 1 for details.</p>",
                page=1,
            ),
            Footnote(id="fn-1", number=1, innerhtml="Smith 2023.", page=1),
        ]
    )
    processor = FootnoteProcessor(resolver=SimpleFootnoteResolver(), update_html=True)
    await processor.process(document)
    html_after_first = document.get_element("para-1").html

    await processor.process(document)

    assert document.get_element("para-1").html == html_after_first


@pytest.mark.anyio
async def test_isolation_multiple_footnotes_each_patched_in_correct_element():
    """Each footnote only patches its resolved element, not the others."""
    document = Document(
        elements=[
            Paragraph(
                id="para-1",
                html="<p>Method 1 was applied here.</p>",
                page=1,
            ),
            Paragraph(
                id="para-2",
                html="<p>Result 2 was observed.</p>",
                page=1,
            ),
            Footnote(id="fn-1", number=1, innerhtml="Standard method.", page=1),
            Footnote(id="fn-2", number=2, innerhtml="Observed result.", page=1),
        ]
    )
    processor = FootnoteProcessor(resolver=SimpleFootnoteResolver(), update_html=True)
    await processor.process(document)

    para1_html = document.get_element("para-1").html
    para2_html = document.get_element("para-2").html

    # Each element has only its own ref tag
    assert '<ref id="fn-1" rel="footnote"/>' in para1_html
    assert "fn-2" not in para1_html
    assert '<ref id="fn-2" rel="footnote"/>' in para2_html
    assert "fn-1" not in para2_html
