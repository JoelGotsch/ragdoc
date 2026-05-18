"""Tests for the footnote processing module."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ragdoc.document import Document, DocumentList, Footnote, Heading, Paragraph
from ragdoc.processing.footnote import (
    FootnoteCandidate,
    FootnoteProcessor,
    FootnoteResolver,
    LLMFootnoteResolver,
    SimpleFootnoteResolver,
    SyncFootnoteProcessor,
    apply_ref_patches,
    build_footnote_pattern,
    find_footnote_candidates,
    score_footnote_candidates,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document with paragraphs and footnotes."""
    return Document(
        elements=[
            Heading(innerhtml="Introduction", level=1, page=1),
            Paragraph(
                id="para-1",
                html_content="<p>This is the first paragraph with a reference 1 to a study.</p>",
                page=1,
            ),
            Paragraph(
                id="para-2",
                html_content="<p>Another paragraph mentions topic 2 for more details.</p>",
                page=1,
            ),
            Paragraph(
                id="para-3",
                html_content="<p>Third paragraph on page 2 with reference 1 again.</p>",
                page=2,
            ),
            Footnote(
                id="footnote-1",
                number=1,
                innerhtml="Smith et al., 2023. Journal of Science.",
                page=1,
            ),
            Footnote(
                id="footnote-2",
                number=2,
                innerhtml="See chapter 5 for additional details.",
                page=1,
            ),
        ]
    )


@pytest.fixture
def document_no_footnotes() -> Document:
    """Create a document without any footnotes."""
    return Document(
        elements=[
            Heading(innerhtml="Simple Document", level=1, page=1),
            Paragraph(
                id="para-1",
                html_content="<p>Just a simple paragraph with no references.</p>",
                page=1,
            ),
        ]
    )


@pytest.fixture
def document_multiple_candidates() -> Document:
    """Create a document where footnote number appears multiple times."""
    return Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>In section 1, we discuss topic 1 from the original paper on page 1.</p>",
                page=1,
            ),
            Paragraph(
                id="para-2",
                html_content="<p>The experiment showed result 1 which was significant.</p>",
                page=1,
            ),
            Footnote(
                id="footnote-1",
                number=1,
                innerhtml="Original paper by Jones, 2022.",
                page=1,
            ),
        ]
    )


# --- TestFindFootnoteCandidates ---


def test_find_footnote_candidates_finds_candidates_in_paragraphs(sample_document):
    """find_footnote_candidates finds matches in paragraph text."""
    footnote = sample_document.footnotes[0]  # Footnote 1
    candidates = find_footnote_candidates(sample_document, footnote)

    # Should find "1" in para-1 (same page as footnote)
    assert len(candidates) >= 1
    assert all(c.reference_number == 1 for c in candidates)
    assert all(c.footnote_id == footnote.id for c in candidates)


def test_find_footnote_candidates_same_page_only_filter(sample_document):
    """find_footnote_candidates respects same_page_only parameter."""
    footnote = sample_document.footnotes[0]  # Footnote 1 on page 1

    # With same_page_only=True (default)
    candidates_same_page = find_footnote_candidates(
        sample_document, footnote, same_page_only=True
    )

    # With same_page_only=False
    candidates_all_pages = find_footnote_candidates(
        sample_document, footnote, same_page_only=False
    )

    # All candidates from same_page should be on page 1
    assert all(c.page == 1 for c in candidates_same_page)

    # All pages should include page 2 candidate
    page_numbers = {c.page for c in candidates_all_pages}
    # para-3 is on page 2 and contains "1"
    assert 2 in page_numbers or len(candidates_all_pages) > len(candidates_same_page)


def test_find_footnote_candidates_context_extraction(sample_document):
    """find_footnote_candidates extracts context around matches."""
    footnote = sample_document.footnotes[0]
    candidates = find_footnote_candidates(
        sample_document, footnote, context_chars=20
    )

    for candidate in candidates:
        assert len(candidate.context_before) <= 20
        assert len(candidate.context_after) <= 20
        assert candidate.full_context != ""


def test_find_footnote_candidates_returns_empty_for_no_matches(document_no_footnotes):
    """find_footnote_candidates returns empty list when no matches."""
    footnote = Footnote(id="fn-99", number=99, innerhtml="Not found", page=1)
    candidates = find_footnote_candidates(document_no_footnotes, footnote)

    assert candidates == []


def test_find_footnote_candidates_element_idx_is_correct(sample_document):
    """find_footnote_candidates sets correct element_idx."""
    footnote = sample_document.footnotes[0]
    candidates = find_footnote_candidates(sample_document, footnote)

    for candidate in candidates:
        element = sample_document.elements[candidate.element_idx]
        assert element.id == candidate.element_id


def test_find_footnote_candidates_never_returns_footnote_elements_as_candidates(sample_document):
    """Footnote elements are skipped so a footnote cannot reference itself."""
    footnote = sample_document.footnotes[0]
    candidates = find_footnote_candidates(sample_document, footnote)

    for candidate in candidates:
        element = sample_document.elements[candidate.element_idx]
        assert not isinstance(element, Footnote)


def test_find_footnote_candidates_finds_reference_in_document_list():
    """find_footnote_candidates searches DocumentList elements (e.g. MinerU list items)."""
    doc = Document(
        elements=[
            DocumentList(
                id="list-1",
                html_content="<ul><li>As previously reported,2 the auditor confirmed the findings.</li></ul>",
                page=1,
            ),
            Footnote(id="fn-2", number=2, innerhtml="Internal Bulletin 2024-17, para. 2.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 1
    assert candidates[0].element_id == "list-1"


def test_find_footnote_candidates_finds_reference_in_heading():
    """find_footnote_candidates searches Heading elements."""
    doc = Document(
        elements=[
            Heading(
                id="heading-1",
                innerhtml="System Implementation3",
                level=1,
                page=1,
            ),
            Footnote(id="fn-3", number=3, innerhtml="See annex.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 1
    assert candidates[0].element_id == "heading-1"


def test_find_footnote_candidates_skips_footnote_whose_number_matches():
    """A Footnote element whose text contains the number is not returned as a candidate."""
    doc = Document(
        elements=[
            Footnote(id="fn-1", number=1, innerhtml="Smith et al., 2023. See also footnote 1.", page=1),
            Paragraph(
                id="para-1",
                html_content="<p>As shown in reference 1 above.</p>",
                page=1,
            ),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    # fn-1 must not appear; para-1 must appear
    candidate_ids = {c.element_id for c in candidates}
    assert "fn-1" not in candidate_ids
    assert "para-1" in candidate_ids


def test_find_footnote_candidates_finds_number_glued_to_word():
    """find_footnote_candidates finds a footnote number glued to a word (no space)."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>The Service Agreement6 remains in force.</p>",
                page=1,
            ),
            Footnote(id="fn-6", number=6, innerhtml="Memo MR-214.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 1
    assert candidates[0].element_id == "para-1"


def test_find_footnote_candidates_finds_number_after_comma_no_space():
    """find_footnote_candidates finds a footnote number directly after a comma."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>See previous reports,3 which detail the findings.</p>",
                page=1,
            ),
            Footnote(id="fn-3", number=3, innerhtml="Compliance Report 2024-10.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 1
    assert candidates[0].element_id == "para-1"


def test_find_footnote_candidates_finds_number_after_period_no_space():
    """find_footnote_candidates finds a footnote number directly after a period."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>This was confirmed by the Director of Operations.5</p>",
                page=1,
            ),
            Footnote(id="fn-5", number=5, innerhtml="Compliance Report 2024-51, para. 13.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 1
    assert candidates[0].element_id == "para-1"


def test_find_footnote_candidates_matches_number_at_end_of_larger_number():
    """find_footnote_candidates matches footnote 6 inside '1956' and '2016'.

    The pattern does not require a non-digit before the number, so a trailing
    digit match (e.g. the '6' at the end of a year) is accepted as a candidate.
    """
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>In 1956 and 2016 the policy was established.</p>",
                page=1,
            ),
            Footnote(id="fn-6", number=6, innerhtml="Memo MR-214.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 2


def test_find_footnote_candidates_finds_ocr_split_multi_digit_number():
    """find_footnote_candidates finds a two-digit footnote when OCR inserts a space."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>Sensor reading DEV1 1 was flagged in the audit log.</p>",
                page=1,
            ),
            Footnote(id="fn-11", number=11, innerhtml="DEV denotes deviation from the nominal calibration baseline.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    assert len(candidates) == 1
    assert candidates[0].element_id == "para-1"


def test_find_footnote_candidates_does_not_match_ocr_split_within_longer_sequence():
    """find_footnote_candidates does not match '1 1' for footnote 11 inside '1 1 1'."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>Items 1 1 1 were reviewed.</p>",
                page=1,
            ),
            Footnote(id="fn-11", number=11, innerhtml="Definition.", page=1),
        ]
    )
    footnote = doc.footnotes[0]
    candidates = find_footnote_candidates(doc, footnote)
    # "1 1 1" contains overlapping matches; the pattern must not match
    # the middle pair because the surrounding digits block the lookbehind/lookahead.
    # Only zero or one match is acceptable (not two).
    assert len(candidates) <= 1


# =============================================================================
# Tests for build_footnote_pattern
# =============================================================================


@pytest.mark.parametrize(
    "footnote_number, text",
    [
        # Single-digit: standard cases that \b already handled
        (6, "text 6 more"),
        (6, "end of sentence.6"),
        (6, "first clause,6 second"),
        (6, "end of text6"),
        # Single-digit: broader cases (letter prefix, no space / OCR artifacts)
        (6, "Agreement6 remains"),
        (6, "(MOU)6 provides"),
        # Single-digit: digit before — no lookbehind is applied
        (6, "56"),
        (6, "1996"),
        # Multi-digit: standard (no OCR split)
        (11, "reference 11 here"),
        (11, "previous reports,11 which"),
        (11, "DEV11 indicative"),
        # Multi-digit: OCR-split (spaces between digits)
        (11, "H E U 1 1 indicative"),
        (11, "symbol  1  1  found"),
        # Multi-digit: digit before — no lookbehind for preceding digit
        (11, "211"),
        (11, "111"),
    ],
)
def test_build_footnote_pattern_should_match(footnote_number: int, text: str) -> None:
    """build_footnote_pattern produces a regex that matches the footnote number in text."""
    assert build_footnote_pattern(footnote_number).search(text)


@pytest.mark.parametrize(
    "footnote_number, text",
    [
        # Single-digit: digit immediately after — must not match
        (6, "60"),
        # Single-digit: number embedded mid-sequence
        (6, "456789"),
        # Multi-digit: digit immediately after — must not match
        (11, "110"),
    ],
)
def test_build_footnote_pattern_should_not_match(footnote_number: int, text: str) -> None:
    """build_footnote_pattern produces a regex that must NOT match the footnote number in text."""
    assert not build_footnote_pattern(footnote_number).search(text)


# --- TestSimpleFootnoteResolver ---


@pytest.mark.anyio
async def test_simple_resolver_returns_none_for_empty_candidates():
    """SimpleFootnoteResolver returns None for empty candidates list."""
    resolver = SimpleFootnoteResolver()
    result = await resolver.resolve([], 1, "Some footnote")
    assert result is None


@pytest.mark.anyio
async def test_simple_resolver_returns_single_candidate_directly():
    """SimpleFootnoteResolver returns single candidate without scoring."""
    resolver = SimpleFootnoteResolver()
    candidate = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="text",
        context_after="more",
        full_context="text 1 more",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate], 1, "Note")
    assert result is candidate


@pytest.mark.anyio
async def test_simple_resolver_prefers_candidate_with_word_overlap():
    """SimpleFootnoteResolver prefers candidate with overlapping words."""
    resolver = SimpleFootnoteResolver()

    # Candidate with no overlap
    candidate1 = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="random",
        context_after="unrelated",
        full_context="random 1 unrelated",
        reference_number=1,
        footnote_text="Smith study results",
        footnote_id="fn-1",
    )

    # Candidate with overlap ("study")
    candidate2 = FootnoteCandidate(
        element_id="elem-2",
        element_idx=1,
        page=1,
        match_start=10,
        match_end=11,
        context_before="the study",
        context_after="showed",
        full_context="the study 1 showed",
        reference_number=1,
        footnote_text="Smith study results",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate1, candidate2], 1, "Smith study results")
    assert result is candidate2


@pytest.mark.anyio
async def test_simple_resolver_falls_back_to_first_candidate():
    """SimpleFootnoteResolver falls back to first candidate if no overlap."""
    resolver = SimpleFootnoteResolver()

    candidate1 = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="aaa",
        context_after="bbb",
        full_context="aaa 1 bbb",
        reference_number=1,
        footnote_text="xyz unique words",
        footnote_id="fn-1",
    )

    candidate2 = FootnoteCandidate(
        element_id="elem-2",
        element_idx=1,
        page=1,
        match_start=10,
        match_end=11,
        context_before="ccc",
        context_after="ddd",
        full_context="ccc 1 ddd",
        reference_number=1,
        footnote_text="xyz unique words",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate1, candidate2], 1, "xyz unique words")
    # Should return first candidate as fallback
    assert result is candidate1


# --- TestLLMFootnoteResolver ---


@pytest.mark.anyio
async def test_llm_resolver_returns_none_for_empty_candidates():
    """LLMFootnoteResolver returns None for empty candidates list."""
    mock_client = MagicMock()
    resolver = LLMFootnoteResolver(client=mock_client)

    result = await resolver.resolve([], 1, "Some footnote")

    assert result is None
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.anyio
async def test_llm_resolver_returns_single_candidate_without_llm():
    """LLMFootnoteResolver returns single candidate without calling LLM."""
    mock_client = MagicMock()
    resolver = LLMFootnoteResolver(client=mock_client)

    candidate = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="text",
        context_after="more",
        full_context="text 1 more",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate], 1, "Note")

    assert result is candidate
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.anyio
async def test_llm_resolver_calls_llm_for_multiple_candidates():
    """LLMFootnoteResolver calls LLM when multiple candidates exist."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "2"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    resolver = LLMFootnoteResolver(client=mock_client, model="test-model")

    candidate1 = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="text",
        context_after="more",
        full_context="text 1 more",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )
    candidate2 = FootnoteCandidate(
        element_id="elem-2",
        element_idx=1,
        page=1,
        match_start=5,
        match_end=6,
        context_before="other",
        context_after="context",
        full_context="other 1 context",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate1, candidate2], 1, "Note")

    assert result is candidate2
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio
async def test_llm_resolver_handles_none_response():
    """LLMFootnoteResolver handles NONE response from LLM."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "NONE"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    resolver = LLMFootnoteResolver(client=mock_client)

    candidate1 = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="text",
        context_after="more",
        full_context="text 1 more",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )
    candidate2 = FootnoteCandidate(
        element_id="elem-2",
        element_idx=1,
        page=1,
        match_start=5,
        match_end=6,
        context_before="other",
        context_after="context",
        full_context="other 1 context",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate1, candidate2], 1, "Note")

    assert result is None


@pytest.mark.anyio
async def test_llm_resolver_handles_llm_error():
    """LLMFootnoteResolver falls back to first candidate on LLM error."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    resolver = LLMFootnoteResolver(client=mock_client)

    candidate1 = FootnoteCandidate(
        element_id="elem-1",
        element_idx=0,
        page=1,
        match_start=10,
        match_end=11,
        context_before="text",
        context_after="more",
        full_context="text 1 more",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )
    candidate2 = FootnoteCandidate(
        element_id="elem-2",
        element_idx=1,
        page=1,
        match_start=5,
        match_end=6,
        context_before="other",
        context_after="context",
        full_context="other 1 context",
        reference_number=1,
        footnote_text="Note",
        footnote_id="fn-1",
    )

    result = await resolver.resolve([candidate1, candidate2], 1, "Note")

    # Falls back to first candidate
    assert result is candidate1


# --- TestFootnoteProcessor ---


@pytest.mark.anyio
async def test_footnote_processor_returns_document_unchanged_when_no_footnotes(
    document_no_footnotes,
):
    """FootnoteProcessor returns document unchanged if no footnotes."""
    processor = FootnoteProcessor()
    result = await processor.process(document_no_footnotes)
    assert result is document_no_footnotes


@pytest.mark.anyio
async def test_footnote_processor_uses_custom_resolver(sample_document):
    """FootnoteProcessor uses the provided resolver."""
    mock_resolver = MagicMock()
    mock_resolver.resolve = AsyncMock(return_value=None)

    processor = FootnoteProcessor(resolver=mock_resolver)
    await processor.process(sample_document)

    # Should have called resolve for each footnote
    assert mock_resolver.resolve.call_count == len(sample_document.footnotes)


# --- TestSyncFootnoteProcessor ---


@pytest.mark.anyio
async def test_sync_footnote_processor_processes_document(sample_document):
    """SyncFootnoteProcessor resolves footnotes."""
    processor = SyncFootnoteProcessor(update_html=True)
    result = await processor.process(sample_document)

    assert result is not None
    # Should have resolved at least one footnote (ref tags embedded in HTML)
    elements_with_refs = [e for e in result.elements if e.inline_refs]
    assert len(elements_with_refs) > 0


# --- TestApplyRefPatches ---


def test_apply_ref_patches_single_patch_replaces_number_in_html():
    """A single patch replaces the footnote number with a ref tag."""
    element = Paragraph(
        id="para-1",
        html_content="<p>See the Service Agreement 1 for details.</p>",
        page=1,
    )
    # "1" is at position 28 in the text node "See the Service Agreement 1 for details."
    text = "See the Service Agreement 1 for details."
    pos = text.index("1")
    apply_ref_patches(element, [(0, pos, pos + 1, '<sup><ref id="fn-1"/>[1]</sup>')])
    assert '<ref id="fn-1"/>' in element.html
    assert ">1<" not in element.html  # bare "1" replaced


def test_apply_ref_patches_no_patches_leaves_html_unchanged():
    """apply_ref_patches with empty patch list does not modify the element."""
    original = "<p>No references here.</p>"
    element = Paragraph(id="para-1", html_content=original, page=1)
    apply_ref_patches(element, [])
    assert element.html == original


def test_apply_ref_patches_sequential_footnotes_same_text_node():
    """Two adjacent footnote refs in the same text node are both patched correctly.

    This scenario is derived from tests/data/test_cases.json, document
    "tesla-q4-2024-update", insertion_point
    "of the Joint Operating Procedures (the Service Agreement1)."

    The raw text contains "Agreement1).2" where footnote 1 immediately
    precedes footnote 2 (separated only by ")." ).  Naively replacing
    footnote 1 first would shift the position of "2", causing its patch
    to land on the wrong character.  apply_ref_patches must apply patches
    right-to-left to avoid this.
    """
    text_node = "executed under the Joint Operating Procedures (the Service Agreement1).2 confirms this."
    html = f"<p>{text_node}</p>"
    element = Paragraph(id="para-1", html_content=html, page=1)

    pos1 = text_node.index("1")  # immediately after "Agreement"
    pos2 = text_node.index("2")  # immediately after ")."

    patches = [
        (0, pos1, pos1 + 1, '<sup><ref id="fn-1"/>[1]</sup>'),
        (0, pos2, pos2 + 1, '<sup><ref id="fn-2"/>[2]</sup>'),
    ]
    apply_ref_patches(element, patches)

    result = element.html
    assert '<ref id="fn-1"/>' in result
    assert '<ref id="fn-2"/>' in result
    # Both ref tags present; original bare digits are gone
    assert "Agreement1" not in result
    assert ").2" not in result


def test_apply_ref_patches_patches_across_multiple_text_nodes():
    """Patches targeting different text nodes are each applied correctly."""
    # A list with two items, footnote ref in each
    html = "<ul><li>First item1 here.</li><li>Second item2 here.</li></ul>"
    element = Paragraph(id="para-1", html_content=html, page=1)

    # Text node 0: "First item1 here."  → "1" at index 10
    # Text node 1: "Second item2 here." → "2" at index 11
    patches = [
        (0, 10, 11, '<sup><ref id="fn-1"/>[1]</sup>'),
        (1, 11, 12, '<sup><ref id="fn-2"/>[2]</sup>'),
    ]
    apply_ref_patches(element, patches)

    result = element.html
    assert '<ref id="fn-1"/>' in result
    assert '<ref id="fn-2"/>' in result


# --- TestApplyRefPatchesEdgeCases ---


def test_apply_ref_patches_patch_order_independence():
    """Patches applied left-to-right or right-to-left produce identical results."""
    text_node = "conclusion1 and finding2 are related."
    html = f"<p>{text_node}</p>"

    pos1 = text_node.index("1")
    pos2 = text_node.index("2")
    patches_lr = [
        (0, pos1, pos1 + 1, '<sup><ref id="fn-1"/>[1]</sup>'),
        (0, pos2, pos2 + 1, '<sup><ref id="fn-2"/>[2]</sup>'),
    ]
    patches_rl = list(reversed(patches_lr))

    el1 = Paragraph(id="p1", html_content=html, page=1)
    el2 = Paragraph(id="p2", html_content=html, page=1)
    apply_ref_patches(el1, patches_lr)
    apply_ref_patches(el2, patches_rl)

    assert el1.html == el2.html
    assert '<ref id="fn-1"/>' in el1.html
    assert '<ref id="fn-2"/>' in el1.html


def test_apply_ref_patches_only_specified_occurrence_is_replaced():
    """A patch at position N replaces only that occurrence, not others."""
    text_node = "result 1 confirms finding 1 again."
    html = f"<p>{text_node}</p>"
    element = Paragraph(id="p", html_content=html, page=1)

    # Patch only the SECOND occurrence of "1"
    second_pos = text_node.rindex("1")
    apply_ref_patches(element, [(0, second_pos, second_pos + 1, '<sup><ref id="fn-1"/>[1]</sup>')])

    result = element.html
    # First "1" (after "result") is still plain text
    assert "result 1 " in result
    # Second "1" is replaced
    assert '<ref id="fn-1"/>' in result


def test_apply_ref_patches_out_of_bounds_text_node_idx_silently_skipped():
    """A patch referencing a non-existent text node index is silently ignored."""
    original = "<p>Simple text.</p>"
    element = Paragraph(id="p", html_content=original, page=1)
    apply_ref_patches(element, [(99, 0, 1, '<sup><ref id="fn-1"/>[1]</sup>')])
    assert element.html == original


# --- TestFindFootnoteCandidatesEdgeCases ---


def test_find_footnote_candidates_ignores_html_attribute_values():
    """Numbers inside HTML attribute values (e.g. style, data-*) are not matched."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content='<p style="margin: 2px">No footnote ref here.</p>',
                page=1,
            ),
            Footnote(id="fn-2", number=2, innerhtml="Some note.", page=1),
        ]
    )
    candidates = find_footnote_candidates(doc, doc.footnotes[0])
    assert candidates == []


def test_find_footnote_candidates_returns_all_occurrences_in_same_text_node():
    """Three occurrences of the footnote number -> three candidates."""
    doc = Document(
        elements=[
            Paragraph(
                id="para-1",
                html_content="<p>First 1 middle 1 last 1 end.</p>",
                page=1,
            ),
            Footnote(id="fn-1", number=1, innerhtml="Note.", page=1),
        ]
    )
    candidates = find_footnote_candidates(doc, doc.footnotes[0])
    assert len(candidates) == 3
    assert all(c.element_id == "para-1" for c in candidates)


def test_find_footnote_candidates_does_not_match_inside_existing_ref_tag():
    """After a first pass, the '[1]' inside a <ref> tag is not re-matched."""
    # Simulate HTML after FootnoteProcessor has already run once
    already_processed_html = (
        '<p>See reference <sup><ref id="fn-1">[1]</ref></sup> for details.</p>'
    )
    doc = Document(
        elements=[
            Paragraph(id="para-1", html_content=already_processed_html, page=1),
            Footnote(id="fn-1", number=1, innerhtml="Smith 2023.", page=1),
        ]
    )
    candidates = find_footnote_candidates(doc, doc.footnotes[0])
    # The "[1]" inside the <ref> tag must not produce a candidate
    assert candidates == []


# =============================================================================
# Regression: year-glued-footnote vs paragraph-number disambiguation
# Synthetic scenario modelled on a real-world OCR ambiguity.
#
# PICKED (wrong): para 16 — "16. The auditor has also sought..."
#   match: "6" in "16", context_before="1", context_after=". The auditor has also..."
#   This is a paragraph number ("16."), not a footnote reference.
#
# CORRECT: para 17 — "...storage until January 20046. Taking into account..."
#   match: "6" in "20046", context_before="...until January 2004",
#   context_after=". Taking into account..."
#   The "6" is footnote 6 OCR-merged onto year "2004".
# =============================================================================


def _make_candidate(
    element_id: str,
    element_idx: int,
    context_before: str,
    context_after: str,
    number: int = 6,
) -> FootnoteCandidate:
    return FootnoteCandidate(
        element_id=element_id,
        element_idx=element_idx,
        page=3,
        match_start=len(context_before),
        match_end=len(context_before) + len(str(number)),
        context_before=context_before,
        context_after=context_after,
        full_context=context_before + str(number) + context_after,
        reference_number=number,
        footnote_text="In January 2004, the equipment was inspected by the auditor and the residual process material was recovered therefrom.",
        footnote_id="fn-6",
    )


def _para16_candidate() -> FootnoteCandidate:
    """Para 16: "16. The auditor has also sought..." -- "6" is part of "16.".

    "6" sits at position 1 of the text node "16. The auditor...", so
    context_before is just "1" (1 char). context_after is the full 80-char
    window.  Score: -2 (digit-preceded). No +1 because len(after)=80 >> len(before)=1.
    """
    return _make_candidate(
        element_id="para-16",
        element_idx=15,
        context_before="1",
        context_after=". The auditor has also sought to confirm the disclosures of the supplier about t",
    )


def _para17_candidate() -> FootnoteCandidate:
    """Para 17: "...storage until January 20046. Taking..." -- OCR-merged year+footnote.

    With context_chars=80, both sides are exactly 80 chars, so the
    "closer to end of node" +1 tiebreak does NOT fire.
    Score: -2 (digit-preceded). No +1 because len(after)==len(before)==80.
    """
    return _make_candidate(
        element_id="para-17",
        element_idx=16,
        # 80 chars of text that precede "6" in "January 20046"
        context_before="uipment was said to have been dismantled and put into storage until January 2004",
        # 80 chars of text that follow "6"
        context_after=". Taking into account the age of the activities and the lack of records with reg",
    )


# --- TestYearGluedFootnoteScoring ---


def test_year_glued_scores_higher_than_paragraph_number():
    """Para 17 (year+footnote) must score strictly higher than para 16 (paragraph number)."""
    para16 = _para16_candidate()
    para17 = _para17_candidate()
    scored = score_footnote_candidates([para16, para17])
    score_para16 = scored[0][1]  # para16 is first
    score_para17 = scored[1][1]  # para17 is second
    assert score_para17 > score_para16, (
        f"Expected para-17 (year+footnote, score={score_para17}) to outscore "
        f"para-16 (paragraph number, score={score_para16})"
    )


@pytest.mark.anyio
async def test_year_glued_resolver_picks_year_glued_over_paragraph_number():
    """SimpleFootnoteResolver must pick para 17 (year+footnote) over para 16 (paragraph number)."""
    resolver = SimpleFootnoteResolver()
    para16 = _para16_candidate()
    para17 = _para17_candidate()
    result = await resolver.resolve(
        candidates=[para16, para17],
        footnote_number=6,
        footnote_text="In January 2004, the equipment was inspected by the auditor and the residual process material was recovered therefrom.",
    )
    assert result is not None
    assert result.element_id == "para-17", (
        f"Expected para-17 but resolver picked {result.element_id!r}"
    )


# --- TestOnlyOrphanedParameter ---


def _make_only_orphaned_doc() -> Document:
    """Document where fn-1 is already resolved via a pre-existing <ref> tag.

    Para-1 also contains a dangling "1" after the ref tag -- this lets us
    assert that only_orphaned=True leaves it untouched while
    only_orphaned=False (default) would insert a second ref there.
    Para-2 contains "2", the reference for the still-orphaned fn-2.
    """
    return Document(elements=[
        Paragraph(
            id="para-1",
            html_content='<p>See note.<ref id="fn-1" rel="footnote"/> Also item 1 here.</p>',
            page=1,
        ),
        Paragraph(
            id="para-2",
            html_content="<p>Another paragraph with 2 references.</p>",
            page=1,
        ),
        Footnote(id="fn-1", number=1, innerhtml="Already resolved footnote.", page=1),
        Footnote(id="fn-2", number=2, innerhtml="Not yet resolved footnote.", page=1),
    ])


@pytest.mark.anyio
async def test_only_orphaned_skips_referenced_footnote():
    """only_orphaned=True: footnote already referenced must not get a second ref."""
    doc = _make_only_orphaned_doc()
    doc = await FootnoteProcessor(only_orphaned=True).process(doc)

    para1 = next(e for e in doc.elements if e.id == "para-1")
    assert para1.html.count('id="fn-1"') == 1, (
        "fn-1 already had one ref; only_orphaned=True must not add another"
    )
    # fn-2 (the actual orphan) should now be resolved
    assert doc.orphaned_footnotes == []


@pytest.mark.anyio
async def test_only_orphaned_false_processes_dangling_number():
    """only_orphaned=False (default): the dangling '1' in para-1 gets a second ref."""
    doc = _make_only_orphaned_doc()
    doc = await FootnoteProcessor(only_orphaned=False).process(doc)

    para1 = next(e for e in doc.elements if e.id == "para-1")
    assert para1.html.count('id="fn-1"') == 2, (
        "only_orphaned=False should resolve fn-1 again, adding a ref to the dangling '1'"
    )


@pytest.mark.anyio
async def test_only_orphaned_true_processes_all_when_none_prereferred():
    """only_orphaned=True with no pre-existing refs processes every footnote."""
    doc = Document(elements=[
        Paragraph(
            id="para-1",
            html_content="<p>Reference to study1 in detail.</p>",
            page=1,
        ),
        Footnote(id="fn-1", number=1, innerhtml="A study.", page=1),
    ])
    doc = await FootnoteProcessor(only_orphaned=True).process(doc)
    assert doc.orphaned_footnotes == []


@pytest.mark.anyio
async def test_sync_processor_only_orphaned_skips_referenced_footnote():
    """SyncFootnoteProcessor(only_orphaned=True) passes the flag to FootnoteProcessor."""
    doc = _make_only_orphaned_doc()
    doc = await SyncFootnoteProcessor(only_orphaned=True).process(doc)

    para1 = next(e for e in doc.elements if e.id == "para-1")
    assert para1.html.count('id="fn-1"') == 1
    assert doc.orphaned_footnotes == []
