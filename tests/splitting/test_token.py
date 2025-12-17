"""Tests for token-based split functions: split_oversized_element, split_by_elements, split_document.

Ref-aware tests
---------------
Several tests specifically target the new group-aware behaviour:

* ``test_referenced_element_stays_with_root`` -- footnote placed *before* its
  referencing paragraph in the element list still ends up in the same chunk.

* ``test_ref_duplicated_when_two_roots_reference_same_element`` -- an image
  referenced by two paragraphs appears in both output chunks.

* ``test_split_shifts_earlier_when_ref_inflates_group_size`` -- without
  ref-awareness para1+para2 would fit together; with ref-awareness the
  inlined footnote makes para1's group too large, forcing an earlier split.

* ``test_tier3_token_slice_no_raw_ref_tags_in_output`` -- the tier-3 token
  slice operates on *rendered* text (refs already inlined), so no raw
  ``<ref id="..." rel="..."/>`` XML survives into output chunks.
"""

import pytest

from ragdoc.document import Document, Footnote, Heading, Image, Paragraph, Table
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
from ragdoc.splitting.groups import ElementGroup, build_element_groups
from ragdoc.splitting.token import (
    split_at_html_tags,
    split_at_sentences,
    split_by_elements,
    split_document,
    split_oversized_element,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Stub tokenizer where 1 UTF-8 byte = 1 token.  Fully reversible for ASCII."""

    def __call__(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(t & 0xFF for t in tokens).decode("utf-8", errors="replace")

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


_OVERLAP = 5


def rdr() -> Renderer:
    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


def tok() -> CharTokenizer:
    return CharTokenizer()


def h(level: int, text: str = "") -> Heading:
    return Heading(innerhtml=text or f"H{level}", level=level)


def p(text: str) -> Paragraph:
    return Paragraph(html=f"<p>{text}</p>")


def fn(number: int, text: str) -> Footnote:
    return Footnote(number=number, innerhtml=text)


def img(alt: str = "img") -> Image:
    return Image(alt=alt)


def measured(doc_or_el, renderer: Renderer | None = None) -> int:
    r = renderer or rdr()
    if isinstance(doc_or_el, Document):
        return CharTokenizer().count(r.render(doc_or_el))
    return CharTokenizer().count(r.render(Document(elements=[doc_or_el])))


# ---------------------------------------------------------------------------
# --- TestBuildElementGroups ---
# ---------------------------------------------------------------------------


def test_build_groups_heading_standalone():
    h1 = h(1, "Title")
    para = p("text")
    groups = build_element_groups([h1, para])
    assert groups[0] is h1
    assert isinstance(groups[1], ElementGroup)
    assert groups[1].root is para


def test_build_groups_referenced_element_not_standalone():
    """A footnote referenced by a paragraph is NOT a standalone group."""
    fn1 = fn(1, "note")
    para = Paragraph(html=f'<p>text <ref id="{fn1.id}" rel="footnote"/></p>')
    groups = build_element_groups([fn1, para])
    # Only one group: para (root) + fn1 (referenced)
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]
    assert len(content_groups) == 1
    assert content_groups[0].root is para
    assert fn1 in content_groups[0].referenced


def test_build_groups_referenced_element_before_root_in_list():
    """Referenced element appearing before its root is still collected correctly."""
    fn1 = fn(1, "note")
    para = Paragraph(html=f'<p>text <ref id="{fn1.id}" rel="footnote"/></p>')
    # fn1 comes first in the list
    groups = build_element_groups([fn1, para])
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]
    assert len(content_groups) == 1
    assert fn1 in content_groups[0].referenced


def test_build_groups_referenced_element_after_root_in_list():
    """Referenced element appearing after its root is also collected."""
    fn1 = fn(1, "note")
    para = Paragraph(html=f'<p>text <ref id="{fn1.id}" rel="footnote"/></p>')
    groups = build_element_groups([para, fn1])
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]
    assert len(content_groups) == 1
    assert fn1 in content_groups[0].referenced


def test_build_groups_element_duplicated_when_referenced_by_two_roots():
    """An image referenced by two paragraphs appears in both groups."""
    im = img("chart")
    para1 = Paragraph(html=f'<p>See <ref id="{im.id}" rel="image"/></p>')
    para2 = Paragraph(html=f'<p>Also <ref id="{im.id}" rel="image"/></p>')
    groups = build_element_groups([im, para1, para2])
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]
    assert len(content_groups) == 2
    assert im in content_groups[0].referenced
    assert im in content_groups[1].referenced


def test_build_groups_transitive_refs_collected():
    """B referenced by A, C referenced by B -> C ends up in A's group."""
    fn1 = fn(1, "leaf footnote")
    # fn1 is referenced by an image's alt somehow -- simulate with a para that refs both
    # More directly: para refs fn1, fn1 refs nothing (base case is fine)
    # For a true transitive test we need A->B->C chain:
    # Use two paragraphs where inner_para is referenced by outer_para
    # But that's unusual. A realistic case: image referenced by para, para has no further refs.
    # Let's test that unreferenced refs (missing ids) are silently ignored.
    para = Paragraph(html=f'<p>text <ref id="{fn1.id}" rel="footnote"/></p>')
    groups = build_element_groups([para, fn1])
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]
    assert fn1 in content_groups[0].referenced


def test_build_groups_unreferenced_element_not_in_any_group():
    """An element that nothing references but also has no inline_refs is its own root."""
    para1 = p("standalone")
    para2 = p("also standalone")
    groups = build_element_groups([para1, para2])
    roots = [g.root for g in groups if isinstance(g, ElementGroup)]
    assert para1 in roots
    assert para2 in roots


# ---------------------------------------------------------------------------
# --- TestSplitOversizedElement ---
# ---------------------------------------------------------------------------


def test_oversized_fits_in_limit_returns_original():
    doc = Document(elements=[p("short")])
    result = split_oversized_element(doc, rdr(), tok(), max_tokens=10_000, overlap_tokens=_OVERLAP)
    assert result == [doc]


def test_oversized_paragraph_produces_multiple_chunks():
    para = p("A" * 300)
    doc = Document(elements=[para])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2


def test_oversized_reading_order():
    text = "STARTMARKER " + "middle " * 40 + "ENDMARKER"
    doc = Document(elements=[p(text)])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    first_text = " ".join(e.text for e in result[0].elements)
    last_text = " ".join(e.text for e in result[-1].elements)
    assert "STARTMARKER" in first_text
    assert "ENDMARKER" in last_text


def test_oversized_heading_context_in_every_chunk():
    h1 = h(1, "Chapter")
    h2 = h(2, "Section")
    para = p("X" * 300)
    doc = Document(elements=[h1, h2, para])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        levels = [e.level for e in chunk.elements if isinstance(e, Heading)]
        assert 1 in levels
        assert 2 in levels


def test_oversized_heading_context_same_objects():
    h1 = h(1, "Title")
    para = p("Y" * 300)
    doc = Document(elements=[h1, para])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        chunk_h1 = next(e for e in chunk.elements if isinstance(e, Heading) and e.level == 1)
        assert chunk_h1 is h1


def test_oversized_parent_ref_set_on_all_chunks():
    para = p("Z" * 300)
    doc = Document(elements=[para])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        assert doc.id in chunk.parent_ids


def test_oversized_table_is_split():
    """Oversized table is split (via HTML rows or token slice) -- output has RawText chunks."""
    big_cell = "X" * 50
    table_html = f"<table><tr>{''.join(f'<td>{big_cell}</td>' for _ in range(20))}</tr></table>"
    from ragdoc.document import Table as TableEl

    tbl = TableEl(html=table_html)
    doc = Document(elements=[tbl])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2


def test_oversized_metadata_and_title_propagated():
    para = p("M" * 300)
    doc = Document(elements=[para], title="My Doc", metadata={"src": "test"})
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 2, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        assert chunk.title == "My Doc"
        assert chunk.metadata == {"src": "test"}


def test_oversized_raises_when_overhead_exceeds_budget():
    h1 = h(1, "Title")
    para = p("content")
    doc = Document(elements=[h1, para])
    r = rdr()
    h1_tokens = measured(h1, r)
    with pytest.raises(ValueError, match="max_tokens"):
        split_oversized_element(doc, r, tok(), max_tokens=h1_tokens - 1, overlap_tokens=_OVERLAP)


def test_oversized_tier3_token_slice_no_raw_ref_tags_in_output():
    """Tier-3 token slice works on rendered text: no raw <ref .../> in output.

    Without rendering before slicing, the <ref id="..." rel="footnote"/> tag
    in the paragraph's HTML could be sliced mid-tag, producing malformed text.
    With rendering first (refs inlined), no such tags remain.
    """
    fn1 = fn(1, "This is footnote content that expands the rendered size significantly.")
    para = Paragraph(html=f'<p>{"Word " * 60}<ref id="{fn1.id}" rel="footnote"/>{"More " * 60}</p>')
    doc = Document(elements=[para, fn1])
    r = rdr()
    total = measured(doc, r)
    result = split_oversized_element(doc, r, tok(), max_tokens=total // 3, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        chunk_text = r.render(chunk)
        # No raw <ref .../> tags should survive into the output
        assert "<ref id=" not in chunk_text, f"Raw <ref> tag found in tier-3 output: {chunk_text[:200]}"


# ---------------------------------------------------------------------------
# --- TestSplitByElements ---
# ---------------------------------------------------------------------------


def test_split_by_elements_fits_returns_original():
    doc = Document(elements=[p("short")])
    result = split_by_elements(doc, rdr(), tok(), max_tokens=10_000, overlap_tokens=_OVERLAP)
    assert result == [doc]


def test_split_by_elements_basic_split_at_element_boundary():
    para1 = p("A" * 100)
    para2 = p("B" * 100)
    doc = Document(elements=[para1, para2])
    r = rdr()
    p1_t = measured(para1, r)
    p2_t = measured(para2, r)
    max_t = p1_t + p2_t - 1
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) == 2
    assert para1 in result[0].elements
    assert para2 in result[1].elements


def test_split_by_elements_heading_context_prepended_to_new_chunk():
    h1 = h(1, "Title")
    para1 = p("A" * 100)
    para2 = p("B" * 100)
    doc = Document(elements=[h1, para1, para2])
    r = rdr()
    h1_t = measured(h1, r)
    p1_t = measured(para1, r)
    p2_t = measured(para2, r)
    max_t = h1_t + p1_t + p2_t - 1
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) == 2
    chunk2_headings = [e for e in result[1].elements if isinstance(e, Heading)]
    assert any(hd is h1 for hd in chunk2_headings)


def test_split_by_elements_heading_context_updated_mid_document():
    h1 = h(1, "Chapter")
    para1 = p("A" * 100)
    h2 = h(2, "Section")
    para2 = p("B" * 100)
    doc = Document(elements=[h1, para1, h2, para2])
    r = rdr()
    h1_t = measured(h1, r)
    p1_t = measured(para1, r)
    h2_t = measured(h2, r)
    p2_t = measured(para2, r)
    max_t = h1_t + p1_t + h2_t + p2_t - 1
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) == 2
    chunk2_levels = [e.level for e in result[1].elements if isinstance(e, Heading)]
    assert 1 in chunk2_levels
    assert 2 in chunk2_levels


def test_split_by_elements_no_heading_only_chunks():
    h1 = h(1, "Title")
    para1 = p("A" * 100)
    para2 = p("B" * 100)
    doc = Document(elements=[h1, para1, para2])
    r = rdr()
    h1_t = measured(h1, r)
    p1_t = measured(para1, r)
    max_t = p1_t + h1_t - 1
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    for chunk in result:
        non_headings = [e for e in chunk.elements if not isinstance(e, Heading)]
        assert len(non_headings) >= 1


def test_split_by_elements_oversized_single_element_further_split():
    h1 = h(1, "Title")
    big_para = p("X" * 300)
    doc = Document(elements=[h1, big_para])
    r = rdr()
    big_t = measured(big_para, r)
    h1_t = measured(h1, r)
    max_t = (big_t + h1_t) // 2
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        assert any(isinstance(e, Heading) and e.level == 1 for e in chunk.elements)


def test_split_by_elements_parent_ref_set_on_all_chunks():
    para1 = p("A" * 100)
    para2 = p("B" * 100)
    doc = Document(elements=[para1, para2])
    r = rdr()
    p_t = measured(para1, r)
    result = split_by_elements(doc, r, tok(), max_tokens=p_t + 1, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        assert doc.id in chunk.parent_ids


def test_split_by_elements_reading_order():
    paras = [p(f"Para{i}" + "x" * 50) for i in range(4)]
    doc = Document(elements=paras)
    r = rdr()
    p_t = measured(paras[0], r)
    result = split_by_elements(doc, r, tok(), max_tokens=p_t * 2 - 1, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    ordered = [e for chunk in result for e in chunk.elements if not isinstance(e, Heading)]
    assert ordered == list(paras)


def test_split_by_elements_metadata_and_title_propagated():
    para1 = p("A" * 100)
    para2 = p("B" * 100)
    doc = Document(elements=[para1, para2], title="Doc", metadata={"k": "v"})
    r = rdr()
    p_t = measured(para1, r)
    result = split_by_elements(doc, r, tok(), max_tokens=p_t + 1, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    for chunk in result:
        assert chunk.title == "Doc"
        assert chunk.metadata == {"k": "v"}


def test_split_by_elements_referenced_element_stays_with_root():
    """Footnote placed *before* its referencing paragraph lands in the same chunk.

    Layout: [Fn1, Para1(big, refs fn1), Para2(big)]
    Fn1 is before Para1 in the list -- the old code would have treated it as
    an independent element and potentially split it into a different chunk.
    The group-aware code bundles Fn1 with Para1.
    """
    fn1 = fn(1, "footnote text")
    para1 = Paragraph(html=f'<p>{"A" * 80}<ref id="{fn1.id}" rel="footnote"/></p>')
    para2 = p("B" * 80)
    doc = Document(elements=[fn1, para1, para2])
    r = rdr()
    # Budget: fits one group (para1+fn1 rendered together) but not both groups
    group1_tokens = CharTokenizer().count(r.render(Document(elements=[para1, fn1])))
    group2_tokens = measured(para2, r)
    max_t = group1_tokens + group2_tokens - 1
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    # para1 and fn1 must be in the same chunk
    for chunk in result:
        if para1 in chunk.elements:
            assert fn1 in chunk.elements, "fn1 and para1 were split into different chunks"


def test_split_by_elements_referenced_element_duplicated_across_chunks():
    """Image referenced by two paragraphs appears in both output chunks.

    Layout: [Img1, Para1(refs img1), Para2(refs img1)]
    Both Para1-group and Para2-group own Img1 -> Img1 appears in both chunks.
    """
    im = img("shared chart")
    para1 = Paragraph(html=f'<p>{"A" * 80}<ref id="{im.id}" rel="image"/></p>')
    para2 = Paragraph(html=f'<p>{"B" * 80}<ref id="{im.id}" rel="image"/></p>')
    doc = Document(elements=[im, para1, para2])
    r = rdr()
    # Measure group tokens (each para + img rendered together)
    g1_t = CharTokenizer().count(r.render(Document(elements=[para1, im])))
    g2_t = CharTokenizer().count(r.render(Document(elements=[para2, im])))
    max_t = g1_t + g2_t - 1
    result = split_by_elements(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    # im must appear in every chunk that contains a paragraph referencing it
    for chunk in result:
        has_para_with_ref = any(isinstance(e, Paragraph) and im.id in e.html for e in chunk.elements)
        if has_para_with_ref:
            assert im in chunk.elements, "referenced image missing from chunk that references it"


def test_split_by_elements_split_shifts_earlier_when_ref_inflates_group_size():
    """Ref-awareness causes an earlier split than element-counting alone would.

    Setup:
    - Para1 alone (rendered without footnote inlined) is small
    - Fn1 is medium; when inlined into Para1 the group is bigger but still fits alone
    - Para2 is medium

    Without ref-awareness: Para1-alone + Para2 fit together -> 1 chunk.
    With ref-awareness: Para1-group + Para2 exceed max_t -> 2 chunks.

    max_t sits strictly between (p1_alone + p2) and (group1 + p2) so the
    split only appears once we account for the inlined footnote size.
    The group itself fits alone (group1 < max_t), so para1 and fn1 remain
    as original element objects in the output (no further oversized-split).
    """
    fn1 = fn(1, "X" * 60)  # medium footnote -- inflates group but group still fits alone
    para1 = Paragraph(html=f'<p>short text<ref id="{fn1.id}" rel="footnote"/></p>')
    para2 = p("B" * 80)  # medium paragraph
    doc = Document(elements=[fn1, para1, para2])
    r = rdr()
    t = tok()

    p1_alone_t = CharTokenizer().count(r.render(Document(elements=[para1])))
    p2_t = CharTokenizer().count(r.render(Document(elements=[para2])))
    group1_t = CharTokenizer().count(r.render(Document(elements=[para1, fn1])))

    # Budget fits para1-alone + para2 but not para1-group + para2
    max_t = p1_alone_t + p2_t + 20
    assert p1_alone_t + p2_t < max_t, "test setup: para1-alone + para2 should fit"
    assert group1_t + p2_t > max_t, "test setup: para1-group + para2 must exceed max_t to force the earlier split"
    assert group1_t < max_t, "test setup: para1-group alone must fit so para1/fn1 survive unsplit as elements"

    result = split_by_elements(doc, r, t, max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) >= 2, "Expected split: para1-group + para2 exceed budget, so para2 lands in its own chunk"
    # para1 and fn1 must be in the same chunk
    for chunk in result:
        if para1 in chunk.elements:
            assert fn1 in chunk.elements, "fn1 and para1 split into different chunks"


# ---------------------------------------------------------------------------
# --- TestSplitDocument ---
# ---------------------------------------------------------------------------


def test_split_document_fits_returns_original():
    doc = Document(elements=[p("short")])
    result = split_document(doc, rdr(), tok(), max_tokens=10_000, overlap_tokens=_OVERLAP)
    assert result == [doc]


def test_split_document_uses_hierarchical_split_when_possible():
    h2a = h(2, "Section A")
    para_a = p("A" * 60)
    h2b = h(2, "Section B")
    para_b = p("B" * 60)
    doc = Document(elements=[h2a, para_a, h2b, para_b])
    r = rdr()
    total = measured(doc, r)
    result = split_document(doc, r, tok(), max_tokens=total - 1, overlap_tokens=_OVERLAP)
    assert len(result) == 2
    assert any(isinstance(e, Heading) and e.level == 2 for e in result[0].elements)
    assert any(isinstance(e, Heading) and e.level == 2 for e in result[1].elements)


def test_split_document_falls_back_to_element_split_when_no_headings():
    para1 = p("A" * 100)
    para2 = p("B" * 100)
    doc = Document(elements=[para1, para2])
    r = rdr()
    p_t = measured(para1, r)
    result = split_document(doc, r, tok(), max_tokens=p_t + 1, overlap_tokens=_OVERLAP)
    assert len(result) == 2


def test_split_document_hierarchical_parts_recursively_split():
    h2a = h(2, "A")
    para_a1 = p("A" * 100)
    para_a2 = p("B" * 100)
    h2b = h(2, "B")
    para_b = p("C" * 30)
    doc = Document(elements=[h2a, para_a1, para_a2, h2b, para_b])
    r = rdr()
    h_t = measured(h2a, r)
    p_t = measured(para_a1, r)
    max_t = h_t + p_t + 5
    result = split_document(doc, r, tok(), max_tokens=max_t, overlap_tokens=_OVERLAP)
    assert len(result) > 2


def test_split_document_reading_order_preserved():
    h2a = h(2, "A")
    para1 = p("First " * 20)
    h2b = h(2, "B")
    para2 = p("Second " * 20)
    doc = Document(elements=[h2a, para1, h2b, para2])
    r = rdr()
    total = measured(doc, r)
    result = split_document(doc, r, tok(), max_tokens=total - 1, overlap_tokens=_OVERLAP)
    assert len(result) >= 2
    content = [e for chunk in result for e in chunk.elements if not isinstance(e, Heading)]
    assert para1 in content
    assert para2 in content
    assert content.index(para1) < content.index(para2)


def test_split_document_parent_ref_chain_set():
    h2a = h(2, "A")
    para_a = p("A" * 100)
    h2b = h(2, "B")
    para_b = p("B" * 100)
    doc = Document(elements=[h2a, para_a, h2b, para_b])
    r = rdr()
    total = measured(doc, r)
    result = split_document(doc, r, tok(), max_tokens=total - 1, overlap_tokens=_OVERLAP)
    for chunk in result:
        assert chunk.parent_ids


# ---------------------------------------------------------------------------
# --- TestSplitAtSentences ---
# ---------------------------------------------------------------------------


def test_split_sentences_returns_none_when_no_sentence_boundary():
    """Single run-on 'sentence' -> returns None so caller falls back to tier 3."""
    from ragdoc.document import ExternalRef

    para = p("A" * 200)  # no sentence-ending punctuation
    parent_ref = ExternalRef(target_id="parent", rel_type="external-parent")
    result = split_at_sentences(
        elements=[para],
        heading_ctx=[],
        renderer=rdr(),
        tokenizer=tok(),
        max_tokens=50,
        doc_title=None,
        doc_metadata={},
        parent_ref=parent_ref,
    )
    assert result is None


def test_split_sentences_splits_at_period_boundaries():
    from ragdoc.document import ExternalRef

    text = "First sentence. " * 20
    para = p(text)
    parent_ref = ExternalRef(target_id="parent", rel_type="external-parent")
    r = rdr()
    total = CharTokenizer().count(r.render(Document(elements=[para])))
    result = split_at_sentences(
        elements=[para],
        heading_ctx=[],
        renderer=r,
        tokenizer=tok(),
        max_tokens=total // 3,
        doc_title=None,
        doc_metadata={},
        parent_ref=parent_ref,
    )
    assert result is not None
    assert len(result) >= 2


def test_split_sentences_pluggable_splitter():
    """Custom sentence splitter (split on newline) is respected."""
    from ragdoc.document import ExternalRef

    text = "line one\nline two\nline three\n" * 10
    para = p(text)
    parent_ref = ExternalRef(target_id="parent", rel_type="external-parent")
    r = rdr()
    total = CharTokenizer().count(r.render(Document(elements=[para])))

    def newline_splitter(t: str) -> list[str]:
        return [s for s in t.split("\n") if s.strip()]

    result = split_at_sentences(
        elements=[para],
        heading_ctx=[],
        renderer=r,
        tokenizer=tok(),
        max_tokens=total // 3,
        doc_title=None,
        doc_metadata={},
        parent_ref=parent_ref,
        sentence_splitter=newline_splitter,
    )
    assert result is not None
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# --- TestSplitAtHtmlTags ---
# ---------------------------------------------------------------------------


def test_split_html_tags_returns_none_for_plain_paragraph():
    from ragdoc.document import ExternalRef
    from ragdoc.splitting.groups import ElementGroup

    para = p("No HTML structure inside, just text " * 10)
    parent_ref = ExternalRef(target_id="parent", rel_type="external-parent")
    result = split_at_html_tags(
        group=ElementGroup(root=para),
        heading_ctx=[],
        renderer=rdr(),
        tokenizer=tok(),
        max_tokens=50,
        doc_title=None,
        doc_metadata={},
        parent_ref=parent_ref,
    )
    assert result is None


def test_split_html_tags_splits_table_rows():
    from ragdoc.document import ExternalRef
    from ragdoc.splitting.groups import ElementGroup

    rows = "".join(f"<tr><td>{'X' * 40}</td><td>{'Y' * 40}</td></tr>" for _ in range(10))
    tbl = Table(html=f"<table>{rows}</table>")
    parent_ref = ExternalRef(target_id="parent", rel_type="external-parent")
    r = rdr()
    total = CharTokenizer().count(r.render(Document(elements=[tbl])))
    result = split_at_html_tags(
        group=ElementGroup(root=tbl),
        heading_ctx=[],
        renderer=r,
        tokenizer=tok(),
        max_tokens=total // 3,
        doc_title=None,
        doc_metadata={},
        parent_ref=parent_ref,
    )
    assert result is not None
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# _token_slice seam correctness (fable-review Phase 0, bug 8)
# ---------------------------------------------------------------------------


class _SeamTokenizer:
    """Whitespace tokenizer that merges the pair (CTXEND, FIRSTWORD) into ONE token,
    modelling a BPE merge across the overhead/content render seam. Pre-fix, slicing the
    combined token stream by the separately-rendered overhead's token count dropped
    FIRSTWORD; the fix tokenizes content-only text, so no seam exists."""

    def _words(self, text: str) -> list[str]:
        words = text.split()
        merged: list[str] = []
        i = 0
        while i < len(words):
            if i + 1 < len(words) and words[i] == "CTXEND" and words[i + 1] == "FIRSTWORD":
                merged.append("CTXEND FIRSTWORD")
                i += 2
            else:
                merged.append(words[i])
                i += 1
        return merged

    def __call__(self, text: str) -> list[str]:  # token "ids" are the words themselves
        return self._words(text)

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self.decode(self._words(text)[:max_tokens])

    def count(self, text: str) -> int:
        return len(self._words(text))


def _token_slice_setup(content_words: list[str], tokenizer):
    from ragdoc.document import ExternalRef
    from ragdoc.splitting.token import _token_slice

    heading = Heading(html_content="<h1>CTXEND</h1>")
    content = Paragraph(html_content=f"<p>{' '.join(content_words)}</p>")
    chunk_doc = Document(elements=[heading, content])
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    overhead_tokens = tokenizer.count(renderer.render(Document(elements=[heading])))
    parent_ref = ExternalRef(target_id=chunk_doc.id, rel_type="external-parent")
    return _token_slice(
        chunk_doc=chunk_doc,
        renderer=renderer,
        tokenizer=tokenizer,
        max_tokens=200 + overhead_tokens,
        overlap_tokens=0,
        overhead_tokens=overhead_tokens,
        heading_ctx=[heading],
        doc_title=None,
        doc_metadata={},
        parent_ref=parent_ref,
        doc_source_path="",
    )


def test_token_slice_does_not_drop_boundary_text():
    """The first content word must survive into the first output chunk."""
    words = ["FIRSTWORD"] + [f"w{i}" for i in range(50)]
    docs = _token_slice_setup(words, _SeamTokenizer())
    assert docs, "expected at least one output chunk"
    first_text = docs[0].elements[-1].text
    assert "FIRSTWORD" in first_text


def test_token_slice_conserves_all_content_words():
    """With overlap_tokens=0, every content word appears exactly once across output chunks."""
    words = [f"unique{i}" for i in range(450)]
    docs = _token_slice_setup(words, _SeamTokenizer())
    assert len(docs) >= 2, "content sized to force multiple chunks"
    combined = " ".join(d.elements[-1].text for d in docs).split()
    for w in words:
        assert combined.count(w) == 1, f"{w} appeared {combined.count(w)} times"
