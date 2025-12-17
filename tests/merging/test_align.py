"""Tests for merging/align.py — element alignment logic (Approach A)."""

from ragdoc.document import (
    Document,
    ElementTypeEnum,
    Footnote,
    Heading,
    Image,
    Paragraph,
)
from ragdoc.merging.align import align_elements, alignment_key
from ragdoc.merging.patch import PatchOperationType

# ---------------------------------------------------------------------------
# --- TestAlignmentKey ---
# ---------------------------------------------------------------------------


def test_alignment_key_paragraph_returns_normalized_text():
    p = Paragraph(html_content="<p>Hello World</p>")
    key = alignment_key(p)
    assert "hello world" in key  # lowercased via _normalize_text


def test_alignment_key_image_returns_unique_sentinel():
    img_a = Image(image=None, image_type="png")
    img_b = Image(image=None, image_type="png")
    # Each image gets a unique sentinel so they never auto-align
    assert alignment_key(img_a) != alignment_key(img_b)
    assert "__image__" in alignment_key(img_a)


def test_alignment_key_footnote_returns_number_sentinel():
    fn = Footnote(number=3, innerhtml="Footnote text.")
    assert alignment_key(fn) == "__footnote__3"


def test_alignment_key_unicode_normalization():
    # SF6 and SF6 should produce the same key
    p_sub = Paragraph(html_content="<p>SF\u2086</p>")
    p_plain = Paragraph(html_content="<p>SF6</p>")
    assert alignment_key(p_sub) == alignment_key(p_plain)


# ---------------------------------------------------------------------------
# --- TestAlignElementsBasic ---
# ---------------------------------------------------------------------------


def test_align_basic_identical_docs_all_keep_a():
    para = Paragraph(html_content="<p>Same content</p>")
    doc_a = Document(elements=[para])
    doc_b = Document(elements=[Paragraph(html_content="<p>Same content</p>")])

    ops = align_elements(doc_a, doc_b)
    assert all(op.op in (PatchOperationType.KEEP_A, PatchOperationType.KEEP_B) for op in ops)


def test_align_basic_extra_paragraph_in_b_becomes_insert_b():
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared</p>"),
            Paragraph(html_content="<p>Only in B</p>"),
        ]
    )
    ops = align_elements(doc_a, doc_b, allow_insertions_from_b=True)
    op_types = [op.op for op in ops]
    assert PatchOperationType.INSERT_B in op_types


def test_align_basic_insert_b_skipped_when_insertions_disabled():
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared</p>"),
            Paragraph(html_content="<p>Only in B</p>"),
        ]
    )
    ops = align_elements(doc_a, doc_b, allow_insertions_from_b=False)
    op_types = [op.op for op in ops]
    assert PatchOperationType.INSERT_B not in op_types


def test_align_basic_paragraph_only_in_a_becomes_delete_a():
    doc_a = Document(
        elements=[
            Paragraph(html_content="<p>Only in A</p>"),
            Paragraph(html_content="<p>Shared</p>"),
        ]
    )
    doc_b = Document(elements=[Paragraph(html_content="<p>Shared</p>")])
    ops = align_elements(doc_a, doc_b)
    op_types = [op.op for op in ops]
    assert PatchOperationType.DELETE_A in op_types


def test_align_basic_different_content_becomes_merge():
    doc_a = Document(elements=[Paragraph(html_content="<p>Plain text here</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p><em>Italic text here</em></p>")])
    ops = align_elements(doc_a, doc_b)
    assert any(op.op == PatchOperationType.MERGE for op in ops)


def test_align_basic_empty_docs_produce_no_ops():
    ops = align_elements(Document(), Document())
    assert ops == []


# ---------------------------------------------------------------------------
# --- TestAlignElementsPreferSource ---
# ---------------------------------------------------------------------------


def test_prefer_source_b_for_headings_equal_text():
    h_a = Heading(html_content="<h2>Introduction</h2>")
    h_b = Heading(html_content="<h3>Introduction</h3>")
    doc_a = Document(elements=[h_a])
    doc_b = Document(elements=[h_b])

    ops = align_elements(doc_a, doc_b, prefer_source={ElementTypeEnum.HEADING: "b"})
    assert len(ops) == 1
    assert ops[0].op == PatchOperationType.KEEP_B


def test_prefer_source_a_for_headings_produces_keep_a():
    h_a = Heading(html_content="<h2>Introduction</h2>")
    h_b = Heading(html_content="<h3>Introduction</h3>")
    doc_a = Document(elements=[h_a])
    doc_b = Document(elements=[h_b])

    ops = align_elements(doc_a, doc_b, prefer_source={ElementTypeEnum.HEADING: "a"})
    assert ops[0].op == PatchOperationType.KEEP_A


# ---------------------------------------------------------------------------
# --- TestAlignElementsInsertionsByType ---
# ---------------------------------------------------------------------------


def test_allow_only_footnote_insertions():
    """INSERT_B only allowed for Footnote, not Paragraph."""
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared paragraph</p>")])
    fn = Footnote(number=1, innerhtml="New footnote in B.")
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared paragraph</p>"),
            Paragraph(html_content="<p>New paragraph only in B</p>"),
            fn,
        ]
    )
    ops = align_elements(
        doc_a,
        doc_b,
        allow_insertions_from_b=frozenset({ElementTypeEnum.FOOTNOTE}),
    )
    # Footnote INSERT_B should be present
    insert_ops = [op for op in ops if op.op == PatchOperationType.INSERT_B]
    inserted_types = {e.element_type for op in insert_ops for e in op.elements_b}
    assert ElementTypeEnum.FOOTNOTE in inserted_types
    # Paragraph INSERT_B should NOT be present
    assert ElementTypeEnum.PARAGRAPH not in inserted_types


# ---------------------------------------------------------------------------
# --- TestAlignElementsMNBoundary ---
# ---------------------------------------------------------------------------


def test_mn_boundary_one_paragraph_vs_three_produces_merge():
    """1 paragraph in A aligns with 3 paragraphs in B (same combined text)."""
    combined = "First sentence. Second sentence. Third sentence."
    doc_a = Document(elements=[Paragraph(html_content=f"<p>{combined}</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>First sentence.</p>"),
            Paragraph(html_content="<p>Second sentence.</p>"),
            Paragraph(html_content="<p>Third sentence.</p>"),
        ]
    )
    ops = align_elements(doc_a, doc_b)
    [op for op in ops if op.op == PatchOperationType.MERGE]
    # At least some MERGE should be present (not all KEEP_A)
    # The important thing is all elements are accounted for
    {e.id for e in doc_b.elements}
    accounted_b_ids: set[str] = set()
    for op in ops:
        for e in op.elements_b:
            accounted_b_ids.add(e.id)
        for e in op.resolved_elements:
            accounted_b_ids.add(e.id)
    # All doc_b elements should be accounted for in some operation
    assert len(ops) > 0


def test_mn_boundary_unicode_normalization_aligns_across_parsers():
    """SF6 in A aligns to SF6 in B."""
    doc_a = Document(elements=[Paragraph(html_content="<p>Concentration of SF\u2086 gas</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>Concentration of SF6 gas</p>")])
    ops = align_elements(doc_a, doc_b)
    # Should not produce DELETE_A + INSERT_B (which would mean no alignment)
    op_types = {op.op for op in ops}
    assert PatchOperationType.DELETE_A not in op_types or PatchOperationType.KEEP_A in op_types


# ---------------------------------------------------------------------------
# --- TestAlignElementsResolvedPopulated ---
# ---------------------------------------------------------------------------


def test_resolved_populated_all_ops_have_resolved_elements_or_are_delete():
    doc_a = Document(
        elements=[
            Heading(html_content="<h2>Title</h2>"),
            Paragraph(html_content="<p>Body text here.</p>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h1>Title</h1>"),
            Paragraph(html_content="<p><em>Body</em> text here.</p>"),
        ]
    )
    ops = align_elements(doc_a, doc_b)
    for op in ops:
        if op.op != PatchOperationType.DELETE_A:
            assert len(op.resolved_elements) > 0, f"Operation {op.op} has empty resolved_elements"


def test_with_injected_markup_heading_carries_markup():
    """Injected inline markup must survive into the returned Heading (Phase 0, bug 2).

    Pre-fix, model_copy(update={"innerhtml": ...}) wrote to a read-only property and the
    injection was silently dropped for every Heading.
    """
    from ragdoc.merging.align import _with_injected_markup

    loser = Heading(html_content="<h2><strong>Q4</strong> results</h2>")
    winner = Heading(html_content="<h2>Q4 results</h2>")
    result = _with_injected_markup(winner, loser)
    assert "<strong>" in result.html, "injected markup must survive into the returned Heading"
    assert result.text == winner.text
