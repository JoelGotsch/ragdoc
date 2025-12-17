"""Tests for split_hierarchical."""
import pytest

from ragdoc.document import Document, Footnote, Heading, Paragraph
from ragdoc.splitting.hierarchical import split_hierarchical


def h(level: int, text: str = "") -> Heading:
    return Heading(innerhtml=text or f"H{level}", level=level)


def p(text: str = "para") -> Paragraph:
    return Paragraph(html=f"<p>{text}</p>")


def heading_levels(doc: Document) -> list[int]:
    return [h.level for h in doc.headings]


# ---------------------------------------------------------------------------
# No-split cases
# ---------------------------------------------------------------------------

def test_no_headings_returns_original():
    doc = Document(elements=[p()])
    result = split_hierarchical(doc)
    assert result == [doc]


def test_single_heading_returns_original():
    doc = Document(elements=[h(1)])
    result = split_hierarchical(doc)
    assert result == [doc]


def test_all_unique_levels_returns_original():
    # [1, 2, 3] — only one heading at each level, no split possible
    doc = Document(elements=[h(1), h(2), h(3)])
    result = split_hierarchical(doc)
    assert result == [doc]


def test_empty_document_returns_original():
    doc = Document(elements=[])
    result = split_hierarchical(doc)
    assert result == [doc]


# ---------------------------------------------------------------------------
# Basic splitting (no context propagation needed)
# ---------------------------------------------------------------------------

def test_two_headings_same_level_splits_into_two():
    h2a, h2b = h(2, "A"), h(2, "B")
    doc = Document(elements=[h2a, h2b])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert result[0].elements == [h2a]
    assert result[1].elements == [h2b]


def test_three_headings_same_level_splits_into_three():
    h2a, h2b, h2c = h(2, "A"), h(2, "B"), h(2, "C")
    doc = Document(elements=[h2a, h2b, h2c])
    result = split_hierarchical(doc)
    assert len(result) == 3
    assert result[0].elements == [h2a]
    assert result[1].elements == [h2b]
    assert result[2].elements == [h2c]


def test_split_includes_child_elements():
    h2a, para_a, h2b, para_b = h(2, "A"), p("a"), h(2, "B"), p("b")
    doc = Document(elements=[h2a, para_a, h2b, para_b])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert result[0].elements == [h2a, para_a]
    assert result[1].elements == [h2b, para_b]


def test_split_includes_child_headings():
    h2a, h3, h2b = h(2, "A"), h(3, "sub"), h(2, "B")
    doc = Document(elements=[h2a, h3, h2b])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert result[0].elements == [h2a, h3]
    assert result[1].elements == [h2b]


# ---------------------------------------------------------------------------
# Context propagation — docstring examples
# ---------------------------------------------------------------------------

def test_docstring_example_2_context_from_preamble():
    """[2, 1, 3, 3, 3] → split_level=3 → [[2,1,3], [2,1,3], [2,1,3]]"""
    h2, h1, h3a, h3b, h3c = h(2), h(1), h(3, "A"), h(3, "B"), h(3, "C")
    doc = Document(elements=[h2, h1, h3a, h3b, h3c])
    result = split_hierarchical(doc)
    assert len(result) == 3
    assert heading_levels(result[0]) == [2, 1, 3]
    assert heading_levels(result[1]) == [2, 1, 3]
    assert heading_levels(result[2]) == [2, 1, 3]
    # All three chunks share the same context headings (by level)
    assert result[0].elements[0].level == 2
    assert result[0].elements[1].level == 1
    assert result[0].elements[2].level == 3


def test_docstring_example_3_context_from_preamble():
    """[1, 2, 3, 3, 2, 3] → split_level=2 → [[1,2,3,3], [1,2,3]]"""
    h1, h2a, h3a, h3b, h2b, h3c = h(1), h(2, "A"), h(3), h(3), h(2, "B"), h(3)
    doc = Document(elements=[h1, h2a, h3a, h3b, h2b, h3c])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert heading_levels(result[0]) == [1, 2, 3, 3]
    assert heading_levels(result[1]) == [1, 2, 3]


def test_docstring_example_5_context_from_within_chunk():
    """[2, 1, 3, 3, 2] → split_level=2 → [[2,1,3,3], [1,2]]"""
    h2a, h1, h3a, h3b, h2b = h(2, "A"), h(1), h(3), h(3), h(2, "B")
    doc = Document(elements=[h2a, h1, h3a, h3b, h2b])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert heading_levels(result[0]) == [2, 1, 3, 3]
    assert heading_levels(result[1]) == [1, 2]


def test_corrected_example_1_context_with_h1_inside_chunk():
    """[2, 1, 3, 2, 3, 4, 4, 3] → split_level=2 → [[2,1,3],[1,2,3,4,4,3]]"""
    h2a, h1, h3a, h2b, h3b, h4a, h4b, h3c = (
        h(2, "A"), h(1), h(3, "a"), h(2, "B"), h(3, "b"), h(4), h(4), h(3, "c")
    )
    doc = Document(elements=[h2a, h1, h3a, h2b, h3b, h4a, h4b, h3c])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert heading_levels(result[0]) == [2, 1, 3]
    assert heading_levels(result[1]) == [1, 2, 3, 4, 4, 3]


# ---------------------------------------------------------------------------
# Context ordering
# ---------------------------------------------------------------------------

def test_context_prepended_in_document_order():
    """Context headings are prepended in document order (h2 before h1 if h2 came first)."""
    h2, h1, h3a, h3b = h(2), h(1), h(3, "A"), h(3, "B")
    doc = Document(elements=[h2, h1, h3a, h3b])
    result = split_hierarchical(doc)
    # split_level=3, context=[h2, h1] in document order
    assert len(result) == 2
    assert result[0].elements[0] is h2
    assert result[0].elements[1] is h1
    assert result[0].elements[2] is h3a


def test_context_updated_from_within_chunk():
    """A parent heading appearing inside a chunk body updates context for subsequent chunks."""
    # [1, 3, 2, 3] → split_level=3 (count(h3)=2, h1 and h2 each appear once)
    # h1 is preamble context; h2 appears inside chunk 1's body, adding it to context for chunk 2
    h1, h3a, h2, h3b = h(1), h(3, "A"), h(2), h(3, "B")
    doc = Document(elements=[h1, h3a, h2, h3b])
    result = split_hierarchical(doc)
    assert len(result) == 2
    # Chunk 1: context [h1] prepended + body [h3a, h2]
    assert heading_levels(result[0]) == [1, 3, 2]
    assert result[0].elements[0] is h1
    assert result[0].elements[1] is h3a
    assert result[0].elements[2] is h2
    # Chunk 2: context now [h1, h2] (h2 was seen in chunk 1 body) + body [h3b]
    assert heading_levels(result[1]) == [1, 2, 3]
    assert result[1].elements[0] is h1
    assert result[1].elements[1] is h2
    assert result[1].elements[2] is h3b


# ---------------------------------------------------------------------------
# Non-heading elements in preamble
# ---------------------------------------------------------------------------

def test_preamble_non_heading_elements_included_in_first_chunk():
    """Paragraphs before first split_level heading go into the first chunk."""
    intro, h2a, h2b = p("intro"), h(2, "A"), h(2, "B")
    doc = Document(elements=[intro, h2a, h2b])
    result = split_hierarchical(doc)
    assert len(result) == 2
    assert intro in result[0].elements
    assert intro not in result[1].elements


# ---------------------------------------------------------------------------
# ExternalRef relationships
# ---------------------------------------------------------------------------

def test_split_docs_have_original_as_parent():
    doc = Document(elements=[h(1, "A"), h(1, "B")])
    result = split_hierarchical(doc)
    for split_doc in result:
        parent_ids = split_doc.parent_ids
        assert doc.id in parent_ids


def test_no_split_no_external_refs_added():
    doc = Document(elements=[h(1)])
    original_refs = list(doc.external_refs)
    result = split_hierarchical(doc)
    assert result == [doc]
    assert doc.external_refs == original_refs


# ---------------------------------------------------------------------------
# Metadata and title propagation
# ---------------------------------------------------------------------------

def test_split_preserves_title():
    doc = Document(elements=[h(1, "A"), h(1, "B")], title="My Doc")
    result = split_hierarchical(doc)
    assert all(d.title == "My Doc" for d in result)


def test_split_preserves_metadata():
    doc = Document(elements=[h(1, "A"), h(1, "B")], metadata={"source": "test"})
    result = split_hierarchical(doc)
    assert all(d.metadata == {"source": "test"} for d in result)


# ---------------------------------------------------------------------------
# Ref-awareness: footnotes across split boundaries (known bug)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="split_hierarchical lacks ref-awareness (known bug)")
def test_footnote_stranded_across_heading_split_boundary():
    """Footnote physically after a split-level heading but referenced by paragraph before it
    ends up in the wrong split — demonstrating the ref-blindness of split_hierarchical.

    Root cause: split_hierarchical assigns elements to chunks purely by their physical
    position relative to headings. It does not use build_element_groups, so a Footnote
    element that lives after the split boundary is placed in the second split even when
    it is referenced (via <ref rel="footnote"/>) by a paragraph in the first split.

    Document structure:
        h2a, para_with_ref → split 1 (para_with_ref contains <ref id=fn1.id .../>)
        h2b, fn1, para_b   → split 2 (fn1 is the footnote body)

    Expected (correct): fn1 travels with para_with_ref into split 1.
    Actual (current bug): fn1 stays in split 2, leaving split 1 with a missing ref
    and split 2 with an orphaned footnote.
    """
    fn1 = Footnote(number=1, innerhtml="Footnote body text.")
    para_with_ref = Paragraph(html=f'<p>Text with footnote.<ref id="{fn1.id}" rel="footnote"/></p>')
    h2a = h(2, "Section A")
    h2b = h(2, "Section B")
    para_b = Paragraph(html="<p>Section B content.</p>")

    # fn1 is physically in Section B (after h2b) but referenced from Section A
    doc = Document(elements=[h2a, para_with_ref, h2b, fn1, para_b])

    result = split_hierarchical(doc)

    assert len(result) == 2
    split_a, split_b = result[0], result[1]

    assert para_with_ref in split_a.elements, "para_with_ref should be in split_a"

    # fn1 should follow its referencing paragraph into split_a, not stay in split_b
    assert fn1 in split_a.elements, (
        f"BUG: fn1 (id={fn1.id!r}) ended up in split_b but is referenced from "
        "para_with_ref in split_a. split_hierarchical lacks ref-awareness."
    )
    assert fn1 not in split_b.elements
