"""Tests for the debug_footnotes utility."""

from ragdoc.document import Document, Footnote, Heading, Paragraph
from ragdoc.utils.debug_footnotes import (
    describe_orphan_footnotes,
    describe_orphan_footnotes_batch,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_heading(text: str, **kwargs) -> Heading:
    return Heading(html_content=f"<h1>{text}</h1>", **kwargs)


def _make_footnote(fn_id: str, number: int, text: str, **kwargs) -> Footnote:
    return Footnote(id=fn_id, number=number, innerhtml=text, **kwargs)


def _make_paragraph(html: str) -> Paragraph:
    return Paragraph(html_content=f"<p>{html}</p>")


# =============================================================================
# Tests — describe_orphan_footnotes
# =============================================================================


def test_no_orphans():
    """Returns [] when all footnotes are referenced."""
    doc = Document(
        elements=[
            _make_paragraph('See <ref id="fn-1" rel="footnote"/>.'),
            _make_footnote("fn-1", 1, "Referenced footnote."),
        ]
    )
    assert describe_orphan_footnotes(doc) == []


def test_orphan_no_context():
    """Orphan with page=0 → page=None; no heading → preceding_heading=None."""
    doc = Document(
        elements=[
            _make_footnote("fn-1", 1, "Lonely footnote.", page=0),
        ]
    )
    result = describe_orphan_footnotes(doc)
    assert len(result) == 1
    info = result[0]
    assert info.number == 1
    assert info.text == "Lonely footnote."
    assert info.page is None
    assert info.preceding_heading is None


def test_orphan_with_page():
    """Orphan with page=3 → page=3."""
    doc = Document(
        elements=[
            _make_footnote("fn-1", 1, "Footnote on page 3.", page=3),
        ]
    )
    result = describe_orphan_footnotes(doc)
    assert len(result) == 1
    assert result[0].page == 3


def test_orphan_preceding_heading():
    """Heading before footnote in elements → correct heading text returned."""
    doc = Document(
        elements=[
            _make_heading("Introduction"),
            _make_footnote("fn-1", 1, "Orphan under Introduction."),
        ]
    )
    result = describe_orphan_footnotes(doc)
    assert len(result) == 1
    assert result[0].preceding_heading == "Introduction"


def test_orphan_skips_closer_heading_after():
    """A heading *after* the footnote is not returned."""
    doc = Document(
        elements=[
            _make_heading("Before"),
            _make_footnote("fn-1", 1, "Orphan footnote."),
            _make_heading("After"),
        ]
    )
    result = describe_orphan_footnotes(doc)
    assert len(result) == 1
    assert result[0].preceding_heading == "Before"


def test_nearest_heading_chosen():
    """Two headings before footnote → nearest (later) one is returned."""
    doc = Document(
        elements=[
            _make_heading("First"),
            _make_heading("Second"),
            _make_footnote("fn-1", 1, "Orphan after two headings."),
        ]
    )
    result = describe_orphan_footnotes(doc)
    assert len(result) == 1
    assert result[0].preceding_heading == "Second"


def test_sorted_by_number():
    """Result list is sorted by footnote number."""
    doc = Document(
        elements=[
            _make_footnote("fn-3", 3, "Third."),
            _make_footnote("fn-1", 1, "First."),
            _make_footnote("fn-2", 2, "Second."),
        ]
    )
    result = describe_orphan_footnotes(doc)
    assert [info.number for info in result] == [1, 2, 3]


# =============================================================================
# Tests — describe_orphan_footnotes_batch
# =============================================================================


def test_batch_excludes_docs_without_orphans():
    """Document with no orphans not in batch output."""
    doc_clean = Document(
        elements=[
            _make_paragraph('See <ref id="fn-1" rel="footnote"/>.'),
            _make_footnote("fn-1", 1, "Referenced."),
        ]
    )
    doc_orphan = Document(
        elements=[
            _make_footnote("fn-2", 2, "Orphan."),
        ]
    )
    result = describe_orphan_footnotes_batch([doc_clean, doc_orphan])
    assert len(result) == 1
    assert result[0][0] is doc_orphan


def test_batch_multiple_docs():
    """Returns one entry per document that has orphans."""
    doc_a = Document(elements=[_make_footnote("fn-1", 1, "Orphan A.")])
    doc_b = Document(elements=[_make_footnote("fn-2", 2, "Orphan B.")])
    result = describe_orphan_footnotes_batch([doc_a, doc_b])
    assert len(result) == 2
    assert result[0][0] is doc_a
    assert result[1][0] is doc_b
