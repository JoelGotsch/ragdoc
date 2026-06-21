"""Integration tests for Approach B: render-merge-reparse."""

import base64

import pytest

from ragdoc.document import Document, ElementTypeEnum, Footnote, Heading, Image, Paragraph
from ragdoc.merging.html_merge import merge_documents_html
from ragdoc.merging.patch import validate_inline_refs

# Mark shared with test_merge.py via conftest.py
# Columns: allow_insertions_from_b, extra_para_present, footnote_present
# Footnotes render as <aside> (ElementTypeEnum.FOOTNOTE), so frozenset({PARAGRAPH})
# allows extra_para but NOT the footnote; frozenset({FOOTNOTE}) would allow both.
_INSERT_PARAMS = pytest.mark.parametrize(
    "allow_insertions_from_b, extra_para_present, footnote_present",
    [
        (True, True, True),
        (False, False, False),
        (frozenset({ElementTypeEnum.PARAGRAPH}), True, False),
        (frozenset({ElementTypeEnum.TABLE}), False, False),
    ],
    ids=["allow_all", "allow_none", "allow_paragraph", "allow_table_only"],
)


# --- TestHtmlMergeBasic ---


def test_html_merge_basic_returns_document():
    doc_a = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    result = merge_documents_html(doc_a, doc_b)
    assert isinstance(result, Document)


def test_html_merge_basic_parser_is_merged():
    doc_a = Document(elements=[Paragraph(html_content="<p>text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>text</p>")])
    result = merge_documents_html(doc_a, doc_b)
    assert result.parser == "merged"


def test_html_merge_basic_metadata_records_source_parsers():
    doc_a = Document(elements=[], parser="mineru")
    doc_b = Document(elements=[], parser="azure_di")
    result = merge_documents_html(doc_a, doc_b)
    assert result.metadata.get("source_parser_a") == "mineru"
    assert result.metadata.get("source_parser_b") == "azure_di"


def test_html_merge_basic_empty_docs_return_empty_document():
    result = merge_documents_html(Document(), Document())
    assert result.elements == []


# --- TestHtmlMergeRichness ---


def test_html_merge_richness_richer_paragraph_from_b_wins():
    doc_a = Document(elements=[Paragraph(html_content="<p>plain body text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p><em>italic</em> body text</p>")])
    result = merge_documents_html(doc_a, doc_b)
    assert len(result.paragraphs) >= 1
    para_html = result.paragraphs[0].html
    assert "<em>" in para_html


def test_html_merge_richness_richer_paragraph_from_a_wins():
    doc_a = Document(elements=[Paragraph(html_content="<p><b>bold</b> text here</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>plain text here</p>")])
    result = merge_documents_html(doc_a, doc_b)
    assert "<b>" in result.paragraphs[0].html


# --- TestHtmlMergePreferSource ---


def test_html_merge_prefer_source_b_for_headings():
    doc_a = Document(
        elements=[
            Heading(html_content="<h3>Section</h3>"),
            Paragraph(html_content="<p>Content</p>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h2>Section</h2>"),
            Paragraph(html_content="<p>Content</p>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b, prefer_source={ElementTypeEnum.HEADING: "b"})
    assert len(result.headings) >= 1
    assert result.headings[0].level == 2  # from doc_b


def test_html_merge_prefer_source_a_for_headings():
    doc_a = Document(
        elements=[
            Heading(html_content="<h1>Title</h1>"),
            Paragraph(html_content="<p>Body</p>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h2>Title</h2>"),
            Paragraph(html_content="<p><em>Body</em></p>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b, prefer_source={ElementTypeEnum.HEADING: "a"})
    # Heading from doc_a (h1)
    assert result.headings[0].level == 1
    # Paragraph still gets <em> from doc_b (no prefer_source for paragraphs)
    assert "<em>" in result.paragraphs[0].html


# --- TestHtmlMergeInsertions ---


def test_html_merge_insertions_disabled_suppresses_extra_b_elements():
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared content</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared content</p>"),
            Paragraph(html_content="<p>Extra only in B</p>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=False)
    assert len(result.paragraphs) == 1


def test_html_merge_insertions_enabled_includes_extra_b_elements():
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared content</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared content</p>"),
            Paragraph(html_content="<p>Extra only in B</p>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=True)
    assert len(result.paragraphs) >= 2


# --- TestHtmlMergeFootnoteNote ---


def test_html_merge_footnote_structure_reconstructed_after_reparse():
    """Footnote structure is preserved: Footnote.html renders as <aside> which is parsed directly."""
    fn = Footnote(number=1, innerhtml="Important note text.")
    para = Paragraph(html_content=f'<p>Body text<ref id="{fn.id}" rel="footnote"/>.</p>')
    doc_a = Document(elements=[Paragraph(html_content="<p>Body text.</p>")])
    doc_b = Document(elements=[para, fn], parser="html")
    result = merge_documents_html(doc_a, doc_b)
    assert len(result.footnotes) == 1
    assert result.footnotes[0].text == "Important note text."
    assert validate_inline_refs(result) == []


def test_html_merge_footnote_inline_ref_intact_after_html_merge():
    """After merging, validate_inline_refs returns [] — inline refs are reconstructed."""
    fn = Footnote(number=1, innerhtml="Detailed footnote content.")
    para = Paragraph(html_content=f'<p>Sentence with note<ref id="{fn.id}" rel="footnote"/>.</p>')
    doc_a = Document(elements=[Paragraph(html_content="<p>Sentence with note.</p>")])
    doc_b = Document(elements=[para, fn], parser="html")
    result = merge_documents_html(doc_a, doc_b)
    assert validate_inline_refs(result) == []


# --- TestHtmlMergeRealWorldScenario ---


@_INSERT_PARAMS
def test_html_merge_real_world_scenario(real_world_docs, allow_insertions_from_b, extra_para_present, footnote_present):
    doc_a, doc_b, fn, _img, _para1, _para2, _para_extra = real_world_docs
    result = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=allow_insertions_from_b)

    assert result.parser == "merged"

    # Image is still lost in Approach B (img tag not a top-level content tag)
    assert len(result.images) == 0

    # Footnote renders as <aside> (FOOTNOTE type). Its presence is gated
    # by whether FOOTNOTE is allowed; it is independent of PARAGRAPH insertions.
    if footnote_present:
        assert len(result.footnotes) >= 1
        assert result.footnotes[0].innerhtml == fn.innerhtml
    else:
        assert len(result.footnotes) == 0

    full_text = " ".join(e.text for e in result.elements)

    # doc_a's <strong> paragraph wins (higher markup richness in rendered HTML)
    assert any("<strong>" in e.html for e in result.elements)

    # "Some additional text" from doc_a is preserved (no doc_b counterpart —
    # in Approach B doc_a wins the replace block, keeping both its elements)
    assert "Some additional text" in full_text

    # Extra doc_b-only paragraph: presence follows allow_insertions_from_b
    extra_in_output = any("only in doc_b" in e.text for e in result.elements)
    assert extra_in_output == extra_para_present


# --- TestHtmlMergeDocAPreservation ---


def test_html_merge_doc_a_unique_element_preserved():
    """Element unique to doc_a with no counterpart in doc_b is kept."""
    doc_a = Document(elements=[Paragraph(html_content="<p>Unique to A only</p>")])
    doc_b = Document(elements=[])
    result = merge_documents_html(doc_a, doc_b)
    assert any("Unique to A" in e.text for e in result.elements)


def test_html_merge_doc_a_unique_element_preserved_alongside_shared():
    """Mix of shared + doc_a-unique elements: all appear in output."""
    doc_a = Document(
        elements=[
            Paragraph(html_content="<p>Shared content</p>"),
            Paragraph(html_content="<p>Background only in A</p>"),
            Paragraph(html_content="<p>Method description</p>"),
        ]
    )
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared content</p>"),
            Paragraph(html_content="<p>Method description</p>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b)
    texts = " ".join(e.text for e in result.elements)
    assert "Background only in A" in texts
    assert "Shared content" in texts
    assert "Method description" in texts


# --- TestHtmlMergeImagePreservation ---


def test_html_merge_image_with_base64_survives_merge():
    """Standalone Image with base64 data in doc_b appears in merged output."""
    data = base64.b64encode(b"fake-png-bytes").decode()
    img = Image(image=data, image_type="png", alt="Chart")
    doc_a = Document(elements=[Paragraph(html_content="<p>Some text</p>")])
    doc_b = Document(elements=[img])
    result = merge_documents_html(doc_a, doc_b)
    assert len(result.images) >= 1
    assert result.images[0].image == data


def test_html_merge_image_without_data_gracefully_lost():
    """Image(image=None) produces no error; image is gracefully absent from output."""
    img = Image(image=None, image_type="png", alt="No binary data")
    doc_a = Document(elements=[Paragraph(html_content="<p>Some text</p>")])
    doc_b = Document(elements=[img])
    result = merge_documents_html(doc_a, doc_b)
    assert len(result.images) == 0


def test_html_merge_image_from_doc_a_preserved():
    """Standalone Image from doc_a (no doc_b counterpart) is kept."""
    data = base64.b64encode(b"img-data-a").decode()
    img = Image(image=data, image_type="png", alt="Only in A")
    doc_a = Document(elements=[img])
    doc_b = Document(elements=[])
    result = merge_documents_html(doc_a, doc_b)
    assert len(result.images) >= 1


# --- TestHtmlMergeHeadingHierarchy ---


def test_html_merge_heading_hierarchy_parser_wins_heading_level():
    """doc_a all h1 (no hierarchy); doc_b has h1/h2/h3 -> merged uses doc_b levels."""
    doc_a = Document(
        elements=[
            Heading(html_content="<h1>Chapter 1</h1>"),
            Heading(html_content="<h1>Section 1.1</h1>"),
            Heading(html_content="<h1>Subsection 1.1.1</h1>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h1>Chapter 1</h1>"),
            Heading(html_content="<h2>Section 1.1</h2>"),
            Heading(html_content="<h3>Subsection 1.1.1</h3>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b)
    levels = [h.level for h in result.headings]
    assert levels == [1, 2, 3]


def test_html_merge_heading_hierarchy_parser_wins_heading_level_inverse():
    """doc_b all h2 (no hierarchy); doc_a has h1/h2/h3 -> merged uses doc_a levels."""
    doc_a = Document(
        elements=[
            Heading(html_content="<h1>Chapter 1</h1>"),
            Heading(html_content="<h2>Section 1.1</h2>"),
            Heading(html_content="<h3>Subsection 1.1.1</h3>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h2>Chapter 1</h2>"),
            Heading(html_content="<h2>Section 1.1</h2>"),
            Heading(html_content="<h2>Subsection 1.1.1</h2>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b)
    levels = [h.level for h in result.headings]
    assert levels == [1, 2, 3]


def test_html_merge_heading_hierarchy_both_present_richness_fallback():
    """Both docs have hierarchy -> richness fallback, not hierarchy override."""
    doc_a = Document(
        elements=[
            Heading(html_content="<h1>Title</h1>"),
            Heading(html_content="<h2>Sub</h2>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h1><strong>Title</strong></h1>"),
            Heading(html_content="<h2>Sub</h2>"),
        ]
    )
    result = merge_documents_html(doc_a, doc_b)
    # Both have hierarchy; richness picks doc_b's <strong> heading
    assert "<strong>" in result.headings[0].html
