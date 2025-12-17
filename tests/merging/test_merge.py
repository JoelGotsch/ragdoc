"""Integration tests for Approach A: element-alignment merge."""
import pytest

from ragdoc.document import Document, ElementTypeEnum, Footnote, Heading, Image, Paragraph, RawText
from ragdoc.merging.merge import compute_patch, merge_documents
from ragdoc.merging.patch import PatchOperationType, validate_inline_refs


# --- TestMergeHeadingLevel ---


def test_heading_level_html_parser_wins_over_mineru():
    """html parser is in trust_parsers -> its heading level takes precedence."""
    doc_a = Document(
        elements=[Heading(html_content="<h3>Introduction</h3>")],
        parser="mineru",
    )
    doc_b = Document(
        elements=[Heading(html_content="<h2>Introduction</h2>")],
        parser="html",
    )
    result = merge_documents(doc_a, doc_b)
    headings = result.headings
    assert len(headings) == 1
    assert headings[0].level == 2  # html parser level wins


def test_heading_level_prefer_source_b_forces_heading_from_b():
    doc_a = Document(elements=[Heading(html_content="<h1>Title</h1>")])
    doc_b = Document(elements=[Heading(html_content="<h2>Title</h2>")])
    result = merge_documents(
        doc_a, doc_b, prefer_source={ElementTypeEnum.HEADING: "b"}
    )
    assert result.headings[0].level == 2


# --- TestMergeInlineMarkup ---


def test_inline_markup_em_paragraph_from_b_preserved():
    doc_a = Document(elements=[Paragraph(html_content="<p>plain body text</p>")])
    doc_b = Document(
        elements=[Paragraph(html_content="<p><em>italic</em> body text</p>")]
    )
    result = merge_documents(doc_a, doc_b)
    assert len(result.paragraphs) == 1
    assert "<em>" in result.paragraphs[0].html


def test_inline_markup_math_formula_from_b_preferred():
    doc_a = Document(
        elements=[Paragraph(html_content="<p>E equals mc squared</p>")]
    )
    doc_b = Document(
        elements=[Paragraph(html_content="<p>E = <math>mc<sup>2</sup></math></p>")]
    )
    result = merge_documents(doc_a, doc_b)
    para_html = result.paragraphs[0].html
    assert "<math>" in para_html or "<sup>" in para_html


# --- TestMergeFootnoteCarried ---


def test_footnote_element_in_output_when_referenced():
    """If doc_b has a footnote referenced in a paragraph, it ends up in output."""
    fn = Footnote(number=1, innerhtml="Detailed footnote text.")
    para = Paragraph(
        html_content=f'<p>Body text<ref id="{fn.id}" rel="footnote"/>.</p>'
    )
    doc_a = Document(elements=[Paragraph(html_content="<p>Body text.</p>")])
    doc_b = Document(elements=[para, fn], parser="html")

    result = merge_documents(doc_a, doc_b)
    assert validate_inline_refs(result) == []


# --- TestMergeImageInlineRef ---


def test_image_inline_ref_intact():
    """Image element and its inline ref survive the merge."""
    img = Image(image=None, image_type="png", alt="Chart")
    para = Paragraph(
        html_content=f'<p>See <ref id="{img.id}" rel="image"/> for details.</p>'
    )
    doc_a = Document(elements=[Paragraph(html_content="<p>See diagram for details.</p>")])
    doc_b = Document(elements=[para, img], parser="html")

    result = merge_documents(doc_a, doc_b)
    assert validate_inline_refs(result) == []


# --- TestMergePerTypeControl ---


def test_per_type_allow_insertions_false_no_extra_elements():
    """Extra elements from doc_b are suppressed."""
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared</p>"),
            Paragraph(html_content="<p>Extra only in B</p>"),
        ]
    )
    result = merge_documents(doc_a, doc_b, allow_insertions_from_b=False)
    # Only the shared paragraph should be present (extra suppressed)
    assert len(result.paragraphs) == 1


def test_per_type_prefer_source_a_for_all_headings():
    """prefer_source forces all headings to come from doc_a."""
    doc_a = Document(
        elements=[
            Heading(html_content="<h1>Main Title</h1>"),
            Paragraph(html_content="<p>Content</p>"),
        ]
    )
    doc_b = Document(
        elements=[
            Heading(html_content="<h2>Main Title</h2>"),
            Paragraph(html_content="<p><em>Content</em></p>"),
        ]
    )
    result = merge_documents(
        doc_a, doc_b, prefer_source={ElementTypeEnum.HEADING: "a"}
    )
    # Heading from doc_a (h1)
    assert result.headings[0].level == 1
    # Paragraph still gets richer markup from doc_b
    assert "<em>" in result.paragraphs[0].html


# --- TestComputePatch ---


def test_compute_patch_returns_document_patch():
    doc_a = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    patch = compute_patch(doc_a, doc_b)
    assert len(patch.operations) > 0


def test_compute_patch_source_parsers_set():
    doc_a = Document(elements=[], parser="mineru")
    doc_b = Document(elements=[], parser="azure_di")
    patch = compute_patch(doc_a, doc_b)
    assert patch.source_parser_a == "mineru"
    assert patch.source_parser_b == "azure_di"


def test_compute_patch_manual_override_honored():
    p_custom = Paragraph(html_content="<p>custom override</p>")
    doc_a = Document(elements=[Paragraph(html_content="<p>From A</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>From B</p>")])
    patch = compute_patch(doc_a, doc_b)
    # Override the first operation's resolved_elements
    patch.operations[0].resolved_elements = [p_custom]
    result = patch.apply()
    assert result.paragraphs[0].id == p_custom.id


def test_compute_patch_merged_document_parser_is_merged():
    doc_a = Document(elements=[Paragraph(html_content="<p>x</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>x</p>")])
    result = merge_documents(doc_a, doc_b)
    assert result.parser == "merged"


def test_compute_patch_serializable():
    """DocumentPatch is JSON-serializable."""
    doc_a = Document(elements=[Paragraph(html_content="<p>text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>text</p>")])
    patch = compute_patch(doc_a, doc_b)
    json_str = patch.model_dump_json()
    assert "operations" in json_str


def test_compute_patch_delete_a_operation_recorded():
    doc_a = Document(
        elements=[
            Paragraph(html_content="<p>Only in A — unique content here</p>"),
            Paragraph(html_content="<p>Shared content</p>"),
        ]
    )
    doc_b = Document(elements=[Paragraph(html_content="<p>Shared content</p>")])
    patch = compute_patch(doc_a, doc_b)
    op_types = {op.op for op in patch.operations}
    assert PatchOperationType.DELETE_A in op_types


def test_compute_patch_output_is_valid_inline_refs():
    doc_a = Document(elements=[Paragraph(html_content="<p>Body text.</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p><em>Body</em> text.</p>")])
    result = merge_documents(doc_a, doc_b)
    assert validate_inline_refs(result) == []


# --- TestMergeRealWorldScenario ---

_INSERT_PARAMS = pytest.mark.parametrize(
    "allow_insertions_from_b, extra_para_present",
    [
        (True, True),
        (False, False),
        (frozenset({ElementTypeEnum.PARAGRAPH}), True),    # Paragraph allowed -> included
        (frozenset({ElementTypeEnum.TABLE}), False),       # Paragraph not in set -> blocked
    ],
    ids=["allow_all", "allow_none", "allow_paragraph", "allow_table_only"],
)


@_INSERT_PARAMS
def test_real_world_structured_doc_wins_and_inline_refs_intact(
    real_world_docs, allow_insertions_from_b, extra_para_present
):
    doc_a, doc_b, fn, img, para1, para2, para_extra = real_world_docs
    result = merge_documents(doc_a, doc_b, allow_insertions_from_b=allow_insertions_from_b)

    # Inline refs must be clean
    assert validate_inline_refs(result) == []

    # doc_b's Footnote element is preserved (it trails an aligned element)
    assert len(result.footnotes) == 1
    assert result.footnotes[0].innerhtml == fn.innerhtml

    # doc_b's Image is preserved (it trails an aligned element)
    assert len(result.images) == 1
    assert result.images[0].alt == img.alt

    # First element: doc_b's Paragraph with doc_a's <strong> injected
    first = result.elements[0]
    assert first.element_type == ElementTypeEnum.PARAGRAPH
    assert first.footnote_ids == [fn.id]
    assert "<strong>text</strong>" in first.html_content

    # Second element: doc_a's unique "Some additional text" preserved
    assert result.elements[1].element_type == ElementTypeEnum.RAW_TEXT
    assert result.elements[1].innerhtml == "Some additional text"

    # Third element: doc_b's "Some text in the middle" paragraph
    assert result.elements[2].element_type == ElementTypeEnum.PARAGRAPH
    assert result.elements[2].html_content == para2.html_content

    # Fourth and fifth: Image and Footnote from doc_b (trailing para2)
    assert result.elements[3].element_type == ElementTypeEnum.IMAGE
    assert result.elements[4].element_type == ElementTypeEnum.FOOTNOTE

    # Extra doc_b-only Paragraph: present iff insertions for its type are allowed
    extra_in_output = any("only in doc_b" in e.text for e in result.elements)
    assert extra_in_output == extra_para_present


def test_real_world_image_and_footnote_from_b_survive_merge():
    """Even when doc_a has no counterparts, Image and Footnote from doc_b are inserted."""
    fn = Footnote(number=1, innerhtml="This is the footnote text")
    img = Image(image=None, image_type="png", alt="Figure 1")
    para1 = Paragraph(
        html_content=f'<p>Body text<ref id="{fn.id}" rel="footnote"/>.</p>'
    )
    doc_a = Document(
        elements=[RawText(innerhtml="Body text1.")],
        parser="mineru",
    )
    doc_b = Document(elements=[para1, img, fn], parser="html")

    result = merge_documents(doc_a, doc_b)

    assert validate_inline_refs(result) == []
    assert len(result.footnotes) >= 1
    assert len(result.images) >= 1


# --- TestMergeParity ---


def test_parity_returns_document():
    doc_a = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    result = merge_documents(doc_a, doc_b)
    assert isinstance(result, Document)


def test_parity_empty_docs_return_empty_document():
    result = merge_documents(Document(), Document())
    assert result.elements == []


def test_parity_richer_paragraph_from_a_wins():
    """Richness selection is symmetric: A can win over B."""
    doc_a = Document(
        elements=[Paragraph(html_content="<p><b>bold</b> text here</p>")]
    )
    doc_b = Document(elements=[Paragraph(html_content="<p>plain text here</p>")])
    result = merge_documents(doc_a, doc_b)
    assert "<b>" in result.paragraphs[0].html


def test_parity_metadata_records_source_parsers():
    """merge_documents result carries source parser metadata on the document."""
    doc_a = Document(elements=[], parser="mineru")
    doc_b = Document(elements=[], parser="azure_di")
    result = merge_documents(doc_a, doc_b)
    assert result.metadata.get("source_parser_a") == "mineru"
    assert result.metadata.get("source_parser_b") == "azure_di"


def test_parity_insertions_enabled_includes_extra_b_elements():
    """Explicit positive case: allow_insertions_from_b=True includes B-only elements."""
    doc_a = Document(elements=[Paragraph(html_content="<p>Shared content</p>")])
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared content</p>"),
            Paragraph(html_content="<p>Extra only in B</p>"),
        ]
    )
    result = merge_documents(doc_a, doc_b, allow_insertions_from_b=True)
    assert len(result.paragraphs) >= 2
