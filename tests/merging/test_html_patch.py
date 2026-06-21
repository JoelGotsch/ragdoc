"""Tests for Phase 5: DocumentHTMLPatch inspectable layer."""

from ragdoc.document import Document, Paragraph
from ragdoc.merging.html_patch import DocumentHtmlPatch, HtmlMergeOperation, compute_html_patch

# --- TestHtmlPatch ---


def test_html_patch_compute_returns_patch():
    doc_a = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    assert isinstance(patch, DocumentHtmlPatch)


def test_html_patch_has_operations():
    doc_a = Document(elements=[Paragraph(html_content="<p>text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>text</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    assert isinstance(patch.operations, list)
    assert len(patch.operations) > 0


def test_html_patch_operations_are_html_merge_operations():
    doc_a = Document(elements=[Paragraph(html_content="<p>content</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>content</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    for op in patch.operations:
        assert isinstance(op, HtmlMergeOperation)


def test_html_patch_source_parsers_set():
    doc_a = Document(elements=[], parser="mineru")
    doc_b = Document(elements=[], parser="azure_di")
    patch = compute_html_patch(doc_a, doc_b)
    assert patch.source_parser_a == "mineru"
    assert patch.source_parser_b == "azure_di"


def test_html_patch_serializable():
    """DocumentHtmlPatch is JSON-serializable."""
    doc_a = Document(elements=[Paragraph(html_content="<p>text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>text</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    json_str = patch.model_dump_json()
    assert "operations" in json_str


def test_html_patch_apply_returns_document():
    doc_a = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>Hello</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    result = patch.apply()
    assert isinstance(result, Document)


def test_html_patch_apply_sets_parser_merged():
    doc_a = Document(elements=[Paragraph(html_content="<p>text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>text</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    result = patch.apply()
    assert result.parser == "merged"


def test_html_patch_apply_sets_metadata():
    doc_a = Document(elements=[], parser="mineru")
    doc_b = Document(elements=[], parser="html")
    patch = compute_html_patch(doc_a, doc_b)
    result = patch.apply()
    assert result.metadata.get("source_parser_a") == "mineru"
    assert result.metadata.get("source_parser_b") == "html"


def test_html_patch_manual_override_honored():
    """Caller can replace op.selected before apply(); result reflects the override."""
    doc_a = Document(elements=[Paragraph(html_content="<p>From A</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>From B</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    for op in patch.operations:
        op.selected = ["<p>Custom override</p>"]
    result = patch.apply()
    assert any("Custom override" in e.text for e in result.elements)


def test_html_patch_empty_selected_produces_empty_document():
    """Clearing all selected produces an empty Document."""
    doc_a = Document(elements=[Paragraph(html_content="<p>text</p>")])
    doc_b = Document(elements=[Paragraph(html_content="<p>text</p>")])
    patch = compute_html_patch(doc_a, doc_b)
    for op in patch.operations:
        op.selected = []
    result = patch.apply()
    assert result.elements == []


def test_html_patch_merge_documents_html_equivalent_to_patch_apply():
    """merge_documents_html and compute_html_patch(...).apply() produce equivalent docs."""
    from ragdoc.merging.html_merge import merge_documents_html

    doc_a = Document(
        elements=[
            Paragraph(html_content="<p>Shared paragraph</p>"),
            Paragraph(html_content="<p>Only in A</p>"),
        ]
    )
    doc_b = Document(
        elements=[
            Paragraph(html_content="<p>Shared paragraph</p>"),
            Paragraph(html_content="<p>Only in B</p>"),
        ]
    )
    direct = merge_documents_html(doc_a, doc_b)
    via_patch = compute_html_patch(doc_a, doc_b).apply()

    assert [e.text for e in direct.elements] == [e.text for e in via_patch.elements]
