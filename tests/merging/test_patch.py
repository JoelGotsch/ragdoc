"""Tests for merging/patch.py — DocumentPatch, PatchOperation, validate_inline_refs."""
import pytest

from ragdoc.document import Document, Footnote, Heading, Image, Paragraph
from ragdoc.merging.patch import (
    DocumentPatch,
    PatchOperation,
    PatchOperationType,
    validate_inline_refs,
)


def _make_patch(*ops: PatchOperation) -> DocumentPatch:
    return DocumentPatch(operations=list(ops))


# --- TestDocumentPatchApply ---


def test_patch_apply_keep_a_uses_elements_a():
    para = Paragraph(html_content="<p>Hello</p>")
    op = PatchOperation(
        op=PatchOperationType.KEEP_A,
        elements_a=[para],
        resolved_elements=[para],
    )
    doc = _make_patch(op).apply()
    assert len(doc.elements) == 1
    assert doc.elements[0].id == para.id


def test_patch_apply_keep_b_uses_elements_b():
    para = Paragraph(html_content="<p>From B</p>")
    op = PatchOperation(
        op=PatchOperationType.KEEP_B,
        elements_b=[para],
        resolved_elements=[para],
    )
    doc = _make_patch(op).apply()
    assert len(doc.elements) == 1
    assert doc.elements[0].id == para.id


def test_patch_apply_insert_b_adds_element():
    para = Paragraph(html_content="<p>New in B</p>")
    op = PatchOperation(
        op=PatchOperationType.INSERT_B,
        elements_b=[para],
        resolved_elements=[para],
    )
    doc = _make_patch(op).apply()
    assert len(doc.elements) == 1


def test_patch_apply_delete_a_drops_elements():
    para = Paragraph(html_content="<p>Only in A</p>")
    op = PatchOperation(
        op=PatchOperationType.DELETE_A,
        elements_a=[para],
        resolved_elements=[],
    )
    doc = _make_patch(op).apply()
    assert len(doc.elements) == 0


def test_patch_apply_merge_uses_resolved_elements():
    p_a = Paragraph(html_content="<p>plain</p>")
    p_b = Paragraph(html_content="<p><em>rich</em></p>")
    op = PatchOperation(
        op=PatchOperationType.MERGE,
        elements_a=[p_a],
        elements_b=[p_b],
        resolved_elements=[p_b],
    )
    doc = _make_patch(op).apply()
    assert doc.elements[0].id == p_b.id


def test_patch_apply_merge_empty_resolved_raises():
    p_a = Paragraph(html_content="<p>A</p>")
    p_b = Paragraph(html_content="<p>B</p>")
    op = PatchOperation(
        op=PatchOperationType.MERGE,
        elements_a=[p_a],
        elements_b=[p_b],
        resolved_elements=[],
    )
    with pytest.raises(ValueError, match="resolved_elements"):
        _make_patch(op).apply()


def test_patch_apply_multiple_ops_concatenates_in_order():
    h = Heading(html_content="<h1>Title</h1>")
    p = Paragraph(html_content="<p>Body</p>")
    op1 = PatchOperation(
        op=PatchOperationType.KEEP_A, elements_a=[h], resolved_elements=[h]
    )
    op2 = PatchOperation(
        op=PatchOperationType.KEEP_B, elements_b=[p], resolved_elements=[p]
    )
    doc = _make_patch(op1, op2).apply()
    assert len(doc.elements) == 2
    assert doc.elements[0].id == h.id
    assert doc.elements[1].id == p.id


def test_patch_apply_manual_override_honored():
    p_a = Paragraph(html_content="<p>plain</p>")
    p_b = Paragraph(html_content="<p><em>rich</em></p>")
    p_custom = Paragraph(html_content="<p>custom override</p>")
    op = PatchOperation(
        op=PatchOperationType.MERGE,
        elements_a=[p_a],
        elements_b=[p_b],
        resolved_elements=[p_b],
    )
    patch = _make_patch(op)
    # User overrides before apply
    patch.operations[0].resolved_elements = [p_custom]
    doc = patch.apply()
    assert doc.elements[0].id == p_custom.id


def test_patch_apply_deduplicates_by_id():
    """Same element ID appearing in two KEEP_A ops -> appears once in output."""
    para = Paragraph(html_content="<p>Shared</p>")
    op1 = PatchOperation(
        op=PatchOperationType.KEEP_A, elements_a=[para], resolved_elements=[para]
    )
    op2 = PatchOperation(
        op=PatchOperationType.KEEP_A, elements_a=[para], resolved_elements=[para]
    )
    doc = _make_patch(op1, op2).apply()
    assert len(doc.elements) == 1


def test_patch_apply_sets_parser_merged():
    op = PatchOperation(
        op=PatchOperationType.KEEP_A,
        elements_a=[Paragraph(html_content="<p>x</p>")],
        resolved_elements=[Paragraph(html_content="<p>x</p>")],
    )
    patch = DocumentPatch(
        source_parser_a="mineru",
        source_parser_b="azure_di",
        operations=[op],
    )
    doc = patch.apply()
    assert doc.parser == "merged"
    assert doc.metadata["source_parser_a"] == "mineru"
    assert doc.metadata["source_parser_b"] == "azure_di"


# --- TestValidateInlineRefs ---


def test_validate_inline_refs_clean_document_returns_empty():
    img = Image(image=None, image_type="png", alt="chart")
    para = Paragraph(html_content=f'<p>See <ref id="{img.id}" rel="image"/> here.</p>')
    doc = Document(elements=[para, img])
    assert validate_inline_refs(doc) == []


def test_validate_inline_refs_broken_ref_returns_id():
    para = Paragraph(html_content='<p>See <ref id="nonexistent-id" rel="image"/> here.</p>')
    doc = Document(elements=[para])
    broken = validate_inline_refs(doc)
    assert "nonexistent-id" in broken


def test_validate_inline_refs_no_refs_returns_empty():
    para = Paragraph(html_content="<p>Plain text, no refs.</p>")
    doc = Document(elements=[para])
    assert validate_inline_refs(doc) == []


def test_validate_inline_refs_multiple_broken_refs_all_returned():
    para = Paragraph(
        html_content='<p><ref id="bad-1" rel="image"/> and <ref id="bad-2" rel="footnote"/></p>'
    )
    doc = Document(elements=[para])
    broken = validate_inline_refs(doc)
    assert "bad-1" in broken
    assert "bad-2" in broken


def test_validate_inline_refs_footnote_ref_resolved():
    fn = Footnote(number=1, innerhtml="Footnote text.")
    para = Paragraph(html_content=f'<p>See<ref id="{fn.id}" rel="footnote"/>.</p>')
    doc = Document(elements=[para, fn])
    assert validate_inline_refs(doc) == []
