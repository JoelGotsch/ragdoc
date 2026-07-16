"""Tests for InlineRef and ExternalRef relationship models."""

import pytest
from pydantic import ValidationError

from ragdoc.document import Document, ExternalRef, Heading, InlineRef, Paragraph

# --- TestInlineRef ---


@pytest.mark.parametrize("rel_type", ["image", "footnote", "table", "figure"])
def test_inline_ref_valid_rel_types(rel_type: str):
    """InlineRef accepts valid relationship types for inline content."""
    ref = InlineRef(target_id="test-id", rel_type=rel_type)
    assert ref.rel_type == rel_type


# --- TestBaseElementInlineRefs ---


def test_inline_refs_derived_from_html():
    """inline_refs is parsed from <ref id='...' rel='...'/> tags in HTML."""

    para = Paragraph(html="<p>See <ref id='img-1' rel='image'/> for details.</p>")
    assert len(para.inline_refs) == 1
    assert para.inline_refs[0].target_id == "img-1"
    assert para.inline_refs[0].rel_type == "image"


def test_image_ids_from_html_refs():
    """image_ids returns IDs from <ref rel='image'/> tags in HTML."""

    para = Paragraph(
        html="<p>See <ref id='img-1' rel='image'/> and <ref id='fn-1' rel='footnote'/> and <ref id='img-2' rel='image'/></p>"
    )
    assert para.image_ids == ["img-1", "img-2"]


def test_footnote_ids_from_html_refs():
    """footnote_ids returns IDs from <ref rel='footnote'/> tags in HTML."""

    para = Paragraph(
        html="<p>Important point<ref id='fn-1' rel='footnote'/> and more<ref id='fn-2' rel='footnote'/></p>"
    )
    assert para.footnote_ids == ["fn-1", "fn-2"]


def test_empty_inline_refs_by_default():
    """Elements with no <ref> tags have empty inline_refs."""

    para = Paragraph(html="<p>No refs here.</p>")
    assert para.inline_refs == []
    assert para.image_ids == []
    assert para.footnote_ids == []


def test_inline_refs_roundtrip_via_html():
    """inline_refs are preserved through serialization because the HTML is stored."""

    para = Paragraph(html="<p>Text <ref id='img-1' rel='image'/></p>")
    data = para.model_dump()
    restored = Paragraph.model_validate(data)
    assert len(restored.inline_refs) == 1
    assert restored.inline_refs[0].target_id == "img-1"


def test_ref_tag_without_rel_is_ignored():
    """<ref> tags without a rel attribute are not included in inline_refs."""

    para = Paragraph(html="<p>Old-style <ref id='img-1'/> ref.</p>")
    assert para.inline_refs == []


def test_multiple_ref_types_in_html():
    """inline_refs returns all <ref> tags regardless of rel_type."""

    para = Paragraph(
        html="<p><ref id='img-1' rel='image'/><ref id='fn-1' rel='footnote'/><ref id='t-1' rel='table'/></p>"
    )
    assert len(para.inline_refs) == 3
    rel_types = {r.rel_type for r in para.inline_refs}
    assert rel_types == {"image", "footnote", "table"}


# --- TestExternalRef ---


@pytest.mark.parametrize(
    "rel_type",
    ["external-parent", "external-child", "external-cites", "external-related"],
)
def test_external_ref_valid_rel_types(rel_type: str):
    """ExternalRef accepts valid relationship types for external references."""
    ref = ExternalRef(target_id="test-id", rel_type=rel_type)
    assert ref.rel_type == rel_type


def test_external_ref_rejects_unprefixed_rel_type():
    """ExternalRef rejects rel_type values not starting with 'external-'."""

    with pytest.raises(ValidationError):
        ExternalRef(target_id="x", rel_type="parent")


# --- TestDocumentExternalRefs ---


def test_document_with_external_refs():
    """Document can have external_refs."""

    doc = Document(
        elements=[Paragraph(html="<p>Content</p>")],
        external_refs=[
            ExternalRef(target_id="parent-doc", rel_type="external-parent"),
            ExternalRef(target_id="child-doc", rel_type="external-child"),
        ],
    )
    assert len(doc.external_refs) == 2
    assert doc.external_refs[0].target_id == "parent-doc"


def test_parent_ids_from_external_refs():
    """parent_ids property returns IDs of external-parent refs."""

    doc = Document(
        elements=[Paragraph(html="<p>Content</p>")],
        external_refs=[
            ExternalRef(target_id="parent-1", rel_type="external-parent"),
            ExternalRef(target_id="child-1", rel_type="external-child"),
            ExternalRef(target_id="parent-2", rel_type="external-parent"),
        ],
    )
    assert doc.parent_ids == ["parent-1", "parent-2"]


def test_child_ids_from_external_refs():
    """child_ids property returns IDs of external-child refs."""

    doc = Document(
        elements=[Paragraph(html="<p>Content</p>")],
        external_refs=[
            ExternalRef(target_id="child-1", rel_type="external-child"),
            ExternalRef(target_id="parent-1", rel_type="external-parent"),
            ExternalRef(target_id="child-2", rel_type="external-child"),
        ],
    )
    assert doc.child_ids == ["child-1", "child-2"]


def test_external_refs_serialization_roundtrip():
    """external_refs survive serialization roundtrip."""

    doc = Document(
        elements=[Paragraph(html="<p>Test</p>")],
        external_refs=[ExternalRef(target_id="parent-doc", rel_type="external-parent")],
    )
    data = doc.model_dump()
    restored = Document.model_validate(data)
    assert len(restored.external_refs) == 1
    assert restored.external_refs[0].target_id == "parent-doc"


def test_base_element_does_not_have_external_refs():
    """BaseElement (and subclasses) should NOT have external_refs attribute."""

    para = Paragraph(html="<p>Test</p>")
    heading = Heading(html="<h1>Title</h1>")

    assert not hasattr(para, "external_refs") or "external_refs" not in para.model_fields
    assert not hasattr(heading, "external_refs") or "external_refs" not in heading.model_fields


def test_bidirectional_parent_child_relationship():
    """Parent/child relationships can be established bidirectionally."""

    parent = Document(
        title="Parent",
        external_refs=[
            ExternalRef(target_id="child-1", rel_type="external-child"),
            ExternalRef(target_id="child-2", rel_type="external-child"),
        ],
    )

    child = Document(
        title="Child 1",
        external_refs=[
            ExternalRef(target_id=parent.id, rel_type="external-parent"),
        ],
    )

    assert "child-1" in parent.child_ids
    assert parent.id in child.parent_ids


# --- TestDocumentParserProvenance ---


def test_parser_field_serialization_roundtrip():
    """Parser field survives serialization roundtrip."""

    doc = Document(
        elements=[Paragraph(html="<p>Test</p>")],
        parser="azure_di",
    )
    data = doc.model_dump()
    restored = Document.model_validate(data)
    assert restored.parser == "azure_di"


# --- TestDocumentGetElement ---


def test_get_element_by_id():
    """Can retrieve element by ID."""

    para = Paragraph(html="<p>Test paragraph</p>")
    heading = Heading(html="<h1>Title</h1>")

    doc = Document(elements=[heading, para])

    assert doc.get_element(para.id) == para
    assert doc.get_element(heading.id) == heading


def test_get_element_returns_none_for_missing():
    """Returns None when element ID not found."""

    doc = Document(elements=[Paragraph(html="<p>Test</p>")])

    assert doc.get_element("nonexistent-id") is None


def test_get_element_empty_document():
    """Returns None for empty document."""

    doc = Document()
    assert doc.get_element("any-id") is None


def test_get_element_finds_in_large_doc():
    """Can find element in document with many elements."""

    paras = [Paragraph(html=f"<p>Para {i}</p>") for i in range(100)]
    doc = Document(elements=paras)

    target_para = paras[50]
    assert doc.get_element(target_para.id) == target_para
