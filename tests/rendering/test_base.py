"""Tests for rendering base classes (rendering/base.py)."""
import pytest

from ragdoc.document import Document, ExternalRef
from ragdoc.rendering.base import ExternalRefProvider, RenderContext


# --- TestRenderContext ---


def test_render_context_get_external_with_dict():
    """RenderContext.get_external works with dict provider."""
    parent_doc = Document(title="Parent")
    child_doc = Document()

    ctx = RenderContext(
        document=child_doc,
        external_refs={parent_doc.id: parent_doc},
    )

    assert ctx.get_external(parent_doc.id) == parent_doc
    assert ctx.get_external("nonexistent") is None


def test_render_context_get_siblings_from_parent():
    """RenderContext.get_siblings computes siblings from parent's children."""
    sibling1 = Document(title="Sibling 1")
    sibling2 = Document(title="Sibling 2")

    parent = Document(
        title="Parent",
        external_refs=[
            ExternalRef(target_id=sibling1.id, rel_type="external-child"),
            ExternalRef(target_id=sibling2.id, rel_type="external-child"),
        ]
    )

    current = Document(
        title="Current",
        external_refs=[ExternalRef(target_id=parent.id, rel_type="external-parent")],
    )

    ctx = RenderContext(
        document=current,
        external_refs={
            parent.id: parent,
            sibling1.id: sibling1,
            sibling2.id: sibling2,
        },
    )

    siblings = ctx.get_siblings(current)

    assert len(siblings) == 2
    assert sibling1 in siblings
    assert sibling2 in siblings
    assert current not in siblings
