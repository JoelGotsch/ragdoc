"""Tests for BaseMetadata TypedDict and metadata_json_schema utility.

Covers §3.1 (BaseMetadata) and §3.3 (metadata_json_schema) from
PLAN_TYPED_METADATA_EXTENSIONS.md.
"""

from __future__ import annotations

from typing import Annotated, get_type_hints

from ragdoc.metadata import BaseMetadata, metadata_json_schema

# ---------------------------------------------------------------------------
# Module-level TypedDict subclasses (Pydantic TypeAdapter cannot resolve
# locally-scoped classes when from __future__ import annotations is active)
# ---------------------------------------------------------------------------


class _SubMeta(BaseMetadata, total=False):
    document_name: str | None
    document_date: str | None


class _Marker:
    """Unknown Annotated extra — should be silently ignored by Pydantic."""


class _AnnotatedMeta(BaseMetadata, total=False):
    tagged: Annotated[str, _Marker()]


# ---------------------------------------------------------------------------
# BaseMetadata
# ---------------------------------------------------------------------------


def test_basemetadata_has_filename_field():
    hints = get_type_hints(BaseMetadata, include_extras=True)
    assert "filename" in hints


def test_basemetadata_total_is_false():
    assert BaseMetadata.__total__ is False


def test_basemetadata_filename_is_required():
    from typing import get_origin

    from typing_extensions import Required, get_type_hints as ext_hints

    hints = ext_hints(BaseMetadata, include_extras=True)
    assert get_origin(hints["filename"]) is Required


def test_basemetadata_subclass_inherits_filename():
    hints = get_type_hints(_SubMeta)
    assert "filename" in hints
    assert "document_name" in hints


def test_basemetadata_exported_from_ragdoc():
    import ragdoc

    assert ragdoc.BaseMetadata is BaseMetadata


# ---------------------------------------------------------------------------
# metadata_json_schema
# ---------------------------------------------------------------------------


def test_metadata_json_schema_base_has_filename_property():
    schema = metadata_json_schema(BaseMetadata)
    assert "filename" in schema.get("properties", {})


def test_metadata_json_schema_subclass_includes_all_fields():
    schema = metadata_json_schema(_SubMeta)
    props = schema.get("properties", {})
    assert "filename" in props
    assert "document_name" in props
    assert "document_date" in props


def test_metadata_json_schema_annotated_extras_ignored():
    """Pydantic's TypeAdapter strips Annotated extras it doesn't recognise."""
    schema = metadata_json_schema(_AnnotatedMeta)
    assert "tagged" in schema.get("properties", {})


def test_metadata_json_schema_exported_from_ragdoc_metadata():
    import ragdoc.metadata

    assert ragdoc.metadata.metadata_json_schema is metadata_json_schema
