"""Tests for QdrantIndex annotation and register_indexes_from_type helper.

Covers §3.2 from PLAN_TYPED_METADATA_EXTENSIONS.md.
All tests stub AsyncQdrantClient — no live Qdrant instance required.
"""
from __future__ import annotations

from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("qdrant_client", reason="qdrant extra not installed")

from ragdoc.integrations.vector_stores.qdrant import (
    QdrantIndex,
    QdrantVectorStore,
    register_indexes_from_type,
)
from ragdoc.metadata import BaseMetadata

try:
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    pytest.skip("qdrant-client not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Module-level TypedDict subclasses (locally-scoped classes don't work with
# get_type_hints when from __future__ import annotations is active)
# ---------------------------------------------------------------------------


class _SingleFieldMeta(BaseMetadata, total=False):
    doc_name: Annotated[str, QdrantIndex(models.PayloadSchemaType.KEYWORD)]


class _TwoFieldMeta(BaseMetadata, total=False):
    doc_name: Annotated[str, QdrantIndex(models.PayloadSchemaType.KEYWORD)]
    doc_date: Annotated[str | None, QdrantIndex(models.PayloadSchemaType.DATETIME)]


class _MixedMeta(BaseMetadata, total=False):
    plain: str
    tagged: Annotated[str, QdrantIndex(models.PayloadSchemaType.KEYWORD)]


class _PlainMeta(BaseMetadata, total=False):
    plain: str
    other: int | None


class _BaseMeta(BaseMetadata, total=False):
    base_field: Annotated[str, QdrantIndex(models.PayloadSchemaType.KEYWORD)]


class _ChildMeta(_BaseMeta, total=False):
    child_field: Annotated[str, QdrantIndex(models.PayloadSchemaType.INTEGER)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client() -> MagicMock:
    c = MagicMock()
    c.create_collection = AsyncMock()
    c.create_payload_index = AsyncMock()
    c.scroll = AsyncMock(return_value=([], None))
    c.upsert = AsyncMock()
    c.delete = AsyncMock()
    return c


def _conflict_client() -> MagicMock:
    """Client whose create_payload_index always raises 409."""
    c = make_client()
    c.create_payload_index = AsyncMock(
        side_effect=UnexpectedResponse(
            status_code=409, reason_phrase="Conflict", content=b"", headers={}
        )
    )
    return c


def _error_client(status: int) -> MagicMock:
    """Client whose create_payload_index raises a given HTTP error."""
    c = make_client()
    c.create_payload_index = AsyncMock(
        side_effect=UnexpectedResponse(
            status_code=status,
            reason_phrase="Error",
            content=b"",
            headers={},
        )
    )
    return c


# ---------------------------------------------------------------------------
# QdrantIndex
# ---------------------------------------------------------------------------


def test_qdrantindex_is_frozen():
    import dataclasses

    assert dataclasses.is_dataclass(QdrantIndex)
    assert QdrantIndex.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_qdrantindex_and_register_exported_from_vector_stores():
    from ragdoc.integrations.vector_stores import QdrantIndex as QI
    from ragdoc.integrations.vector_stores import register_indexes_from_type as r

    assert QI is QdrantIndex
    assert r is register_indexes_from_type


# ---------------------------------------------------------------------------
# register_indexes_from_type
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_register_indexes_calls_create_for_annotated_fields():
    client = make_client()
    await register_indexes_from_type(_TwoFieldMeta, client, "col")

    assert client.create_payload_index.call_count == 2
    field_names = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
    assert field_names == {"doc_name", "doc_date"}


@pytest.mark.anyio
async def test_register_indexes_skips_unannotated_fields():
    client = make_client()
    await register_indexes_from_type(_MixedMeta, client, "col")

    assert client.create_payload_index.call_count == 1
    assert client.create_payload_index.call_args.kwargs["field_name"] == "tagged"


@pytest.mark.anyio
async def test_register_indexes_handles_typeddict_inheritance():
    client = make_client()
    await register_indexes_from_type(_ChildMeta, client, "col")

    field_names = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
    assert "base_field" in field_names
    assert "child_field" in field_names


@pytest.mark.anyio
async def test_register_indexes_ignores_409_conflict():
    await register_indexes_from_type(_SingleFieldMeta, _conflict_client(), "col")
    # No exception raised — test passes by not raising.


@pytest.mark.anyio
async def test_register_indexes_reraises_non_409_error():
    with pytest.raises(UnexpectedResponse):
        await register_indexes_from_type(_SingleFieldMeta, _error_client(500), "col")


@pytest.mark.anyio
async def test_register_indexes_noop_when_no_annotated_fields():
    client = make_client()
    await register_indexes_from_type(_PlainMeta, client, "col")
    client.create_payload_index.assert_not_called()


# ---------------------------------------------------------------------------
# QdrantVectorStore.create with metadata_type
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_registers_metadata_indexes_when_type_provided():
    client = make_client()
    await QdrantVectorStore.create(client, "col", vector_size=128, metadata_type=_SingleFieldMeta)

    field_names = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
    assert "doc_name" in field_names
    assert "source_id" in field_names  # always created


@pytest.mark.anyio
async def test_create_skips_metadata_registration_when_type_is_none():
    client = make_client()
    await QdrantVectorStore.create(client, "col", vector_size=128)

    assert client.create_payload_index.call_count == 1
    assert client.create_payload_index.call_args.kwargs["field_name"] == "source_id"
