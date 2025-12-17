"""Unit tests for QdrantDocumentStore.

All tests stub AsyncQdrantClient with MagicMock/AsyncMock — no live Qdrant instance required.
Mirrors tests/integrations/test_qdrant_index_registration.py conventions.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("qdrant_client", reason="qdrant extra not installed")

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.integrations.document_stores.qdrant import (
    _NAMESPACE,
    DocumentTooLargeError,
    QdrantDocumentStore,
    _document_to_point,
    _point_id,
)
from ragdoc.pipeline.stores import DocumentStore, SourceState

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    pytest.skip("qdrant-client not installed", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_document(source_id: str, source_hash: str = "h", body: str = "Body.") -> Document:
    doc = Document(
        title=source_id,
        elements=[Heading(innerhtml=source_id, level=1), Paragraph(html_content=f"<p>{body}</p>")],  # type: ignore[call-arg]  # Heading innerhtml/level consumed by validator
    )
    doc.source_id = source_id
    doc.source_hash = source_hash
    return doc


def _scroll_point(doc: Document) -> MagicMock:
    """A scrolled point exposing the 3 hash keys (cheap-read shape)."""
    point = MagicMock()
    point.payload = {
        "source_id": doc.source_id,
        "source_hash": doc.source_hash,
        "content_hash": doc.content_hash(),
    }
    return point


def _retrieve_record(doc: Document) -> MagicMock:
    """A retrieved record carrying the full serialized document in payload."""
    record = MagicMock()
    record.payload = doc.model_dump(mode="json")
    return record


@pytest.fixture
def client() -> MagicMock:
    c = MagicMock()
    c.scroll = AsyncMock(return_value=([], None))
    c.upsert = AsyncMock()
    c.delete = AsyncMock()
    c.retrieve = AsyncMock(return_value=[])
    c.create_collection = AsyncMock()
    c.create_payload_index = AsyncMock()
    return c


@pytest.fixture
def store(client: MagicMock) -> QdrantDocumentStore:
    return QdrantDocumentStore(client, "docs")


# ---------------------------------------------------------------------------
# point construction
# ---------------------------------------------------------------------------


def test_point_id_is_deterministic_uuid5():
    assert _point_id("a.pdf") == str(uuid.uuid5(_NAMESPACE, "a.pdf"))
    assert _point_id("a.pdf") == _point_id("a.pdf")
    assert _point_id("a.pdf") != _point_id("b.pdf")


def test_document_to_point_payload_and_vector():
    doc = make_document("a.pdf", "hash-a", body="Distinctive.")
    point = _document_to_point(doc)
    assert str(point.id) == _point_id("a.pdf")
    assert point.vector == [0.0]  # dummy, never queried
    assert point.payload["source_id"] == "a.pdf"
    assert point.payload["source_hash"] == "hash-a"
    assert point.payload["content_hash"] == doc.content_hash()
    # full document is serialized into the payload
    assert "elements" in point.payload


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upsert_calls_client_and_returns_source_ids(store, client):
    result = await store.upsert([make_document("a.pdf"), make_document("b.pdf")])
    assert result == ["a.pdf", "b.pdf"]
    client.upsert.assert_called_once()


@pytest.mark.anyio
async def test_upsert_requires_source_id(store):
    doc = Document(elements=[Paragraph(html_content="<p>x</p>")])  # no source_id
    with pytest.raises(ValueError, match="source_id"):
        await store.upsert([doc])


@pytest.mark.anyio
async def test_upsert_oversized_payload_raises_document_too_large(client):
    too_big = UnexpectedResponse(status_code=413, reason_phrase="Payload Too Large", content=b"too big", headers=None)
    client.upsert = AsyncMock(side_effect=too_big)
    store = QdrantDocumentStore(client, "docs")

    with pytest.raises(DocumentTooLargeError) as exc:
        await store.upsert([make_document("big.pdf")])
    assert exc.value.source_id == "big.pdf"
    assert exc.value.size_bytes > 0


@pytest.mark.anyio
async def test_upsert_400_larger_than_allowed_raises_document_too_large(client):
    # Qdrant reports oversized payloads as a 400 with "larger than allowed" wording.
    rejected = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b"JSON payload (1234 bytes) is larger than allowed (limit 1024 bytes)",
        headers=None,
    )
    client.upsert = AsyncMock(side_effect=rejected)
    store = QdrantDocumentStore(client, "docs")
    with pytest.raises(DocumentTooLargeError) as exc:
        await store.upsert([make_document("big.pdf")])
    assert exc.value.source_id == "big.pdf"


@pytest.mark.anyio
async def test_upsert_400_unrelated_payload_error_propagates(client):
    # A 400 that merely mentions "payload" (a validation error, not a size error) must NOT be
    # misclassified as too-large — it propagates as the raw UnexpectedResponse.
    other = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b"payload index for field 'foo' does not exist",
        headers=None,
    )
    client.upsert = AsyncMock(side_effect=other)
    store = QdrantDocumentStore(client, "docs")
    with pytest.raises(UnexpectedResponse):
        await store.upsert([make_document("a.pdf")])


@pytest.mark.anyio
async def test_upsert_unrelated_error_propagates(client):
    other = UnexpectedResponse(status_code=500, reason_phrase="Server Error", content=b"boom", headers=None)
    client.upsert = AsyncMock(side_effect=other)
    store = QdrantDocumentStore(client, "docs")
    with pytest.raises(UnexpectedResponse):
        await store.upsert([make_document("a.pdf")])


# ---------------------------------------------------------------------------
# get_document
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_document_round_trips(store, client):
    doc = make_document("a.pdf", "h", body="Round trip me.")
    client.retrieve = AsyncMock(return_value=[_retrieve_record(doc)])

    loaded = await store.get_document("a.pdf")
    assert loaded is not None
    assert loaded.source_id == "a.pdf"
    assert loaded.content_hash() == doc.content_hash()
    # retrieval is by the deterministic point id (O(1), not a scroll)
    _, kwargs = client.retrieve.call_args
    ids = kwargs.get("ids") or client.retrieve.call_args[0][1]
    assert str(ids[0]) == _point_id("a.pdf")


@pytest.mark.anyio
async def test_get_document_absent_returns_none(store, client):
    client.retrieve = AsyncMock(return_value=[])
    assert await store.get_document("missing") is None


# ---------------------------------------------------------------------------
# delete_by_source
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_by_source_uses_point_id(store, client):
    await store.delete_by_source("a.pdf")
    client.delete.assert_called_once()


# ---------------------------------------------------------------------------
# list_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_source_ids_paginates(store, client):
    page1 = ([_scroll_point(make_document("a.pdf"))], "off-1")
    page2 = ([_scroll_point(make_document("b.pdf"))], None)
    client.scroll = AsyncMock(side_effect=[page1, page2])
    assert await store.list_source_ids() == {"a.pdf", "b.pdf"}
    assert client.scroll.call_count == 2


# ---------------------------------------------------------------------------
# list_source_state
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_source_state_reports_hashes(store, client):
    d1 = make_document("a.pdf", "ha")
    d2 = make_document("b.pdf", "hb")
    client.scroll = AsyncMock(return_value=([_scroll_point(d1), _scroll_point(d2)], None))

    state = await store.list_source_state()
    assert state == {
        "a.pdf": SourceState(source_hash="ha", content_hash=d1.content_hash()),
        "b.pdf": SourceState(source_hash="hb", content_hash=d2.content_hash()),
    }


@pytest.mark.anyio
async def test_list_source_state_is_a_cheap_read(store, client):
    """Bulk read must not pull full document payloads or vectors (keeps plan() cheap)."""
    client.scroll = AsyncMock(return_value=([], None))
    await store.list_source_state()
    _, kwargs = client.scroll.call_args
    assert kwargs.get("with_vectors") is False
    assert set(kwargs.get("with_payload")) == {"source_id", "source_hash", "content_hash"}


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_makes_collection_and_index(client):
    store = await QdrantDocumentStore.create(client, "docs")
    assert isinstance(store, QdrantDocumentStore)
    client.create_collection.assert_called_once()
    client.create_payload_index.assert_called_once()


@pytest.mark.anyio
async def test_create_is_idempotent_on_conflict(client):
    client.create_collection = AsyncMock(
        side_effect=UnexpectedResponse(status_code=409, reason_phrase="Conflict", content=b"exists", headers=None)
    )
    store = await QdrantDocumentStore.create(client, "docs")  # must not raise
    assert isinstance(store, QdrantDocumentStore)


# ---------------------------------------------------------------------------
# protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_document_store_protocol(store):
    assert isinstance(store, DocumentStore)
