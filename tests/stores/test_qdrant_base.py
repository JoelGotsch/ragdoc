"""Unit tests for the shared Qdrant base store (`_QdrantCollectionStore`).

Covers the collection-agnostic plumbing extracted from the five Qdrant stores
(Phase 3b of PLAN-fable-review-remediation): ensure-collection with per-index
409 handling, scroll pagination, batched upsert, and namespaced point ids.
All tests stub AsyncQdrantClient — no live Qdrant instance required.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("qdrant_client", reason="qdrant extra not installed")

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from ragdoc.integrations._qdrant_base import _QdrantCollectionStore

_NS_A = uuid.UUID("00000000-0000-0000-0000-00000000000a")
_NS_B = uuid.UUID("00000000-0000-0000-0000-00000000000b")


class _StoreA(_QdrantCollectionStore):
    _ID_NAMESPACE = _NS_A


class _StoreB(_QdrantCollectionStore):
    _ID_NAMESPACE = _NS_B


class _NoNamespaceStore(_QdrantCollectionStore):
    pass


@pytest.fixture
def client() -> MagicMock:
    c = MagicMock()
    c.scroll = AsyncMock(return_value=([], None))
    c.upsert = AsyncMock()
    c.delete = AsyncMock()
    c.create_collection = AsyncMock()
    c.create_payload_index = AsyncMock()
    return c


@pytest.fixture
def store(client: MagicMock) -> _StoreA:
    return _StoreA(client, "base_col")


def _point(payload: dict | None) -> MagicMock:
    p = MagicMock()
    p.payload = payload
    return p


def _dummy_points(n: int) -> list[models.PointStruct]:
    return [models.PointStruct(id=str(uuid.uuid4()), vector=[0.0], payload={"i": i}) for i in range(n)]


def _conflict() -> UnexpectedResponse:
    return UnexpectedResponse(status_code=409, reason_phrase="Conflict", content=b"exists", headers={})  # type: ignore[arg-type]


def _server_error() -> UnexpectedResponse:
    return UnexpectedResponse(status_code=500, reason_phrase="Server Error", content=b"boom", headers={})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _point_id — determinism + namespacing
# ---------------------------------------------------------------------------


def test_point_id_is_deterministic_uuid5(store: _StoreA) -> None:
    assert store._point_id("key-1") == str(uuid.uuid5(_NS_A, "key-1"))
    assert store._point_id("key-1") == store._point_id("key-1")
    assert store._point_id("key-1") != store._point_id("key-2")


def test_point_id_differs_across_namespaces(client: MagicMock) -> None:
    a, b = _StoreA(client, "col"), _StoreB(client, "col")
    assert a._point_id("same-key") != b._point_id("same-key")
    assert b._point_id("same-key") == str(uuid.uuid5(_NS_B, "same-key"))


def test_point_id_without_namespace_raises(client: MagicMock) -> None:
    store = _NoNamespaceStore(client, "col")
    with pytest.raises(TypeError, match="_ID_NAMESPACE"):
        store._point_id("key")


# ---------------------------------------------------------------------------
# _scroll_payloads — pagination
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_scroll_paginates_across_batches(store: _StoreA, client: MagicMock) -> None:
    page1 = ([_point({"n": 1}), _point({"n": 2})], "off-1")
    page2 = ([_point({"n": 3})], "off-2")
    page3 = ([_point({"n": 4})], None)
    client.scroll = AsyncMock(side_effect=[page1, page2, page3])

    payloads = [p async for p in store._scroll_payloads(True)]

    assert [p["n"] for p in payloads] == [1, 2, 3, 4]
    assert client.scroll.call_count == 3
    # the offset returned by each page is threaded into the next call
    offsets = [c.kwargs["offset"] for c in client.scroll.call_args_list]
    assert offsets == [None, "off-1", "off-2"]


@pytest.mark.anyio
async def test_scroll_never_fetches_vectors_and_uses_own_collection(store: _StoreA, client: MagicMock) -> None:
    [_ async for _ in store._scroll_payloads(["source_id"])]
    kwargs = client.scroll.call_args.kwargs
    assert kwargs["with_vectors"] is False
    assert kwargs["with_payload"] == ["source_id"]
    assert kwargs["collection_name"] == "base_col"
    assert kwargs["scroll_filter"] is None


@pytest.mark.anyio
async def test_scroll_collection_override_and_filter(store: _StoreA, client: MagicMock) -> None:
    flt = models.Filter(must=[models.FieldCondition(key="k", match=models.MatchValue(value="v"))])
    [_ async for _ in store._scroll_payloads(True, scroll_filter=flt, collection="other_col")]
    kwargs = client.scroll.call_args.kwargs
    assert kwargs["collection_name"] == "other_col"
    assert kwargs["scroll_filter"] is flt


@pytest.mark.anyio
async def test_scroll_none_payload_yields_empty_dict(store: _StoreA, client: MagicMock) -> None:
    client.scroll = AsyncMock(return_value=([_point(None)], None))
    payloads = [p async for p in store._scroll_payloads(True)]
    assert payloads == [{}]


# ---------------------------------------------------------------------------
# _batched_upsert — batch-size boundary
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_batched_upsert_empty_is_noop(store: _StoreA, client: MagicMock) -> None:
    await store._batched_upsert([])
    client.upsert.assert_not_called()


@pytest.mark.anyio
async def test_batched_upsert_exactly_one_batch(store: _StoreA, client: MagicMock) -> None:
    points = _dummy_points(_QdrantCollectionStore._UPSERT_BATCH)
    await store._batched_upsert(points)
    assert client.upsert.call_count == 1
    assert client.upsert.call_args.kwargs["points"] == points


@pytest.mark.anyio
async def test_batched_upsert_splits_past_the_boundary(store: _StoreA, client: MagicMock) -> None:
    batch = _QdrantCollectionStore._UPSERT_BATCH
    points = _dummy_points(batch + 1)
    await store._batched_upsert(points)
    sizes = [len(c.kwargs["points"]) for c in client.upsert.call_args_list]
    assert sizes == [batch, 1]
    # order is preserved across batches
    sent = [p for c in client.upsert.call_args_list for p in c.kwargs["points"]]
    assert sent == points


@pytest.mark.anyio
async def test_batched_upsert_collection_override(store: _StoreA, client: MagicMock) -> None:
    await store._batched_upsert(_dummy_points(1), collection="other_col")
    assert client.upsert.call_args.kwargs["collection_name"] == "other_col"


# ---------------------------------------------------------------------------
# _ensure_collection — creation matrix (per-index 409 handling)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ensure_collection_fresh_creates_collection_and_indexes(client: MagicMock) -> None:
    await _QdrantCollectionStore._ensure_collection(
        client,
        "col",
        models.VectorParams(size=4, distance=models.Distance.COSINE),
        payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD, "kind": models.PayloadSchemaType.KEYWORD},
    )
    client.create_collection.assert_called_once()
    assert "sparse_vectors_config" not in client.create_collection.call_args.kwargs
    created = {c.kwargs["field_name"] for c in client.create_payload_index.await_args_list}
    assert created == {"source_id", "kind"}


@pytest.mark.anyio
async def test_ensure_collection_409_still_creates_indexes(client: MagicMock) -> None:
    """A 409 on create_collection must NOT skip payload-index creation (fable-review Phase 0, bug 4)."""
    client.create_collection = AsyncMock(side_effect=_conflict())
    await _QdrantCollectionStore._ensure_collection(
        client,
        "existing_col",
        models.VectorParams(size=4, distance=models.Distance.COSINE),
        payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD},
    )
    created = {c.kwargs["field_name"] for c in client.create_payload_index.await_args_list}
    assert created == {"source_id"}


@pytest.mark.anyio
async def test_ensure_collection_index_409_is_ignored(client: MagicMock) -> None:
    client.create_payload_index = AsyncMock(side_effect=_conflict())
    await _QdrantCollectionStore._ensure_collection(
        client,
        "col",
        models.VectorParams(size=4, distance=models.Distance.COSINE),
        payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD},
    )  # must not raise


@pytest.mark.anyio
async def test_ensure_collection_non_409_collection_error_reraised(client: MagicMock) -> None:
    client.create_collection = AsyncMock(side_effect=_server_error())
    with pytest.raises(UnexpectedResponse):
        await _QdrantCollectionStore._ensure_collection(
            client, "col", models.VectorParams(size=4, distance=models.Distance.COSINE)
        )


@pytest.mark.anyio
async def test_ensure_collection_non_409_index_error_reraised(client: MagicMock) -> None:
    client.create_payload_index = AsyncMock(side_effect=_server_error())
    with pytest.raises(UnexpectedResponse):
        await _QdrantCollectionStore._ensure_collection(
            client,
            "col",
            models.VectorParams(size=4, distance=models.Distance.COSINE),
            payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD},
        )


@pytest.mark.anyio
async def test_ensure_collection_passes_sparse_config_when_given(client: MagicMock) -> None:
    sparse = {"sparse": models.SparseVectorParams()}
    await _QdrantCollectionStore._ensure_collection(
        client,
        "col",
        {"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config=sparse,
    )
    assert client.create_collection.call_args.kwargs["sparse_vectors_config"] is sparse


# ---------------------------------------------------------------------------
# _list_source_state / _collect_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_source_state_first_seen_wins_and_cheap_keys(store: _StoreA, client: MagicMock) -> None:
    points = [
        _point({"source_id": "a", "source_hash": "h1", "content_hash": "c1"}),
        _point({"source_id": "a", "source_hash": "h2", "content_hash": "c2"}),  # ignored: first seen wins
        _point({"source_id": "b", "source_hash": None, "content_hash": None}),  # missing hashes normalized
    ]
    client.scroll = AsyncMock(return_value=(points, None))

    state = await store._list_source_state()

    assert state["a"].source_hash == "h1" and state["a"].content_hash == "c1"
    assert state["b"].source_hash == "" and state["b"].content_hash is None
    assert set(client.scroll.call_args.kwargs["with_payload"]) == {"source_id", "source_hash", "content_hash"}


@pytest.mark.anyio
async def test_collect_source_ids_deduplicates_and_skips_missing(store: _StoreA, client: MagicMock) -> None:
    points = [_point({"source_id": "a"}), _point({"source_id": "a"}), _point({}), _point({"source_id": "b"})]
    client.scroll = AsyncMock(return_value=(points, None))
    assert await store._collect_source_ids() == {"a", "b"}


@pytest.mark.anyio
async def test_collect_source_ids_custom_key(store: _StoreA, client: MagicMock) -> None:
    client.scroll = AsyncMock(return_value=([_point({"entity_id": "e1"})], None))
    assert await store._collect_source_ids(key="entity_id") == {"e1"}
    assert client.scroll.call_args.kwargs["with_payload"] == ["entity_id"]


# ---------------------------------------------------------------------------
# _delete_by_source_filter
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_by_source_filter_builds_source_id_filter(store: _StoreA, client: MagicMock) -> None:
    await store._delete_by_source_filter("doc-a")
    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "base_col"
    selector = kwargs["points_selector"]
    assert isinstance(selector, models.Filter)
    assert selector.must[0].key == "source_id"
    assert selector.must[0].match.value == "doc-a"


@pytest.mark.anyio
async def test_delete_by_source_filter_collection_override(store: _StoreA, client: MagicMock) -> None:
    await store._delete_by_source_filter("doc-a", collection="other_col")
    assert client.delete.call_args.kwargs["collection_name"] == "other_col"
