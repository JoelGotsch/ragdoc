"""Unit tests for QdrantVectorStore.

All tests stub AsyncQdrantClient with MagicMock/AsyncMock — no live Qdrant instance required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

pytest.importorskip("qdrant_client", reason="qdrant extra not installed")

from ragdoc.chunking import Chunk
from ragdoc.integrations.vector_stores.qdrant import QdrantVectorStore, ServerSideVector, _chunk_to_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(**overrides: object) -> Chunk:
    defaults: dict = dict(
        prompt_content="prompt",
        embedding_content="embed",
        named_embeddings={"dense": [0.1, 0.2, 0.3]},
        source_id="doc-a",
        source_hash="abc123",
    )
    return Chunk(**(defaults | overrides))


def _make_point(source_id: str | None = "doc-a", source_hash: str | None = "abc123") -> MagicMock:
    point = MagicMock()
    point.payload = {"source_id": source_id, "source_hash": source_hash}
    return point


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def store(client: MagicMock) -> QdrantVectorStore:
    return QdrantVectorStore(client, "test_collection")


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upsert_calls_qdrant_upsert(store: QdrantVectorStore, client: MagicMock) -> None:
    chunk = make_chunk()
    await store.upsert([chunk])
    client.upsert.assert_called_once()
    args, kwargs = client.upsert.call_args
    assert kwargs.get("collection_name") == "test_collection" or args[0] == "test_collection"


@pytest.mark.anyio
async def test_upsert_payload_contains_source_fields(store: QdrantVectorStore, client: MagicMock) -> None:
    chunk = make_chunk()
    await store.upsert([chunk])
    _, kwargs = client.upsert.call_args
    points = kwargs.get("points") or client.upsert.call_args[0][1]
    point = points[0]
    assert point.payload["source_id"] == "doc-a"
    assert point.payload["source_hash"] == "abc123"
    assert point.payload["prompt_content"] == "prompt"


@pytest.mark.anyio
async def test_upsert_missing_embedding_raises(store: QdrantVectorStore) -> None:
    chunk = make_chunk(named_embeddings={})
    with pytest.raises(ValueError, match="embedding"):
        await store.upsert([chunk])


@pytest.mark.anyio
async def test_upsert_returns_chunk_ids(store: QdrantVectorStore, client: MagicMock) -> None:
    chunk = make_chunk()
    result = await store.upsert([chunk])
    assert result == [chunk.id]


@pytest.mark.anyio
async def test_upsert_point_id_matches_chunk_id(store: QdrantVectorStore, client: MagicMock) -> None:
    chunk = make_chunk()
    await store.upsert([chunk])
    _, kwargs = client.upsert.call_args
    points = kwargs.get("points") or client.upsert.call_args[0][1]
    assert str(points[0].id) == chunk.id


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_calls_qdrant_delete(store: QdrantVectorStore, client: MagicMock) -> None:
    await store.delete(["id-1", "id-2"])
    client.delete.assert_called_once()


@pytest.mark.anyio
async def test_delete_empty_list_is_noop(store: QdrantVectorStore, client: MagicMock) -> None:
    await store.delete([])
    client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# get_source_hash
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_source_hash_found(store: QdrantVectorStore, client: MagicMock) -> None:
    client.scroll = AsyncMock(return_value=([_make_point("doc-a", "abc123")], None))
    result = await store.get_source_hash("doc-a")
    assert result == "abc123"


@pytest.mark.anyio
async def test_get_source_hash_not_found(store: QdrantVectorStore, client: MagicMock) -> None:
    client.scroll = AsyncMock(return_value=([], None))
    result = await store.get_source_hash("doc-missing")
    assert result is None


# ---------------------------------------------------------------------------
# delete_by_source
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_by_source_uses_filter(store: QdrantVectorStore, client: MagicMock) -> None:
    await store.delete_by_source("doc-a")
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    # The call must pass a points_selector that is a Filter (not a list of IDs)
    selector = kwargs.get("points_selector") or client.delete.call_args[0][1]
    # We just verify it's not a plain list (it should be a Filter object)
    assert not isinstance(selector, list)


# ---------------------------------------------------------------------------
# list_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_source_ids_empty(store: QdrantVectorStore, client: MagicMock) -> None:
    client.scroll = AsyncMock(return_value=([], None))
    result = await store.list_source_ids()
    assert result == set()


@pytest.mark.anyio
async def test_list_source_ids_single_page(store: QdrantVectorStore, client: MagicMock) -> None:
    points = [_make_point("doc-a"), _make_point("doc-b")]
    client.scroll = AsyncMock(return_value=(points, None))
    result = await store.list_source_ids()
    assert result == {"doc-a", "doc-b"}


@pytest.mark.anyio
async def test_list_source_ids_paginates(store: QdrantVectorStore, client: MagicMock) -> None:
    page1 = ([_make_point("doc-a")], "offset-1")
    page2 = ([_make_point("doc-b")], None)
    client.scroll = AsyncMock(side_effect=[page1, page2])
    result = await store.list_source_ids()
    assert result == {"doc-a", "doc-b"}
    assert client.scroll.call_count == 2


@pytest.mark.anyio
async def test_list_source_ids_deduplicates(store: QdrantVectorStore, client: MagicMock) -> None:
    points = [_make_point("doc-a"), _make_point("doc-a"), _make_point("doc-a")]
    client.scroll = AsyncMock(return_value=(points, None))
    result = await store.list_source_ids()
    assert result == {"doc-a"}


# ---------------------------------------------------------------------------
# create classmethod
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_creates_collection(client: MagicMock) -> None:
    # Simulate collection does not exist (no exception)
    client.create_collection = AsyncMock()
    store = await QdrantVectorStore.create(client, "my_col", vector_size=128)
    client.create_collection.assert_called_once()
    assert isinstance(store, QdrantVectorStore)


@pytest.mark.anyio
async def test_create_skips_if_collection_exists(client: MagicMock) -> None:
    from qdrant_client.http.exceptions import UnexpectedResponse

    # Simulate the exception Qdrant raises when a collection already exists
    conflict = UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b"Collection already exists",
        headers={},  # type: ignore[arg-type]
    )
    client.create_collection = AsyncMock(side_effect=conflict)
    # Should not raise
    store = await QdrantVectorStore.create(client, "existing_col", vector_size=128)
    assert isinstance(store, QdrantVectorStore)


# ---------------------------------------------------------------------------
# ServerSideVector / BM25 sparse
# ---------------------------------------------------------------------------


def test_chunk_to_point_with_sparse_adds_document() -> None:
    from qdrant_client import models as qdrant_models

    chunk = make_chunk(prompt_content="full text for bm25")
    bm25 = ServerSideVector(model="Qdrant/bm25")
    point = _chunk_to_point(chunk, sparse_vectors={"sparse": bm25})

    assert isinstance(point.vector, dict)
    assert "dense" in point.vector
    assert "sparse" in point.vector
    sparse_val = point.vector["sparse"]
    assert isinstance(sparse_val, qdrant_models.Document)
    assert sparse_val.text == "full text for bm25"
    assert sparse_val.model == "Qdrant/bm25"


def test_chunk_to_point_sparse_only_no_dense_raises() -> None:
    chunk = make_chunk(named_embeddings={})
    with pytest.raises(ValueError, match="embedding"):
        _chunk_to_point(chunk, sparse_vectors=None)


def test_chunk_to_point_sparse_only_with_server_side_passes() -> None:
    from qdrant_client import models as qdrant_models

    chunk = make_chunk(named_embeddings={})
    bm25 = ServerSideVector(model="Qdrant/bm25")
    point = _chunk_to_point(chunk, sparse_vectors={"sparse": bm25})
    assert isinstance(point.vector["sparse"], qdrant_models.Document)


def test_server_side_vector_custom_text_fn() -> None:
    from qdrant_client import models as qdrant_models

    chunk = make_chunk(embedding_content="embedding summary", prompt_content="full text")
    bm25 = ServerSideVector(model="Qdrant/bm25", text_fn=lambda c: c.embedding_content)
    point = _chunk_to_point(chunk, sparse_vectors={"sparse": bm25})
    assert point.vector["sparse"].text == "embedding summary"


@pytest.mark.anyio
async def test_create_with_sparse_vectors_passes_sparse_config(client: MagicMock) -> None:
    bm25 = ServerSideVector(model="Qdrant/bm25")
    store = await QdrantVectorStore.create(
        client, "hybrid_col", vector_size=128, sparse_vectors={"sparse": bm25}
    )
    assert isinstance(store, QdrantVectorStore)
    _, kwargs = client.create_collection.call_args
    assert "sparse_vectors_config" in kwargs
    assert "sparse" in kwargs["sparse_vectors_config"]
    # Dense vector must be named (dict form) when sparse is present
    assert isinstance(kwargs["vectors_config"], dict)
    assert "dense" in kwargs["vectors_config"]


@pytest.mark.anyio
async def test_create_without_sparse_vectors_uses_unnamed_config(client: MagicMock) -> None:
    from qdrant_client import models as qdrant_models

    store = await QdrantVectorStore.create(client, "dense_only_col", vector_size=128)
    assert isinstance(store, QdrantVectorStore)
    _, kwargs = client.create_collection.call_args
    assert "sparse_vectors_config" not in kwargs
    assert isinstance(kwargs["vectors_config"], qdrant_models.VectorParams)


@pytest.mark.anyio
async def test_upsert_with_sparse_vectors_includes_document(client: MagicMock) -> None:
    from qdrant_client import models as qdrant_models

    bm25 = ServerSideVector(model="Qdrant/bm25")
    store = QdrantVectorStore(client, "col", sparse_vectors={"sparse": bm25})
    chunk = make_chunk(prompt_content="text for bm25")
    await store.upsert([chunk])

    _, kwargs = client.upsert.call_args
    point = (kwargs.get("points") or client.upsert.call_args[0][1])[0]
    assert isinstance(point.vector["sparse"], qdrant_models.Document)
    assert point.vector["sparse"].text == "text for bm25"
