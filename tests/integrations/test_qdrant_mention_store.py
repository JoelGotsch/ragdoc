"""Unit tests for QdrantMentionStore (AsyncQdrantClient stubbed; no live Qdrant)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("qdrant_client")

from ragdoc.extraction.mention import Mention
from ragdoc.integrations.mention_stores.qdrant import QdrantMentionStore, _point_id


class Event(BaseModel):
    title: str = Field(description="t")


def make_mention(mention_id: str, source_id: str, *, source_hash: str = "h", content_hash: str = "c") -> Mention[Event]:
    return Mention[Event](
        mention_id=mention_id,
        source_id=source_id,
        source_hash=source_hash,
        content_hash=content_hash,
        payload=Event(title=f"E-{mention_id}"),
    )


@pytest.fixture
def client() -> MagicMock:
    c = MagicMock()
    c.scroll = AsyncMock(return_value=([], None))
    c.upsert = AsyncMock()
    c.delete = AsyncMock()
    c.create_collection = AsyncMock()
    c.create_payload_index = AsyncMock()
    return c


def make_embedder():
    async def _embed(payloads):  # type: ignore[no-untyped-def]
        return [[float(i), 0.0, 0.0] for i, _ in enumerate(payloads)]

    return _embed


@pytest.fixture
def store(client: MagicMock) -> QdrantMentionStore:
    return QdrantMentionStore(client, "mentions", Event, make_embedder())


@pytest.mark.anyio
async def test_upsert_one_point_per_mention_with_identity_vector(store, client):
    mentions = [make_mention("m0", "s1"), make_mention("m1", "s1")]
    result = await store.upsert(mentions)

    assert result == ["m0", "m1"]
    client.upsert.assert_called_once()
    points = client.upsert.call_args.kwargs["points"]
    assert [p.id for p in points] == [_point_id("m0"), _point_id("m1")]
    # the identity embedding is stored as the point vector (doubles as the blocking index)
    assert points[0].vector == [0.0, 0.0, 0.0]
    assert points[1].vector == [1.0, 0.0, 0.0]
    assert points[0].payload["source_id"] == "s1"


@pytest.mark.anyio
async def test_delete_by_source_uses_filter(store, client):
    await store.delete_by_source("s1")
    selector = client.delete.call_args.kwargs["points_selector"]
    assert selector.must[0].match.value == "s1"


@pytest.mark.anyio
async def test_list_source_state_bulk_read(store, client):
    point = MagicMock()
    point.payload = {"source_id": "s1", "source_hash": "h1", "content_hash": "c1"}
    client.scroll = AsyncMock(return_value=([point], None))

    state = await store.list_source_state()
    assert state["s1"].source_hash == "h1" and state["s1"].content_hash == "c1"


@pytest.mark.anyio
async def test_list_mentions_rehydrates(store, client):
    m = make_mention("m0", "s1")
    point = MagicMock()
    point.payload = m.model_dump(mode="json")
    client.scroll = AsyncMock(return_value=([point], None))

    mentions = await store.list_mentions()
    assert len(mentions) == 1
    assert isinstance(mentions[0], Mention)
    assert mentions[0].payload.title == "E-m0"


@pytest.mark.anyio
async def test_create_indexes_created_when_collection_already_exists(client):
    """A 409 on create_collection must NOT skip payload-index creation (fable-review Phase 0, bug 4)."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    conflict = UnexpectedResponse(status_code=409, reason_phrase="Conflict", content=b"exists", headers={})  # type: ignore[arg-type]
    client.create_collection = AsyncMock(side_effect=conflict)
    client.create_payload_index = AsyncMock()
    await QdrantMentionStore.create(client, "mentions", Event, make_embedder(), vector_size=4)
    created = {c.kwargs["field_name"] for c in client.create_payload_index.await_args_list}
    assert created == {"source_id"}
