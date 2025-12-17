"""Unit tests for QdrantEntityStore (AsyncQdrantClient stubbed; no live Qdrant)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("qdrant_client")

from qdrant_client import models

from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.entity import Entity
from ragdoc.extraction.query import DateRange, EntityQuery
from ragdoc.integrations.entity_stores.qdrant import QdrantEntityStore, _point_id


class Event(BaseModel):
    title: str = Field(description="t")


def make_entity(entity_id: str, source_ids: list[str], *, edtf: str | None = None) -> Entity[Event]:
    date = FuzzyDate(original_text=edtf or "", edtf=edtf, precision="YEAR") if edtf else None
    return Entity[Event](
        entity_id=entity_id,
        payload=Event(title=f"E-{entity_id}"),
        member_mention_ids=[f"{entity_id}-m"],
        source_ids=source_ids,
        date=date,
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


@pytest.fixture
def store(client: MagicMock) -> QdrantEntityStore:
    return QdrantEntityStore(client, "entities", Event)


@pytest.mark.anyio
async def test_upsert_dummy_vector_and_ids(store, client):
    result = await store.upsert([make_entity("e1", ["s1"]), make_entity("e2", ["s1", "s2"])])
    assert result == ["e1", "e2"]
    points = client.upsert.call_args.kwargs["points"]
    assert [p.id for p in points] == [_point_id("e1"), _point_id("e2")]
    assert points[0].vector == [0.0]  # throwaway 1-dim vector
    assert points[1].payload["source_ids"] == ["s1", "s2"]


@pytest.mark.anyio
async def test_delete_by_ids(store, client):
    await store.delete(["e1", "e2"])
    selector = client.delete.call_args.kwargs["points_selector"]
    assert selector.points == [_point_id("e1"), _point_id("e2")]


@pytest.mark.anyio
async def test_list_entities_rehydrates(store, client):
    e = make_entity("e1", ["s1"])
    point = MagicMock()
    point.payload = e.model_dump(mode="json")
    client.scroll = AsyncMock(return_value=([point], None))

    entities = await store.list_entities()
    assert len(entities) == 1 and isinstance(entities[0], Entity)
    assert entities[0].payload.title == "E-e1"


@pytest.mark.anyio
async def test_upsert_adds_flat_date_bounds(store, client):
    await store.upsert([make_entity("e1", ["s1"], edtf="1994")])
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["date_start"].startswith("1994-01-01")
    assert payload["date_end"].startswith("1995-01-01")  # exclusive end of the year


@pytest.mark.anyio
async def test_create_indexes_date_bounds_and_payload_fields(client):
    await QdrantEntityStore.create(
        client, "entities", Event, indexed_fields={"title": models.PayloadSchemaType.KEYWORD}
    )
    indexed = {c.kwargs["field_name"]: c.kwargs["field_schema"] for c in client.create_payload_index.call_args_list}
    assert indexed["date_start"] == models.PayloadSchemaType.DATETIME
    assert indexed["date_end"] == models.PayloadSchemaType.DATETIME
    assert indexed["payload.title"] == models.PayloadSchemaType.KEYWORD


@pytest.mark.anyio
async def test_query_builds_overlap_and_where_filter(store, client):
    import datetime as dt

    e = make_entity("e1", ["s1"], edtf="1994")
    point = MagicMock()
    point.payload = e.model_dump(mode="json")
    client.scroll = AsyncMock(return_value=([point], None))

    q = EntityQuery(date=DateRange(dt.date(1990, 1, 1), dt.date(1996, 1, 1)), where={"title": "E-e1"})
    results = await store.query(q)

    assert [r.entity_id for r in results] == ["e1"]
    flt = client.scroll.call_args.kwargs["scroll_filter"]
    keys = {c.key for c in flt.must}
    # half-open overlap: date_start < query.end AND date_end > query.start, plus the where match
    assert keys == {"date_start", "date_end", "payload.title"}


@pytest.mark.anyio
async def test_delete_by_source_only_solely_owned(store, client):
    solely = make_entity("e1", ["s1"])  # only from s1 → deleted
    shared = make_entity("e2", ["s1", "s2"])  # also from s2 → kept
    p1, p2 = MagicMock(), MagicMock()
    p1.payload = solely.model_dump(mode="json")
    p2.payload = shared.model_dump(mode="json")
    client.scroll = AsyncMock(return_value=([p1, p2], None))

    await store.delete_by_source("s1")
    selector = client.delete.call_args.kwargs["points_selector"]
    assert selector.points == [_point_id("e1")]


@pytest.mark.anyio
async def test_create_indexes_created_when_collection_already_exists(client):
    """A 409 on create_collection must NOT skip index creation — incl. user indexed_fields (bug 4)."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    conflict = UnexpectedResponse(status_code=409, reason_phrase="Conflict", content=b"exists", headers={})  # type: ignore[arg-type]
    client.create_collection = AsyncMock(side_effect=conflict)
    client.create_payload_index = AsyncMock()
    await QdrantEntityStore.create(
        client, "entities", Event, indexed_fields={"title": models.PayloadSchemaType.KEYWORD}
    )
    created = {c.kwargs["field_name"] for c in client.create_payload_index.await_args_list}
    assert created == {"entity_id", "date_start", "date_end", "payload.title"}
