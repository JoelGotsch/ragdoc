"""Unit tests for QdrantGraphStore (AsyncQdrantClient stubbed; no live Qdrant)."""

from __future__ import annotations

import hashlib
import uuid
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip("qdrant_client")

from ragdoc.extraction.entity import Entity
from ragdoc.extraction.schema import EdgeRef, GraphSchema
from ragdoc.integrations.graph_stores.qdrant import _NAMESPACE, QdrantGraphStore


def _point_id(entity_id: str) -> str:
    """Expected deterministic point id (UUID5 in the graph store's namespace)."""
    return str(uuid.uuid5(_NAMESPACE, entity_id))


class Person(BaseModel):
    kind: Literal["Person"] = "Person"
    full_name: str


class Company(BaseModel):
    kind: Literal["Company"] = "Company"
    name: str


class Employment(BaseModel):
    kind: Literal["Employment"] = "Employment"
    refs: EdgeRef
    role: str | None = None


SCHEMA = GraphSchema(
    node_types=(Person, Company),
    edge_types=(Employment,),
    patterns=((Person, Employment, Company),),
)


def _eid(member_ids: list[str]) -> str:
    return hashlib.sha256("\x00".join(sorted(member_ids)).encode("utf-8")).hexdigest()


def node(payload, member_ids, source_ids=("d",)) -> Entity:
    return Entity(
        entity_id=_eid(member_ids), payload=payload, member_mention_ids=member_ids, source_ids=list(source_ids)
    )


@pytest.fixture
def client() -> MagicMock:
    c = MagicMock()
    c.scroll = AsyncMock(return_value=([], None))
    c.retrieve = AsyncMock(return_value=[])
    c.upsert = AsyncMock()
    c.delete = AsyncMock()
    c.create_collection = AsyncMock()
    c.create_payload_index = AsyncMock()
    return c


@pytest.fixture
def store(client: MagicMock) -> QdrantGraphStore:
    return QdrantGraphStore(client, "kg", SCHEMA)


def _point(entity: Entity) -> MagicMock:
    p = MagicMock()
    p.payload = entity.model_dump(mode="json")
    return p


@pytest.mark.anyio
async def test_create_makes_two_collections_with_indexes(client):
    await QdrantGraphStore.create(client, "kg", SCHEMA)
    created = {c.kwargs["collection_name"] for c in client.create_collection.call_args_list}
    assert created == {"kg_nodes", "kg_edges"}
    indexed = {
        (c.kwargs["collection_name"], c.kwargs["field_name"]) for c in client.create_payload_index.call_args_list
    }
    # edges index the endpoint refs + kind so neighbors filters server-side
    assert ("kg_edges", "payload.refs.source_mention_id") in indexed
    assert ("kg_edges", "payload.refs.target_mention_id") in indexed
    assert ("kg_edges", "payload.kind") in indexed
    assert ("kg_nodes", "payload.kind") in indexed


@pytest.mark.anyio
async def test_upsert_nodes_and_edges_target_right_collections(store, client):
    alice = node(Person(full_name="Alice"), ["p-1"])
    await store.upsert_nodes([alice])
    edge = node(Employment(refs=EdgeRef(source_mention_id="a", target_mention_id="b")), ["e-1"])
    await store.upsert_edges([edge])

    calls = {c.kwargs["collection_name"]: c.kwargs["points"] for c in client.upsert.call_args_list}
    assert calls["kg_nodes"][0].id == _point_id(alice.entity_id)
    assert calls["kg_nodes"][0].vector == [0.0]
    assert calls["kg_edges"][0].payload["payload"]["kind"] == "Employment"


@pytest.mark.anyio
async def test_get_node_rehydrates_typed_payload(store, client):
    alice = node(Person(full_name="Alice"), ["p-1"])
    client.retrieve = AsyncMock(return_value=[_point(alice)])
    got = await store.get_node(alice.entity_id)
    assert isinstance(got.payload, Person) and got.payload.full_name == "Alice"
    assert client.retrieve.call_args.kwargs["ids"] == [_point_id(alice.entity_id)]


@pytest.mark.anyio
async def test_list_nodes_filters_kind_server_side(store, client):
    await store.list_nodes(Company)
    flt = client.scroll.call_args.kwargs["scroll_filter"]
    cond = flt.must[0]
    assert cond.key == "payload.kind" and cond.match.value == "Company"


@pytest.mark.anyio
async def test_neighbors_filters_edges_and_retrieves_nodes(store, client):
    alice = node(Person(full_name="Alice"), ["p-1"])
    acme = node(Company(name="Acme"), ["c-1"])
    edge = node(
        Employment(refs=EdgeRef(source_mention_id=alice.entity_id, target_mention_id=acme.entity_id)),
        ["e-1"],
    )
    # edges scroll returns the employment edge; node retrieve returns Acme
    client.scroll = AsyncMock(return_value=([_point(edge)], None))
    client.retrieve = AsyncMock(return_value=[_point(acme)])

    result = await store.neighbors(alice.entity_id, edge_types=["Employment"])

    # edge filter: (source==id OR target==id) AND kind in edge_types
    flt = client.scroll.call_args.kwargs["scroll_filter"]
    should_keys = {c.key for c in flt.must[0].should}
    assert should_keys == {"payload.refs.source_mention_id", "payload.refs.target_mention_id"}
    assert flt.must[1].key == "payload.kind"
    # the resolved neighbour is Acme, fetched from the nodes collection by point id
    assert client.retrieve.call_args.kwargs["ids"] == [_point_id(acme.entity_id)]
    assert [n.entity_id for n in result] == [acme.entity_id]


@pytest.mark.anyio
async def test_neighbors_empty_without_edges(store, client):
    client.scroll = AsyncMock(return_value=([], None))
    assert await store.neighbors("whatever") == []
    client.retrieve.assert_not_called()


@pytest.mark.anyio
async def test_delete_targets_both_collections(store, client):
    await store.delete(["e1"])
    deleted = {c.kwargs["collection_name"]: c.kwargs["points_selector"] for c in client.delete.call_args_list}
    assert set(deleted) == {"kg_nodes", "kg_edges"}
    assert deleted["kg_nodes"].points == [_point_id("e1")]


@pytest.mark.anyio
async def test_delete_by_source_only_solely_owned(store, client):
    solely = node(Person(full_name="Alice"), ["p-1"], source_ids=("d1",))
    shared = node(Person(full_name="Bob"), ["p-2"], source_ids=("d1", "d2"))

    # nodes scroll returns both; edges scroll returns nothing
    def scroll_side(**kwargs):
        coll = kwargs["collection_name"]
        return ([_point(solely), _point(shared)], None) if coll == "kg_nodes" else ([], None)

    client.scroll = AsyncMock(side_effect=scroll_side)
    await store.delete_by_source("d1")

    node_delete = [c for c in client.delete.call_args_list if c.kwargs["collection_name"] == "kg_nodes"]
    assert node_delete[0].kwargs["points_selector"].points == [_point_id(solely.entity_id)]


@pytest.mark.anyio
async def test_create_indexes_created_when_collections_already_exist(client):
    """A 409 on either collection must NOT skip that collection's field indexes (bug 4)."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    conflict = UnexpectedResponse(status_code=409, reason_phrase="Conflict", content=b"exists", headers={})  # type: ignore[arg-type]
    client.create_collection = AsyncMock(side_effect=conflict)
    client.create_payload_index = AsyncMock()
    await QdrantGraphStore.create(client, "kg", SCHEMA)
    indexed = {
        (c.kwargs["collection_name"], c.kwargs["field_name"]) for c in client.create_payload_index.await_args_list
    }
    assert ("kg_nodes", "entity_id") in indexed
    assert ("kg_nodes", "payload.kind") in indexed
    assert ("kg_edges", "payload.refs.source_mention_id") in indexed
    assert ("kg_edges", "payload.refs.target_mention_id") in indexed
