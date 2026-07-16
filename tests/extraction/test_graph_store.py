"""Tests for GraphStore Protocol + LocalGraphStore — JSON-per-entity on disk."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from ragdoc.extraction.entity import Entity
from ragdoc.extraction.graph_store import LocalGraphStore
from ragdoc.extraction.schema import EdgeRef, GraphSchema

# ---------------------------------------------------------------------------
# Sample schema
# ---------------------------------------------------------------------------


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


class Authorship(BaseModel):
    kind: Literal["Authorship"] = "Authorship"
    refs: EdgeRef


SCHEMA = GraphSchema(
    node_types=(Person, Company),
    edge_types=(Employment, Authorship),
    patterns=(
        (Person, Employment, Company),
        (Person, Authorship, Company),
    ),
)


def _entity_id(member_mention_ids: list[str]) -> str:
    return hashlib.sha256("\x00".join(sorted(member_mention_ids)).encode("utf-8")).hexdigest()


def _node_entity(payload, member_ids, source_ids) -> Entity:
    return Entity(
        entity_id=_entity_id(member_ids),
        payload=payload,
        member_mention_ids=member_ids,
        source_ids=source_ids,
    )


def _edge_entity(payload, member_ids, source_ids) -> Entity:
    return Entity(
        entity_id=_entity_id(member_ids),
        payload=payload,
        member_mention_ids=member_ids,
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# 1. Round-trip nodes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_round_trip_nodes(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["doc1"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["doc1"])
    await store.upsert_nodes([alice, acme])

    got_alice = await store.get_node(alice.entity_id)
    got_acme = await store.get_node(acme.entity_id)
    assert got_alice is not None and isinstance(got_alice.payload, Person)
    assert got_alice.payload.full_name == "Alice"
    assert got_acme is not None and got_acme.payload.name == "Acme"


@pytest.mark.anyio
async def test_get_node_missing_returns_none(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    assert await store.get_node("no-such-id") is None


# ---------------------------------------------------------------------------
# 2. Round-trip edges + neighbors
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_neighbors_finds_connected_node_by_edge(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["doc1"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["doc1"])
    await store.upsert_nodes([alice, acme])

    employment = _edge_entity(
        Employment(
            refs=EdgeRef(source_mention_id=alice.entity_id, target_mention_id=acme.entity_id),
            role="Engineer",
        ),
        ["e-1"],
        ["doc1"],
    )
    await store.upsert_edges([employment])

    # neighbors(alice) → [acme]
    neighbors = await store.neighbors(alice.entity_id)
    assert len(neighbors) == 1 and neighbors[0].entity_id == acme.entity_id

    # neighbors are bidirectional — calling with the target also returns the source
    reverse = await store.neighbors(acme.entity_id)
    assert len(reverse) == 1 and reverse[0].entity_id == alice.entity_id


@pytest.mark.anyio
async def test_neighbors_empty_when_no_edges(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["doc1"])
    await store.upsert_nodes([alice])
    assert await store.neighbors(alice.entity_id) == []


# ---------------------------------------------------------------------------
# 3. neighbors with edge_types filter
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_neighbors_filtered_by_edge_type(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["doc1"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["doc1"])
    initech = _node_entity(Company(name="Initech"), ["c-2"], ["doc1"])
    await store.upsert_nodes([alice, acme, initech])

    employment = _edge_entity(
        Employment(refs=EdgeRef(source_mention_id=alice.entity_id, target_mention_id=acme.entity_id)),
        ["e-1"],
        ["doc1"],
    )
    authorship = _edge_entity(
        Authorship(refs=EdgeRef(source_mention_id=alice.entity_id, target_mention_id=initech.entity_id)),
        ["e-2"],
        ["doc1"],
    )
    await store.upsert_edges([employment, authorship])

    # No filter: both Acme and Initech
    both = await store.neighbors(alice.entity_id)
    assert {n.entity_id for n in both} == {acme.entity_id, initech.entity_id}

    # Filter to Employment: only Acme
    employment_only = await store.neighbors(alice.entity_id, edge_types=["Employment"])
    assert [n.entity_id for n in employment_only] == [acme.entity_id]

    # Filter to Authorship: only Initech
    authorship_only = await store.neighbors(alice.entity_id, edge_types=["Authorship"])
    assert [n.entity_id for n in authorship_only] == [initech.entity_id]


# ---------------------------------------------------------------------------
# 4. delete_by_source cascade — single-source entities removed, multi-source preserved
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_by_source_cascades_nodes_and_edges(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["doc1"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["doc1"])
    edge = _edge_entity(
        Employment(refs=EdgeRef(source_mention_id=alice.entity_id, target_mention_id=acme.entity_id)),
        ["e-1"],
        ["doc1"],
    )
    await store.upsert_nodes([alice, acme])
    await store.upsert_edges([edge])

    await store.delete_by_source("doc1")
    assert await store.list_nodes() == []
    assert await store.list_edges() == []


@pytest.mark.anyio
async def test_delete_by_source_preserves_multi_source_entities(tmp_path: Path):
    """An entity whose source_ids has TWO origins is kept when only one source is deleted."""
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice_multi = _node_entity(Person(full_name="Alice"), ["p-1", "p-2"], ["doc1", "doc2"])
    bob_single = _node_entity(Person(full_name="Bob"), ["p-3"], ["doc1"])
    await store.upsert_nodes([alice_multi, bob_single])

    await store.delete_by_source("doc1")
    remaining = await store.list_nodes()
    assert {n.entity_id for n in remaining} == {alice_multi.entity_id}


# ---------------------------------------------------------------------------
# 5. Re-extraction stability — entity_id stable across re-runs (same member ids)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_entity_id_stable_across_reruns(tmp_path: Path):
    """The Entity scheme hashes sorted member_mention_ids; re-extraction with the same
    member ids must produce the same entity_id, so cross-document edges keep resolving."""
    store = LocalGraphStore(tmp_path, SCHEMA)

    # First write
    e1 = _node_entity(Person(full_name="Alice"), ["p-1", "p-2"], ["doc1", "doc2"])
    await store.upsert_nodes([e1])

    # Same member ids, possibly different payload (a re-extraction edge case)
    e2 = _node_entity(Person(full_name="Alice Smith"), ["p-1", "p-2"], ["doc1", "doc2"])

    assert e1.entity_id == e2.entity_id
    # Upsert replaces in place
    await store.upsert_nodes([e2])
    got = await store.get_node(e1.entity_id)
    assert got is not None and got.payload.full_name == "Alice Smith"


# ---------------------------------------------------------------------------
# 6. list_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_source_ids_aggregates_nodes_and_edges(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["doc-A"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["doc-B"])
    edge = _edge_entity(
        Employment(refs=EdgeRef(source_mention_id=alice.entity_id, target_mention_id=acme.entity_id)),
        ["e-1"],
        ["doc-C"],
    )
    await store.upsert_nodes([alice, acme])
    await store.upsert_edges([edge])

    assert await store.list_source_ids() == {"doc-A", "doc-B", "doc-C"}


# ---------------------------------------------------------------------------
# 7. list_nodes / list_edges payload_type filter
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_nodes_filter_by_payload_type(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["d"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["d"])
    await store.upsert_nodes([alice, acme])

    persons = await store.list_nodes(Person)
    companies = await store.list_nodes(Company)
    assert [n.payload.full_name for n in persons] == ["Alice"]
    assert [n.payload.name for n in companies] == ["Acme"]


@pytest.mark.anyio
async def test_delete_by_id_removes_specific_entities(tmp_path: Path):
    store = LocalGraphStore(tmp_path, SCHEMA)
    alice = _node_entity(Person(full_name="Alice"), ["p-1"], ["d"])
    acme = _node_entity(Company(name="Acme"), ["c-1"], ["d"])
    await store.upsert_nodes([alice, acme])

    await store.delete([alice.entity_id])
    remaining = await store.list_nodes()
    assert [n.entity_id for n in remaining] == [acme.entity_id]
