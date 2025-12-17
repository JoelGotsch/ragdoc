"""Tests for KnowledgeGraphResolutionPipeline — per-type node resolution + edge rewrite."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from ragdoc.extraction.kg_resolution import (
    KGResolutionResult,
    KnowledgeGraphResolutionPipeline,
    _rewrite_edge_endpoints,
)
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.schema import EdgeRef, GraphSchema

from .conftest import (
    MemoryEntityStore,
    dict_embed,
    merge_all_reviewer,
    no_merge_reviewer,
)

# ---------------------------------------------------------------------------
# Sample schema (binary + n-ary) used across tests
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """A person."""

    kind: Literal["Person"] = "Person"
    full_name: str


class Company(BaseModel):
    """A company."""

    kind: Literal["Company"] = "Company"
    name: str


class Employment(BaseModel):
    """Employment edge."""

    kind: Literal["Employment"] = "Employment"
    refs: EdgeRef
    role: str | None = None


class Payment(BaseModel):
    """A reified n-ary node — a Payment event."""

    kind: Literal["Payment"] = "Payment"
    amount: str


class PaymentPayer(BaseModel):
    kind: Literal["PaymentPayer"] = "PaymentPayer"
    refs: EdgeRef


class PaymentPayee(BaseModel):
    kind: Literal["PaymentPayee"] = "PaymentPayee"
    refs: EdgeRef


class PaymentSubject(BaseModel):
    kind: Literal["PaymentSubject"] = "PaymentSubject"
    refs: EdgeRef


BINARY_SCHEMA = GraphSchema(
    node_types=(Person, Company),
    edge_types=(Employment,),
    patterns=((Person, Employment, Company),),
)

NARY_SCHEMA = GraphSchema(
    node_types=(Person, Company, Payment),
    edge_types=(PaymentPayer, PaymentPayee, PaymentSubject),
    patterns=(
        (Payment, PaymentPayer, Person),
        (Payment, PaymentPayee, Person),
        (Payment, PaymentSubject, Company),
    ),
)


# ---------------------------------------------------------------------------
# Mention builders
# ---------------------------------------------------------------------------


def _node_mention(payload, mention_id: str, source_id: str = "src-1") -> Mention:
    return Mention(
        mention_id=mention_id,
        source_id=source_id,
        source_hash="h",
        content_hash="c",
        payload=payload,
    )


def _edge_mention(edge_payload, mention_id: str, source_id: str = "src-1") -> Mention:
    return Mention(
        mention_id=mention_id,
        source_id=source_id,
        source_hash="h",
        content_hash="c",
        payload=edge_payload,
    )


# ---------------------------------------------------------------------------
# 1. Node resolution per type
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_node_resolution_per_type_separates_node_types(mstore):
    """Person mentions cluster separately from Company mentions."""
    await mstore.upsert(
        [
            _node_mention(Person(full_name="Alice"), "p-1"),
            _node_mention(Person(full_name="Alice"), "p-2"),  # duplicate Person → cluster
            _node_mention(Company(name="Acme"), "c-1"),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()

    pipeline = KnowledgeGraphResolutionPipeline(
        schema=BINARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,  # exact-key seed handles the Alice duplicate
    )
    result = await pipeline.resolve()

    by_kind: dict[str, list] = {}
    for e in result.node_entities:
        by_kind.setdefault(e.payload.kind, []).append(e)

    assert len(by_kind["Person"]) == 1, "two Alice mentions should cluster into one Person"
    assert len(by_kind["Company"]) == 1
    assert {mid for e in by_kind["Person"] for mid in e.member_mention_ids} == {"p-1", "p-2"}


# ---------------------------------------------------------------------------
# 2-3. Edge endpoint rewrite + 1:1 promotion (default)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edge_endpoint_rewrite_to_entity_ids(mstore):
    await mstore.upsert(
        [
            _node_mention(Person(full_name="Alice"), "p-1"),
            _node_mention(Company(name="Acme"), "c-1"),
            _edge_mention(
                Employment(
                    refs=EdgeRef(source_mention_id="p-1", target_mention_id="c-1"),
                    role="Engineer",
                ),
                "e-1",
            ),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()

    pipeline = KnowledgeGraphResolutionPipeline(
        schema=BINARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,
    )
    result = await pipeline.resolve()

    person_entity = next(e for e in result.node_entities if e.payload.kind == "Person")
    company_entity = next(e for e in result.node_entities if e.payload.kind == "Company")
    employment_entity = next(e for e in result.edge_entities if e.payload.kind == "Employment")

    assert employment_entity.payload.refs.source_mention_id == person_entity.entity_id
    assert employment_entity.payload.refs.target_mention_id == company_entity.entity_id


@pytest.mark.anyio
async def test_edges_promoted_one_to_one_by_default(mstore):
    """Two docs both stating 'Alice works at Acme' → two distinct Employment entities (no dedup)."""
    await mstore.upsert(
        [
            # doc 1
            _node_mention(Person(full_name="Alice"), "p-1a", source_id="doc1"),
            _node_mention(Company(name="Acme"), "c-1a", source_id="doc1"),
            _edge_mention(
                Employment(refs=EdgeRef(source_mention_id="p-1a", target_mention_id="c-1a")),
                "e-1",
                source_id="doc1",
            ),
            # doc 2
            _node_mention(Person(full_name="Alice"), "p-1b", source_id="doc2"),
            _node_mention(Company(name="Acme"), "c-1b", source_id="doc2"),
            _edge_mention(
                Employment(refs=EdgeRef(source_mention_id="p-1b", target_mention_id="c-1b")),
                "e-2",
                source_id="doc2",
            ),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()

    pipeline = KnowledgeGraphResolutionPipeline(
        schema=BINARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,
    )
    result = await pipeline.resolve()

    # Nodes deduplicate (exact-key on identical Alice / Acme strings).
    assert len([e for e in result.node_entities if e.payload.kind == "Person"]) == 1
    assert len([e for e in result.node_entities if e.payload.kind == "Company"]) == 1

    # Edges DO NOT deduplicate by default — both attestations stay.
    employment_entities = [e for e in result.edge_entities if e.payload.kind == "Employment"]
    assert len(employment_entities) == 2

    # Each edge entity carries its own single-source provenance.
    assert {e.source_ids[0] for e in employment_entities} == {"doc1", "doc2"}


# ---------------------------------------------------------------------------
# 4. Provenance retained on edges
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edge_provenance_retained_after_rewrite(mstore):
    await mstore.upsert(
        [
            _node_mention(Person(full_name="Alice"), "p-1", source_id="src-A"),
            _node_mention(Company(name="Acme"), "c-1", source_id="src-A"),
            _edge_mention(
                Employment(refs=EdgeRef(source_mention_id="p-1", target_mention_id="c-1")),
                "e-1",
                source_id="src-A",
            ),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()
    pipeline = KnowledgeGraphResolutionPipeline(
        schema=BINARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,
    )
    result = await pipeline.resolve()
    edge = result.edge_entities[0]
    assert edge.source_ids == ["src-A"]
    assert edge.member_mention_ids == ["e-1"]


# ---------------------------------------------------------------------------
# 5. Orphan edges surfaced
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orphan_edge_surfaced_not_dropped(mstore):
    """An edge whose endpoint mention isn't resolvable lands in orphan_edges."""
    await mstore.upsert(
        [
            _node_mention(Person(full_name="Alice"), "p-1"),
            # No Company mention — c-99 is orphan
            _edge_mention(
                Employment(refs=EdgeRef(source_mention_id="p-1", target_mention_id="c-99")),
                "e-1",
            ),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()
    pipeline = KnowledgeGraphResolutionPipeline(
        schema=BINARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,
    )
    result = await pipeline.resolve()
    assert result.edge_entities == []
    assert len(result.orphan_edges) == 1
    assert result.orphan_edges[0].mention_id == "e-1"


def test_rewrite_edge_endpoints_pure_function():
    """The rewrite helper is testable in isolation."""
    edge = _edge_mention(
        Employment(refs=EdgeRef(source_mention_id="p-1", target_mention_id="c-1")),
        "e-1",
    )
    node_map = {"p-1": "node-entity-A", "c-1": "node-entity-B"}
    rewritten, orphans = _rewrite_edge_endpoints([edge], node_map)
    assert not orphans
    assert rewritten[0].payload.refs.source_mention_id == "node-entity-A"
    assert rewritten[0].payload.refs.target_mention_id == "node-entity-B"


# ---------------------------------------------------------------------------
# 6. N-ary reified-node end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_nary_reified_node_three_edges(mstore):
    """A Payment node with three role-typed edges to Alice, Bob, and Acme."""
    await mstore.upsert(
        [
            _node_mention(Person(full_name="Alice"), "p-1"),
            _node_mention(Person(full_name="Bob"), "p-2"),
            _node_mention(Company(name="Acme"), "c-1"),
            _node_mention(Payment(amount="$5000"), "pay-1"),
            _edge_mention(
                PaymentPayer(refs=EdgeRef(source_mention_id="pay-1", target_mention_id="p-1")),
                "ep-1",
            ),
            _edge_mention(
                PaymentPayee(refs=EdgeRef(source_mention_id="pay-1", target_mention_id="p-2")),
                "ep-2",
            ),
            _edge_mention(
                PaymentSubject(refs=EdgeRef(source_mention_id="pay-1", target_mention_id="c-1")),
                "ep-3",
            ),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()
    pipeline = KnowledgeGraphResolutionPipeline(
        schema=NARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,
    )
    result = await pipeline.resolve()

    # 4 node entities (Alice, Bob, Acme, Payment)
    assert len(result.node_entities) == 4
    # 3 edge entities (one of each edge kind)
    edge_kinds = {e.payload.kind for e in result.edge_entities}
    assert edge_kinds == {"PaymentPayer", "PaymentPayee", "PaymentSubject"}

    # All three edges' source endpoints point at the Payment node entity_id
    payment_entity = next(e for e in result.node_entities if e.payload.kind == "Payment")
    for edge in result.edge_entities:
        assert edge.payload.refs.source_mention_id == payment_entity.entity_id


# ---------------------------------------------------------------------------
# 7. Opt-in edge dedup via edge_identity_text_fns
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_opt_in_edge_dedup_collapses_duplicates(mstore):
    """When edge_identity_text_fns provides a fn for Employment, repeated edges collapse."""
    await mstore.upsert(
        [
            _node_mention(Person(full_name="Alice"), "p-1a", source_id="doc1"),
            _node_mention(Company(name="Acme"), "c-1a", source_id="doc1"),
            _edge_mention(
                Employment(refs=EdgeRef(source_mention_id="p-1a", target_mention_id="c-1a")),
                "e-1",
                source_id="doc1",
            ),
            _node_mention(Person(full_name="Alice"), "p-1b", source_id="doc2"),
            _node_mention(Company(name="Acme"), "c-1b", source_id="doc2"),
            _edge_mention(
                Employment(refs=EdgeRef(source_mention_id="p-1b", target_mention_id="c-1b")),
                "e-2",
                source_id="doc2",
            ),
        ]
    )
    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()
    # The identity_text_fn keys on the (already-rewritten) endpoint entity ids — same endpoints
    # → identical text → exact-key seed merges them.
    pipeline = KnowledgeGraphResolutionPipeline(
        schema=BINARY_SCHEMA,
        mention_store=mstore,
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=merge_all_reviewer,
        edge_identity_text_fns={
            Employment: lambda e: f"{e.refs.source_mention_id}->{e.refs.target_mention_id}",
        },
    )
    result = await pipeline.resolve()
    employments = [e for e in result.edge_entities if e.payload.kind == "Employment"]
    assert len(employments) == 1, "edge dedup should collapse the two attestations into one"
    assert set(employments[0].source_ids) == {"doc1", "doc2"}
    assert set(employments[0].member_mention_ids) == {"e-1", "e-2"}


# ---------------------------------------------------------------------------
# 8. Result shape sanity
# ---------------------------------------------------------------------------


def test_resolution_result_default_fields():
    r = KGResolutionResult()
    assert r.node_entities == []
    assert r.edge_entities == []
    assert r.orphan_edges == []
    assert r.pending == []
    assert r.iterations == 0


# ---------------------------------------------------------------------------
# 9. mint_entity_id — one shared minting function (F12)
# ---------------------------------------------------------------------------


def test_mint_entity_id_is_order_invariant():
    """Both resolution paths mint entity ids through one function; it sorts its members."""
    from ragdoc.extraction.entity import mint_entity_id

    assert mint_entity_id(["b", "a"]) == mint_entity_id(["a", "b"])
    assert mint_entity_id(["a"]) != mint_entity_id(["a", "b"])
