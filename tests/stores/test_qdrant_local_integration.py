"""Integration tests against Qdrant's in-process local mode (``AsyncQdrantClient(":memory:")``).

Real Qdrant id validation and filter evaluation — no stubs, no server. Covers:

- pipeline-minted default chunk ids are valid Qdrant point ids (regression: the former
  sha256-hexdigest ids were rejected by Qdrant, which accepts only UUIDs or unsigned ints);
- QdrantEntityStore.query date-overlap parity with LocalEntityStore, including half-bounded
  and undated entities (missing ``date_start``/``date_end`` payload keys = unbounded);
- every store's ``create`` classmethod is idempotent (the local client reports an existing
  collection with ``ValueError``, not the HTTP 409 the server returns).
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("qdrant_client", reason="qdrant extra not installed")

from qdrant_client import AsyncQdrantClient

from ragdoc.chunking import Chunk
from ragdoc.chunking.provenance import mint_chunk_id
from ragdoc.document import Document, Paragraph
from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.entity import Entity, LocalEntityStore
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.query import DateRange, EntityQuery
from ragdoc.extraction.schema import EdgeRef, GraphSchema
from ragdoc.integrations.document_stores.qdrant import QdrantDocumentStore
from ragdoc.integrations.entity_stores.qdrant import QdrantEntityStore
from ragdoc.integrations.graph_stores.qdrant import QdrantGraphStore
from ragdoc.integrations.mention_stores.qdrant import QdrantMentionStore
from ragdoc.integrations.vector_stores.qdrant import QdrantVectorStore

# ---------------------------------------------------------------------------
# F1 regression: default chunk ids are valid Qdrant point ids
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pipeline_minted_chunk_id_is_accepted_by_qdrant():
    client = AsyncQdrantClient(":memory:")
    store = await QdrantVectorStore.create(client, "chunks", vector_size=3)

    chunk = Chunk(
        id=mint_chunk_id("report.pdf", 1, 0, "content-hash"),
        prompt_content="p",
        embedding_content="e",
        named_embeddings={"dense": [0.1, 0.2, 0.3]},
        source_id="report.pdf",
        source_hash="abc",
    )
    stored_ids = await store.upsert([chunk])  # raises ValueError on a non-UUID id

    assert stored_ids == [chunk.id]
    assert await store.list_source_ids() == {"report.pdf"}
    # id is stable: re-minting the same identity re-upserts the same point (no duplicates)
    chunk2 = chunk.model_copy(update={"id": mint_chunk_id("report.pdf", 1, 0, "content-hash")})
    await store.upsert([chunk2])
    count = await client.count("chunks")
    assert count.count == 1


# ---------------------------------------------------------------------------
# F5: entity date-filter parity between QdrantEntityStore and LocalEntityStore
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    """Minimal payload for entity-store parity tests."""

    title: str = Field(description="Event title.")


def _entity(entity_id: str, edtf: str | None, *, dated: bool = True) -> Entity[_Event]:
    date = FuzzyDate(original_text=edtf or "sometime", edtf=edtf, precision="YEAR") if dated else None
    return Entity[_Event](
        entity_id=entity_id,
        payload=_Event(title=f"E-{entity_id}"),
        member_mention_ids=[f"{entity_id}-m"],
        source_ids=["s1"],
        date=date,
    )


@pytest.fixture
def parity_entities() -> list[Entity[_Event]]:
    return [
        _entity("bounded", "1994"),  # [1994-01-01, 1995-01-01)
        _entity("open-start", "../1989"),  # start unbounded, ends 1990-01-01
        _entity("open-end", "1994/.."),  # starts 1994-01-01, end unbounded
        _entity("interval-unknown", None),  # FuzzyDate present but no parseable interval
        _entity("undated", None, dated=False),  # no date at all
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "date_range",
    [
        pytest.param(DateRange(dt.date(1990, 1, 1), dt.date(1996, 1, 1)), id="fully-bounded-window"),
        pytest.param(DateRange(start=dt.date(2000, 1, 1)), id="open-ended-window"),
        pytest.param(DateRange(end=dt.date(1980, 1, 1)), id="open-start-window"),
        pytest.param(DateRange(), id="unbounded-window"),
    ],
)
async def test_query_date_overlap_parity_with_local_store(tmp_path, parity_entities, date_range):
    local = LocalEntityStore(tmp_path, _Event)
    await local.upsert(parity_entities)

    client = AsyncQdrantClient(":memory:")
    qdrant = await QdrantEntityStore.create(client, "entities", _Event)
    await qdrant.upsert(parity_entities)

    query = EntityQuery(date=date_range)
    local_ids = {e.entity_id for e in await local.query(query)}
    qdrant_ids = {e.entity_id for e in await qdrant.query(query)}

    assert qdrant_ids == local_ids


@pytest.mark.anyio
async def test_query_half_bounded_entity_matches_open_window(tmp_path, parity_entities):
    """The reported bug: a half-bounded entity must satisfy the bound it lacks (missing key = unbounded)."""
    client = AsyncQdrantClient(":memory:")
    qdrant = await QdrantEntityStore.create(client, "entities", _Event)
    await qdrant.upsert(parity_entities)

    # "1994/.." has no date_end payload key; a window requiring date_end > 2000 must still match it.
    hits = {e.entity_id for e in await qdrant.query(EntityQuery(date=DateRange(start=dt.date(2000, 1, 1))))}
    assert "open-end" in hits
    assert "bounded" not in hits  # genuinely ends 1995 → correctly excluded
    assert "undated" not in hits  # no date at all → excluded by any date filter (local parity)


@pytest.mark.anyio
async def test_query_where_filter_parity(tmp_path, parity_entities):
    local = LocalEntityStore(tmp_path, _Event)
    await local.upsert(parity_entities)
    client = AsyncQdrantClient(":memory:")
    qdrant = await QdrantEntityStore.create(client, "entities", _Event)
    await qdrant.upsert(parity_entities)

    query = EntityQuery(date=DateRange(dt.date(1990, 1, 1), dt.date(1996, 1, 1)), where={"title": "E-bounded"})
    assert (
        {e.entity_id for e in await qdrant.query(query)}
        == {e.entity_id for e in await local.query(query)}
        == {"bounded"}
    )


# ---------------------------------------------------------------------------
# QdrantDocumentStore round trip against the real payload it writes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("source_hash", ["abc123", None])
async def test_document_store_round_trips_the_document_it_wrote(source_hash: str | None):
    """The stored payload carries extra top-level hash keys; reading must still yield the Document."""
    from ragdoc.document import Document, Paragraph
    from ragdoc.integrations.document_stores.qdrant import QdrantDocumentStore

    client = AsyncQdrantClient(":memory:")
    store = await QdrantDocumentStore.create(client, "docs")
    document = Document(
        elements=[Paragraph(html="<p>hello</p>")],
        source_id="a.html",
        source_hash=source_hash,
        metadata={"filename": "a.html"},
    )

    await store.upsert([document])
    loaded = await store.get_document("a.html")

    assert loaded == document
    assert loaded is not None
    assert loaded.source_hash == source_hash
    assert loaded.content_hash() == document.content_hash()


# ---------------------------------------------------------------------------
# create() is idempotent on the local client (and keeps the stored data)
# ---------------------------------------------------------------------------


class _Person(BaseModel):
    kind: Literal["Person"] = "Person"
    full_name: str = Field(description="Full name.")


class _Company(BaseModel):
    kind: Literal["Company"] = "Company"
    name: str = Field(description="Company name.")


class _Employment(BaseModel):
    kind: Literal["Employment"] = "Employment"
    refs: EdgeRef = Field(description="Source and target node refs.")


_SCHEMA = GraphSchema(
    node_types=(_Person, _Company), edge_types=(_Employment,), patterns=((_Person, _Employment, _Company),)
)


async def _embed_payloads(payloads: list[BaseModel]) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in payloads]


@pytest.mark.anyio
async def test_vector_store_create_twice_keeps_the_collection():
    client = AsyncQdrantClient(":memory:")
    first = await QdrantVectorStore.create(client, "chunks", vector_size=3)
    await first.upsert(
        [
            Chunk(
                id=mint_chunk_id("a.txt", 1, 0, "c"),
                prompt_content="p",
                embedding_content="e",
                named_embeddings={"dense": [0.1, 0.2, 0.3]},
                source_id="a.txt",
                source_hash="h",
            )
        ]
    )

    second = await QdrantVectorStore.create(client, "chunks", vector_size=3)  # raised ValueError before

    assert await second.list_source_ids() == {"a.txt"}  # same collection, not re-created


@pytest.mark.anyio
async def test_document_store_create_twice_keeps_the_collection():
    client = AsyncQdrantClient(":memory:")
    first = await QdrantDocumentStore.create(client, "documents")
    await first.upsert([Document(elements=[Paragraph(html="<p>kept</p>")], source_id="a.txt")])

    second = await QdrantDocumentStore.create(client, "documents")

    assert await second.list_source_ids() == {"a.txt"}


@pytest.mark.anyio
async def test_mention_store_create_twice_keeps_the_collection():
    client = AsyncQdrantClient(":memory:")
    first = await QdrantMentionStore.create(client, "mentions", _Event, _embed_payloads, vector_size=3)
    mention = Mention[_Event](
        mention_id="m1", source_id="a.txt", source_hash="h", content_hash="c", payload=_Event(title="kept")
    )
    await first.upsert([mention])

    second = await QdrantMentionStore.create(client, "mentions", _Event, _embed_payloads, vector_size=3)

    assert set(await second.list_source_state()) == {"a.txt"}


@pytest.mark.anyio
async def test_entity_store_create_twice_keeps_the_collection():
    client = AsyncQdrantClient(":memory:")
    first = await QdrantEntityStore.create(client, "entities", _Event)
    await first.upsert([_entity("kept", "1994")])

    second = await QdrantEntityStore.create(client, "entities", _Event)

    assert [e.entity_id for e in await second.query(EntityQuery())] == ["kept"]


@pytest.mark.anyio
async def test_graph_store_create_twice_keeps_both_collections():
    client = AsyncQdrantClient(":memory:")
    await QdrantGraphStore.create(client, "graph", _SCHEMA)

    await QdrantGraphStore.create(client, "graph", _SCHEMA)

    assert await client.collection_exists("graph_nodes")
    assert await client.collection_exists("graph_edges")
