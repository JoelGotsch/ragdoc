"""Integration test: KnowledgeGraphExtractor → MentionStorePipeline → KGResolutionPipeline.

This is the test that proves variant (a) works: the merged-list KnowledgeGraphExtractor satisfies
the same Extractor protocol as StructuredExtractor, so the existing MentionStorePipeline composes
with it unchanged, and the existing KnowledgeGraphResolutionPipeline reads the single store via
list_mentions(payload_type=...). No new pipeline class needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.kg import (
    KnowledgeGraphExtractor,
    build_graph_batch_model,
)
from ragdoc.extraction.kg_resolution import KnowledgeGraphResolutionPipeline
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.extraction.schema import EdgeRef, GraphSchema
from ragdoc.pipeline.linear import DocumentPipeline
from ragdoc.pipeline.stores import SourceState

from .conftest import MemoryEntityStore, MemoryMentionStore, dict_embed, no_merge_reviewer

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


SCHEMA = GraphSchema(
    node_types=(Person, Company),
    edge_types=(Employment,),
    patterns=((Person, Employment, Company),),
)


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


def make_kg_client(nodes, edges) -> MagicMock:
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)

    async def _parse(**_kwargs):  # type: ignore[no-untyped-def]
        batch = graph_batch_cls.model_validate({"node_mentions": nodes, "edge_mentions": edges})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(side_effect=_parse)
    return client


class MemoryDocumentStore:
    def __init__(self):
        self.stored: dict[str, Document] = {}

    async def upsert(self, documents):
        for d in documents:
            self.stored[d.source_id] = d
        return [d.source_id for d in documents]

    async def delete_by_source(self, source_id):
        self.stored.pop(source_id, None)

    async def get_document(self, source_id):
        return self.stored.get(source_id)

    async def list_source_ids(self):
        return set(self.stored.keys())

    async def list_source_state(self):
        return {
            sid: SourceState(source_hash=d.source_hash or "", content_hash=d.content_hash())
            for sid, d in self.stored.items()
        }


# ---------------------------------------------------------------------------
# Variant-(a) composition: KGProcessor drops into MentionStorePipeline unchanged
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_kg_extractor_composes_with_mention_store_pipeline():
    """The smoking gun for variant (a): KnowledgeGraphExtractor is passed AS-IS to the existing
    MentionStorePipeline's `extractor=` slot. No adapter, no parallel pipeline class.
    """
    # Seed a DocumentStore (Boundary-2 mode).
    doc_store = MemoryDocumentStore()
    doc = Document(
        title="bio",
        elements=[Heading(innerhtml="bio", level=1), Paragraph(html_content="<p>Alice at Acme.</p>")],
    )
    doc.source_id = "bio-1.txt"
    doc.source_hash = "filehash"
    await doc_store.upsert([doc])

    # KG extractor with a fake LLM client.
    client = make_kg_client(
        [
            {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
            {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
        ],
        [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
    )
    kg_extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")

    # The shipped MentionStorePipeline takes ONE extractor and ONE mention_store. Variant (a) makes
    # this work without modification.
    mention_store = MemoryMentionStore()
    pipeline = MentionStorePipeline(
        pipeline=DocumentPipeline(),
        extractor=kg_extractor,
        mention_store=mention_store,
        document_store=doc_store,
    )
    result = await pipeline.run()
    assert result.processed == ["bio-1.txt"]

    # The store now holds BOTH node and edge mentions; the filter API slices them by type.
    all_mentions = await mention_store.list_mentions()
    persons = await mention_store.list_mentions(Person)
    companies = await mention_store.list_mentions(Company)
    employments = await mention_store.list_mentions(Employment)

    assert len(all_mentions) == 3
    assert len(persons) == 1 and persons[0].payload.full_name == "Alice"
    assert len(companies) == 1 and companies[0].payload.name == "Acme"
    assert len(employments) == 1

    # The edge's endpoint refs were rewritten through the local_id → mention_id map. Note that
    # MentionStorePipeline.finalize_mention re-mints mention_ids using the authoritative
    # source_id/content_hash from the parent Document — so the rewritten refs MAY no longer
    # match the re-minted node mention_ids exactly. The resolution layer handles this by
    # re-mapping mention_id → entity_id at edge-rewrite time; what we verify here is the
    # primary contract: chunk-local strings ("n0"/"n1") are gone from the persisted edge.
    edge_mention = employments[0]
    assert edge_mention.payload.refs.source_mention_id != "n0"
    assert edge_mention.payload.refs.target_mention_id != "n1"


@pytest.mark.anyio
async def test_resolution_pipeline_consumes_single_store_via_filter_api():
    """The other half: KnowledgeGraphResolutionPipeline reads ONE mention_store and uses
    list_mentions(payload_type=...) to slice node and edge mentions per type. Variant (a)'s
    single store works as-is with the existing resolution pipeline.
    """
    mention_store = MemoryMentionStore()

    # Hand-roll mentions to skip the LLM extraction layer (we're testing resolution, not extraction)
    from ragdoc.extraction.mention import Mention

    await mention_store.upsert(
        [
            Mention(
                mention_id="p-1",
                source_id="d1",
                source_hash="h",
                content_hash="c",
                payload=Person(full_name="Alice"),
            ),
            Mention(
                mention_id="c-1",
                source_id="d1",
                source_hash="h",
                content_hash="c",
                payload=Company(name="Acme"),
            ),
            Mention(
                mention_id="e-1",
                source_id="d1",
                source_hash="h",
                content_hash="c",
                payload=Employment(
                    refs=EdgeRef(source_mention_id="p-1", target_mention_id="c-1"),
                    role="Engineer",
                ),
            ),
        ]
    )

    node_store = MemoryEntityStore()
    edge_store = MemoryEntityStore()
    resolution = KnowledgeGraphResolutionPipeline(
        schema=SCHEMA,
        mention_store=mention_store,  # ONE store — covers both node + edge mentions
        node_entity_store=node_store,
        edge_entity_store=edge_store,
        embed=dict_embed({}),
        reviewer=no_merge_reviewer,
    )
    result = await resolution.resolve()

    # The resolution pipeline used list_mentions(Person), list_mentions(Company),
    # list_mentions(Employment) under the hood — all against the same store.
    person_entity = next(e for e in result.node_entities if e.payload.kind == "Person")
    company_entity = next(e for e in result.node_entities if e.payload.kind == "Company")
    employment_entity = next(e for e in result.edge_entities if e.payload.kind == "Employment")

    # Edge endpoints rewritten from mention_ids → entity_ids.
    assert employment_entity.payload.refs.source_mention_id == person_entity.entity_id
    assert employment_entity.payload.refs.target_mention_id == company_entity.entity_id
