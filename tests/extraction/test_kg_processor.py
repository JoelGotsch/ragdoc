"""Tests for KnowledgeGraphProcessor — multi-type extraction with chunk-local-id rewrite.

The processor writes ONE merged list to ``document.metadata[metadata_key]`` (nodes first, then
edges); downstream code distinguishes them by ``payload.kind``. This shape satisfies the same
interface as ``StructuredExtractionProcessor`` (``metadata_key`` + ``payload_model`` +
``process()``), so the existing ``MentionStorePipeline`` composes with it unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.kg_processor import (
    KnowledgeGraphProcessor,
    build_graph_batch_model,
)
from ragdoc.extraction.processor import ExtractionSettings
from ragdoc.extraction.schema import EdgeRef, GraphSchema

# ---------------------------------------------------------------------------
# Sample schema (shared across tests)
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """A real person mentioned in the document."""

    kind: Literal["Person"] = "Person"
    full_name: str


class Company(BaseModel):
    """A legal entity."""

    kind: Literal["Company"] = "Company"
    name: str


class Employment(BaseModel):
    """The relationship: a Person employed at a Company."""

    kind: Literal["Employment"] = "Employment"
    refs: EdgeRef
    role: str | None = None


SCHEMA = GraphSchema(
    node_types=(Person, Company),
    edge_types=(Employment,),
    patterns=((Person, Employment, Company),),
)


# ---------------------------------------------------------------------------
# Mock client helpers
# ---------------------------------------------------------------------------


def make_kg_client(schema: GraphSchema, batches: list[tuple[list, list]]) -> MagicMock:
    """Mock client whose successive parse() calls yield successive (nodes, edges) GraphBatches.

    Each batch's nodes/edges are raw dicts validated through the runtime GraphBatch model.
    """
    graph_batch_cls, _ = build_graph_batch_model(schema)
    calls = {"i": 0}

    async def _parse(**_kwargs):  # type: ignore[no-untyped-def]
        i = min(calls["i"], len(batches) - 1)
        calls["i"] += 1
        nodes, edges = batches[i]
        batch = graph_batch_cls.model_validate({"node_mentions": nodes, "edge_mentions": edges})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(side_effect=_parse)
    return client


def make_doc(body: str = "Alice works at Acme.") -> Document:
    return Document(
        title="t",
        elements=[Heading(innerhtml="t", level=1), Paragraph(html_content=f"<p>{body}</p>")],
    )


def _split_by_kind(mentions: list[dict], schema: GraphSchema) -> tuple[list[dict], list[dict]]:
    """Helper: split the merged list into (node_mentions, edge_mentions) by payload.kind."""
    node_kinds = {t.model_fields["kind"].default for t in schema.node_types}
    edge_kinds = {t.model_fields["kind"].default for t in schema.edge_types}
    nodes = [m for m in mentions if m["payload"]["kind"] in node_kinds]
    edges = [m for m in mentions if m["payload"]["kind"] in edge_kinds]
    assert len(nodes) + len(edges) == len(mentions), "unknown kind in merged list"
    return nodes, edges


# ---------------------------------------------------------------------------
# 0. StructuredExtractionProcessor interface compatibility (the point of variant a)
# ---------------------------------------------------------------------------


def test_satisfies_structured_extraction_processor_interface():
    """KnowledgeGraphProcessor must look like StructuredExtractionProcessor to MentionStorePipeline:
    expose `metadata_key`, `payload_model`, and `process()`. The merged-key shape (variant a) is
    what makes this possible."""
    processor = KnowledgeGraphProcessor(SCHEMA, client=MagicMock(), model="m")
    assert processor.metadata_key == "kg_mentions"  # single key (configurable)
    payload_model = processor.payload_model
    # The payload model is the merged discriminated union over node + edge types — Pydantic can
    # validate any node or edge dict through it.
    from pydantic import TypeAdapter

    adapter = TypeAdapter(payload_model)
    p = adapter.validate_python({"kind": "Person", "full_name": "Alice"})
    e = adapter.validate_python({"kind": "Employment", "refs": {"source_mention_id": "x", "target_mention_id": "y"}})
    assert isinstance(p, Person)
    assert isinstance(e, Employment)


def test_custom_metadata_key():
    processor = KnowledgeGraphProcessor(SCHEMA, client=MagicMock(), model="m", metadata_key="my_kg")
    assert processor.metadata_key == "my_kg"


# ---------------------------------------------------------------------------
# 1. One .parse() call returns both nodes and edges in the merged list
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_one_parse_call_returns_merged_list():
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [
                    {
                        "kind": "Employment",
                        "refs": {"source_mention_id": "n0", "target_mention_id": "n1"},
                        "role": "Engineer",
                    }
                ],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    doc = await processor.process(make_doc())

    merged = doc.metadata["kg_mentions"]
    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 2
    assert len(edges) == 1
    # Nodes come first in the merged list — order matters for the contiguous ordinal scheme.
    assert merged[0]["payload"]["kind"] in {"Person", "Company"}
    assert merged[-1]["payload"]["kind"] == "Employment"
    assert {n["payload"]["kind"] for n in nodes} == {"Person", "Company"}
    assert edges[0]["payload"]["role"] == "Engineer"


@pytest.mark.anyio
async def test_contiguous_ordinals_across_merged_list():
    """Ordinals across the merged node + edge list are 0..N-1 (single position-within-source)."""
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    doc = await processor.process(make_doc())

    ordinals = [m["ordinal"] for m in doc.metadata["kg_mentions"]]
    assert ordinals == [0, 1, 2]


# ---------------------------------------------------------------------------
# 2. Local-id → mention-id rewrite
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_local_id_rewritten_to_real_mention_id():
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    doc = await processor.process(make_doc())

    nodes, edges = _split_by_kind(doc.metadata["kg_mentions"], SCHEMA)
    # nth node corresponds to local_id n{i}
    by_local = {f"n{i}": n["mention_id"] for i, n in enumerate(nodes)}

    edge = edges[0]
    assert edge["payload"]["refs"]["source_mention_id"] == by_local["n0"]
    assert edge["payload"]["refs"]["target_mention_id"] == by_local["n1"]
    # Sanity: chunk-local strings are gone.
    assert edge["payload"]["refs"]["source_mention_id"] != "n0"


# ---------------------------------------------------------------------------
# 3. Edge pointing at non-existent local id → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_edge_with_unknown_local_id_raises():
    client = make_kg_client(
        SCHEMA,
        [
            (
                [{"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}}],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n9"}}],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    with pytest.raises(ValueError, match="unknown target local_id"):
        await processor.process(make_doc())


@pytest.mark.anyio
async def test_duplicate_local_id_raises():
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n0", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    with pytest.raises(ValueError, match="Duplicate node local_id"):
        await processor.process(make_doc())


# ---------------------------------------------------------------------------
# 4. Discriminator round-trip on the LLM-facing schema (GraphBatch)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_discriminator_invalid_kind_validation_error():
    """A `kind` that isn't a registered variant should fail validation on the LLM-facing schema."""
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)
    with pytest.raises(Exception):  # ValidationError
        graph_batch_cls.model_validate(
            {
                "node_mentions": [{"local_id": "n0", "node": {"kind": "Alien", "full_name": "x"}}],
                "edge_mentions": [],
            }
        )


# ---------------------------------------------------------------------------
# 5. Provenance + metadata copy
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_provenance_and_metadata_copied():
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "A"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "B"}},
                ],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    doc = make_doc()
    doc.source_id = "src-1"
    doc.source_hash = "filehash"
    doc.source_path = "/path/to/file.txt"
    doc.metadata = {"split_sequence": 2, "split_total": 5, "custom": "x"}

    await processor.process(doc)

    for m in doc.metadata["kg_mentions"]:
        assert m["source_id"] == "src-1"
        assert m["source_hash"] == "filehash"
        assert m["source_path"] == "/path/to/file.txt"
        assert m["content_hash"] is not None
        # metadata snapshot taken BEFORE writing our key — must contain custom but NOT
        # the merged kg_mentions key itself (no recursive embedding).
        assert m["metadata"]["custom"] == "x"
        assert m["metadata"]["split_sequence"] == 2
        assert "kg_mentions" not in m["metadata"]


# ---------------------------------------------------------------------------
# 6. Idempotency guard (single key)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_idempotency_skips_when_key_set():
    client = make_kg_client(SCHEMA, [([], [])])
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    doc = make_doc()
    doc.metadata = {"kg_mentions": [{"placeholder": True}]}

    await processor.process(doc)
    assert client.beta.chat.completions.parse.call_count == 0
    assert doc.metadata["kg_mentions"] == [{"placeholder": True}]


@pytest.mark.anyio
async def test_overwrite_true_forces_extraction():
    client = make_kg_client(SCHEMA, [([], [])])
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m", overwrite=True)
    doc = make_doc()
    doc.metadata = {"kg_mentions": [{"x": 1}]}

    await processor.process(doc)
    assert client.beta.chat.completions.parse.call_count == 1
    assert doc.metadata["kg_mentions"] == []  # overwritten


# ---------------------------------------------------------------------------
# 7. min_tokens skip
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_min_tokens_skips_extraction():
    client = make_kg_client(SCHEMA, [([], [])])
    settings = ExtractionSettings(min_tokens=10_000)
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m", settings=settings)
    doc = make_doc("tiny")

    await processor.process(doc)
    assert client.beta.chat.completions.parse.call_count == 0
    assert doc.metadata["kg_mentions"] == []


# ---------------------------------------------------------------------------
# 8. Gleaning toggle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gleaning_runs_second_call_and_merges():
    client = make_kg_client(
        SCHEMA,
        [
            # First call
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
            ),
            # Second call (gleaning): one new node, one new edge from it to an existing node
            (
                [{"local_id": "n0", "node": {"kind": "Person", "full_name": "Bob"}}],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
            ),
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m", gleaning=True)
    doc = await processor.process(make_doc())

    assert client.beta.chat.completions.parse.call_count == 2
    nodes, edges = _split_by_kind(doc.metadata["kg_mentions"], SCHEMA)
    assert len(nodes) == 3  # 2 from first + 1 from gleaning
    assert len(edges) == 2

    # The second edge's source endpoint must point at the newly-added Bob's mention_id.
    bob_mention_id = nodes[2]["mention_id"]
    acme_mention_id = nodes[1]["mention_id"]
    assert edges[1]["payload"]["refs"]["source_mention_id"] == bob_mention_id
    assert edges[1]["payload"]["refs"]["target_mention_id"] == acme_mention_id


@pytest.mark.anyio
async def test_gleaning_off_by_default():
    client = make_kg_client(SCHEMA, [([], [])])
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    await processor.process(make_doc())
    assert client.beta.chat.completions.parse.call_count == 1  # only the primary call


# ---------------------------------------------------------------------------
# 9. Ordinal disambiguation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ordinal_disambiguates_identical_payloads():
    """Two same-content node mentions in one split get distinct mention_ids via ordinal."""
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Person", "full_name": "Alice"}},
                ],
                [],
            )
        ],
    )
    processor = KnowledgeGraphProcessor(SCHEMA, client=client, model="m")
    doc = await processor.process(make_doc())

    ids = [m["mention_id"] for m in doc.metadata["kg_mentions"]]
    assert len(ids) == 2 and ids[0] != ids[1]
