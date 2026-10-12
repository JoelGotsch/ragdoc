"""Tests for KnowledgeGraphExtractor — multi-type extraction with chunk-local-id rewrite.

``extract()`` RETURNS one merged mention list (nodes first, then edges); downstream code
distinguishes them by ``payload.kind``. It implements the same ``Extractor`` protocol as
``StructuredExtractor``, so the existing ``MentionStorePipeline`` composes with it unchanged.
Schema patterns are enforced: illegal edges are dropped (counted + warned) or raise, per
``ExtractionSettings.on_pattern_violation``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError

openai = pytest.importorskip("openai", reason="llm extra not installed")

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.kg import (
    KnowledgeGraphExtractor,
    build_graph_batch_model,
)
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.schema import EdgeRef, GraphSchema
from ragdoc.extraction.structured import ExtractionSettings

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
    client.chat.completions.parse = AsyncMock(side_effect=_parse)
    return client


def make_doc(body: str = "Alice works at Acme.") -> Document:
    return Document(
        title="t",
        elements=[Heading(html="<h1>t</h1>"), Paragraph(html=f"<p>{body}</p>")],
    )


def _split_by_kind(
    mentions: list[Mention[BaseModel]], schema: GraphSchema
) -> tuple[list[Mention[BaseModel]], list[Mention[BaseModel]]]:
    """Helper: split the merged list into (node_mentions, edge_mentions) by payload.kind."""
    node_kinds = {t.model_fields["kind"].default for t in schema.node_types}
    edge_kinds = {t.model_fields["kind"].default for t in schema.edge_types}
    nodes = [m for m in mentions if m.payload.kind in node_kinds]  # type: ignore[attr-defined]
    edges = [m for m in mentions if m.payload.kind in edge_kinds]  # type: ignore[attr-defined]
    assert len(nodes) + len(edges) == len(mentions), "unknown kind in merged list"
    return nodes, edges


# ---------------------------------------------------------------------------
# 0. Public surface: schema + payload_model kept as documented attributes
# ---------------------------------------------------------------------------


def test_construction_without_client_fails_loud():
    """Client resolution happens in __init__ (fail-loud at construction), never at extract time."""
    from ragdoc.config import RagdocConfig, configure
    from ragdoc.llm import LLMNotConfiguredError

    with configure(RagdocConfig()), pytest.raises(LLMNotConfiguredError):
        KnowledgeGraphExtractor(SCHEMA)


@pytest.mark.anyio
async def test_source_hash_none_when_document_has_none():
    """source_hash is honestly optional — never faked from the content hash."""
    client = make_kg_client(
        SCHEMA,
        [([{"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}}], [])],
    )
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    doc = make_doc()  # no source_hash set
    merged = await extractor.extract(doc)
    assert merged and all(m.source_hash is None for m in merged)
    assert all(m.content_hash == doc.content_hash() for m in merged)


def test_payload_model_is_merged_union():
    """`payload_model` stays on the concrete class (store queries / ChangeSet parametrization)."""
    extractor = KnowledgeGraphExtractor(SCHEMA, client=MagicMock(), model="m")
    from pydantic import TypeAdapter

    adapter = TypeAdapter(extractor.payload_model)
    p = adapter.validate_python({"kind": "Person", "full_name": "Alice"})
    e = adapter.validate_python({"kind": "Employment", "refs": {"source_mention_id": "x", "target_mention_id": "y"}})
    assert isinstance(p, Person)
    assert isinstance(e, Employment)
    assert extractor.schema is SCHEMA


# ---------------------------------------------------------------------------
# 1. One .parse() call returns both nodes and edges in the merged, RETURNED list
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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    doc = make_doc()
    merged = await extractor.extract(doc)

    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 2
    assert len(edges) == 1
    # Nodes come first in the merged list — order matters for the contiguous ordinal scheme.
    assert merged[0].payload.kind in {"Person", "Company"}  # type: ignore[attr-defined]
    assert merged[-1].payload.kind == "Employment"  # type: ignore[attr-defined]
    assert {n.payload.kind for n in nodes} == {"Person", "Company"}  # type: ignore[attr-defined]
    assert edges[0].payload.role == "Engineer"  # type: ignore[attr-defined]
    # extract() never touches document metadata.
    assert doc.metadata == {}


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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    merged = await extractor.extract(make_doc())
    assert [m.ordinal for m in merged] == [0, 1, 2]


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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    merged = await extractor.extract(make_doc())

    nodes, edges = _split_by_kind(merged, SCHEMA)
    # nth node corresponds to local_id n{i}
    by_local = {f"n{i}": n.mention_id for i, n in enumerate(nodes)}

    edge = edges[0]
    assert edge.payload.refs.source_mention_id == by_local["n0"]  # type: ignore[attr-defined]
    assert edge.payload.refs.target_mention_id == by_local["n1"]  # type: ignore[attr-defined]
    # Sanity: chunk-local strings are gone.
    assert edge.payload.refs.source_mention_id != "n0"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. Duplicate node local ids → ValueError (dangling edge refs: see section 13)
# ---------------------------------------------------------------------------


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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    with pytest.raises(ValueError, match="Duplicate node local_id"):
        await extractor.extract(make_doc())


# ---------------------------------------------------------------------------
# 4. Discriminator round-trip on the LLM-facing schema (GraphBatch)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_discriminator_invalid_kind_validation_error():
    """A `kind` that isn't a registered variant should fail validation on the LLM-facing schema."""
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)
    with pytest.raises(ValidationError):
        graph_batch_cls.model_validate(
            {
                "node_mentions": [{"local_id": "n0", "node": {"kind": "Alien", "full_name": "x"}}],
                "edge_mentions": [],
            }
        )


# ---------------------------------------------------------------------------
# 5. Provenance + metadata copy (document metadata untouched)
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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    doc = make_doc()
    doc.source_id = "src-1"
    doc.source_hash = "filehash"
    doc.source_path = "/path/to/file.txt"
    doc.metadata = {"split_sequence": 2, "split_total": 5, "custom": "x"}

    merged = await extractor.extract(doc)

    for m in merged:
        assert m.source_id == "src-1"
        assert m.source_hash == "filehash"
        assert m.source_path == "/path/to/file.txt"
        assert m.content_hash is not None
        assert m.metadata["custom"] == "x"
        assert m.metadata["split_sequence"] == 2
        assert "kg_mentions" not in m.metadata
        assert "mentions" not in m.metadata
    # extract() wrote nothing back
    assert doc.metadata == {"split_sequence": 2, "split_total": 5, "custom": "x"}


# ---------------------------------------------------------------------------
# 6. min_tokens / empty render gates return []
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_min_tokens_skips_extraction():
    client = make_kg_client(SCHEMA, [([], [])])
    settings = ExtractionSettings(min_tokens=10_000)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    doc = make_doc("tiny")

    merged = await extractor.extract(doc)
    assert client.chat.completions.parse.call_count == 0
    assert merged == []
    assert doc.metadata == {}


# ---------------------------------------------------------------------------
# 7. Gleaning toggle (constructor override + settings)
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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    merged = await extractor.extract(make_doc())

    assert client.chat.completions.parse.call_count == 2
    # The fake's second answer is only meaningful if the model was actually shown its first one.
    gleaning_messages = client.chat.completions.parse.call_args_list[1].kwargs["messages"]
    assert [m["role"] for m in gleaning_messages] == ["system", "user", "assistant", "user"]
    assert '"full_name":"Alice"' in gleaning_messages[2]["content"]
    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 3  # 2 from first + 1 from gleaning
    assert len(edges) == 2

    # The second edge's source endpoint must point at the newly-added Bob's mention_id.
    bob_mention_id = nodes[2].mention_id
    acme_mention_id = nodes[1].mention_id
    assert edges[1].payload.refs.source_mention_id == bob_mention_id  # type: ignore[attr-defined]
    assert edges[1].payload.refs.target_mention_id == acme_mention_id  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_gleaning_merge_with_noncontiguous_primary_ids():
    """Primary local_ids need not be contiguous n0..n{k-1}; renumbered gleaning ids must be
    minted collision-free against the actual primary id set (F3)."""
    client = make_kg_client(
        SCHEMA,
        [
            # Primary: ids n0 and n2 (the LLM skipped n1) — offset-based renumbering would
            # naively hand the gleaned node "n2" and collide.
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n2", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n2"}}],
            ),
            # Gleaning: one new node + an edge from it to the PRIMARY company (ref "n2" is not a
            # secondary id, so it must survive the renumbering untouched).
            (
                [{"local_id": "n0", "node": {"kind": "Person", "full_name": "Bob"}}],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n2"}}],
            ),
        ],
    )
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    merged = await extractor.extract(make_doc())

    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 3
    assert len(edges) == 2
    by_name = {getattr(n.payload, "full_name", None) or getattr(n.payload, "name", None): n.mention_id for n in nodes}
    assert edges[1].payload.refs.source_mention_id == by_name["Bob"]  # type: ignore[attr-defined]
    assert edges[1].payload.refs.target_mention_id == by_name["Acme"]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_gleaning_duplicate_secondary_local_ids_raise():
    """Duplicate local_ids WITHIN the gleaning batch fail loud (F4) — silent last-wins remapping
    would attach edges referencing the duplicated id to the wrong node."""
    client = make_kg_client(
        SCHEMA,
        [
            ([{"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}}], []),
            (
                [
                    {"local_id": "g0", "node": {"kind": "Person", "full_name": "Bob"}},
                    {"local_id": "g0", "node": {"kind": "Company", "name": "Beta"}},
                ],
                [],
            ),
        ],
    )
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    with pytest.raises(ValueError, match=r"[Dd]uplicate node local_id"):
        await extractor.extract(make_doc())


@pytest.mark.anyio
async def test_gleaning_off_by_default():
    client = make_kg_client(SCHEMA, [([], [])])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    await extractor.extract(make_doc())
    assert client.chat.completions.parse.call_count == 1  # only the primary call


@pytest.mark.anyio
async def test_gleaning_from_settings():
    """ExtractionSettings(gleaning=True) triggers the second parse call (env-read folded in)."""
    client = make_kg_client(SCHEMA, [([], [])])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=ExtractionSettings(gleaning=True))
    await extractor.extract(make_doc())
    assert client.chat.completions.parse.call_count == 2


# ---------------------------------------------------------------------------
# 8. Ordinal disambiguation
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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    merged = await extractor.extract(make_doc())
    ids = [m.mention_id for m in merged]
    assert len(ids) == 2 and ids[0] != ids[1]


# ---------------------------------------------------------------------------
# 9. Pattern enforcement (Layer A: prompt; Layer B: post-parse validation)
# ---------------------------------------------------------------------------

ILLEGAL_EDGE_BATCH: tuple[list, list] = (
    [
        {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
        {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
    ],
    [
        # Legal: (Person)-[Employment]->(Company)
        {"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}},
        # Illegal: (Company)-[Employment]->(Person) — not in SCHEMA.patterns
        {"kind": "Employment", "refs": {"source_mention_id": "n1", "target_mention_id": "n0"}},
    ],
)


@pytest.mark.anyio
async def test_patterns_injected_into_system_prompt():
    client = make_kg_client(SCHEMA, [([], [])])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    await extractor.extract(make_doc())
    messages = client.chat.completions.parse.call_args.kwargs["messages"]
    system = messages[0]["content"]
    assert "(Person)-[Employment]->(Company)" in system
    assert "ONLY these combinations" in system


@pytest.mark.anyio
async def test_pattern_violation_dropped_with_warning(caplog: pytest.LogCaptureFixture):
    client = make_kg_client(SCHEMA, [ILLEGAL_EDGE_BATCH])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    with caplog.at_level(logging.WARNING, logger="ragdoc.extraction.kg"):
        merged = await extractor.extract(make_doc())

    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 2  # nodes unaffected
    assert len(edges) == 1  # only the legal edge survives
    assert edges[0].payload.refs.source_mention_id == nodes[0].mention_id  # type: ignore[attr-defined]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "dropped" in r.getMessage()]
    assert len(warnings) == 1  # one WARNING per document
    message = warnings[0].getMessage()
    assert "1 edge(s)" in message
    assert "('Company', 'Employment', 'Person')" in message


@pytest.mark.anyio
async def test_pattern_violation_error_mode_raises():
    client = make_kg_client(SCHEMA, [ILLEGAL_EDGE_BATCH])
    settings = ExtractionSettings(on_pattern_violation="error")
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    with pytest.raises(ValueError, match=r"\(Company\)-\[Employment\]->\(Person\)"):
        await extractor.extract(make_doc())


@pytest.mark.anyio
async def test_pattern_conservation(caplog: pytest.LogCaptureFixture):
    """emitted + dropped == parsed — no silent third path for edges."""
    parsed_edges = len(ILLEGAL_EDGE_BATCH[1])
    client = make_kg_client(SCHEMA, [ILLEGAL_EDGE_BATCH])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    with caplog.at_level(logging.WARNING, logger="ragdoc.extraction.kg"):
        merged = await extractor.extract(make_doc())

    _, emitted_edges = _split_by_kind(merged, SCHEMA)
    warnings = [r for r in caplog.records if "dropped" in r.getMessage()]
    dropped = warnings[0].args[0] if warnings else 0  # first %d in the log record
    assert len(emitted_edges) + dropped == parsed_edges


@pytest.mark.anyio
async def test_legal_edges_pass_pattern_enforcement_silently(caplog: pytest.LogCaptureFixture):
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
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    with caplog.at_level(logging.WARNING, logger="ragdoc.extraction.kg"):
        merged = await extractor.extract(make_doc())
    assert len(merged) == 3
    assert not [r for r in caplog.records if "dropped" in r.getMessage()]


# ---------------------------------------------------------------------------
# 10. Halving knobs come from settings
# ---------------------------------------------------------------------------


def make_failing_client() -> MagicMock:
    """Every call fails with a SIZE-RELATED error (truncated output) — the kind that triggers halving."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=openai.LengthFinishReasonError(completion=MagicMock()))
    return client


@pytest.mark.anyio
async def test_halving_depth_from_settings_zero_disables_halving():
    """halving_max_depth=0 → the first failure re-raises with no halving attempt."""
    client = make_failing_client()
    settings = ExtractionSettings(max_retries=0, halving_max_depth=0, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    with pytest.raises(openai.LengthFinishReasonError):
        await extractor.extract(make_doc("one line\n\nanother line"))
    assert client.chat.completions.parse.call_count == 1


@pytest.mark.anyio
async def test_halving_depth_from_settings_one_allows_one_split():
    """halving_max_depth=1 → primary + two half-calls (each re-raising at depth 1)."""
    client = make_failing_client()
    settings = ExtractionSettings(max_retries=0, halving_max_depth=1, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    with pytest.raises(openai.LengthFinishReasonError):
        await extractor.extract(make_doc("one line\n\nanother line"))
    assert client.chat.completions.parse.call_count == 3  # primary + both halves, no deeper


@pytest.mark.anyio
async def test_halving_min_chars_from_settings_blocks_halving():
    """A short text below halving_min_chars re-raises instead of splitting."""
    client = make_failing_client()
    settings = ExtractionSettings(max_retries=0, halving_max_depth=3, halving_min_chars=100_000)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    with pytest.raises(openai.LengthFinishReasonError):
        await extractor.extract(make_doc("one line\n\nanother line"))
    assert client.chat.completions.parse.call_count == 1


# ---------------------------------------------------------------------------
# 11. Halving fires only for size-related failures
# ---------------------------------------------------------------------------

_REQUEST = httpx.Request("POST", "https://api.test/v1")


def _status_error(cls: type, status: int, message: str = "error", body: object | None = None) -> Exception:
    return cls(message, response=httpx.Response(status, request=_REQUEST), body=body)


def _auth_error() -> Exception:
    return _status_error(openai.AuthenticationError, 401, "bad key")


def _rate_limit_error() -> Exception:
    return _status_error(openai.RateLimitError, 429, "quota")


def _server_error() -> Exception:
    return _status_error(openai.InternalServerError, 500, "boom")


def _invalid_request_error() -> Exception:
    return _status_error(openai.BadRequestError, 400, "invalid schema")


def _connection_error() -> Exception:
    return openai.APIConnectionError(request=_REQUEST)


def _timeout_error() -> Exception:
    return openai.APITimeoutError(request=_REQUEST)


def _context_length_error() -> Exception:
    body = {"message": "maximum context length exceeded", "code": "context_length_exceeded"}
    return _status_error(openai.BadRequestError, 400, "too long", body)


def _length_finish_error() -> Exception:
    return openai.LengthFinishReasonError(completion=MagicMock())


def _malformed_output_error() -> Exception:
    """What ``chat.completions.parse`` raises when the completion does not validate against the schema."""
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)
    try:
        graph_batch_cls.model_validate_json('{"node_mentions": [{"local_id": "n0", "node": {"kind": "Alien"')
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    import ragdoc.llm

    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(ragdoc.llm, "_sleep", _fake_sleep)
    return slept


def big_doc() -> Document:
    """~9.5k characters in 14 paragraphs: comfortably above halving_min_chars, 3 halving levels deep."""
    para = "Alice knows Bob. " * 40
    return Document(elements=[Paragraph(html=f"<p>{para}</p>") for _ in range(14)])


def make_raising_client(make_error) -> MagicMock:  # type: ignore[no-untyped-def]
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=lambda **_kwargs: (_ for _ in ()).throw(make_error()))
    return client


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("make_error", "expected_calls"),
    [
        pytest.param(_auth_error, 1, id="401-auth"),
        pytest.param(_invalid_request_error, 1, id="400-not-about-length"),
        pytest.param(lambda: RuntimeError("boom"), 1, id="non-openai-error"),
        pytest.param(_rate_limit_error, 3, id="429-rate-limit"),
        pytest.param(_connection_error, 3, id="connection"),
        pytest.param(_server_error, 3, id="500"),
    ],
)
async def test_non_size_failure_propagates_without_halving(make_error, expected_calls: int, no_sleep: list[float]):  # type: ignore[no-untyped-def]
    """Auth / rate-limit / connection / server / other errors cost exactly the shared retry policy
    (max_retries=2 -> 1 or 3 calls) and then propagate. Before: 15 calls for a 401, 45 for a 429."""
    client = make_raising_client(make_error)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=ExtractionSettings())

    with pytest.raises(type(make_error())):
        await extractor.extract(big_doc())

    assert client.chat.completions.parse.call_count == expected_calls


@pytest.mark.anyio
async def test_refusal_propagates_without_halving():
    """parsed=None (refusal) is deterministic for the input family — no retry, no halving."""
    from ragdoc.llm import LLMRefusalError

    client = MagicMock()
    refusal = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))])
    client.chat.completions.parse = AsyncMock(return_value=refusal)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")

    with pytest.raises(LLMRefusalError):
        await extractor.extract(big_doc())
    assert client.chat.completions.parse.call_count == 1


def make_two_paragraph_doc() -> Document:
    return Document(
        elements=[Paragraph(html="<p>Alice works at Acme.</p>"), Paragraph(html="<p>Bob works at Beta.</p>")]
    )


def make_halving_client(make_error) -> MagicMock:  # type: ignore[no-untyped-def]
    """Fails (with *make_error*) while the input holds BOTH paragraphs; each half alone succeeds.

    Both halves answer with the same chunk-local ids (n0, n1) — as independent LLM calls do.
    """
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)

    def _batch(person: str, company: str) -> SimpleNamespace:
        batch = graph_batch_cls.model_validate(
            {
                "node_mentions": [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": person}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": company}},
                ],
                "edge_mentions": [
                    {"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}
                ],
            }
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])

    async def _parse(**kwargs):  # type: ignore[no-untyped-def]
        text = kwargs["messages"][1]["content"]
        if "Alice" in text and "Bob" in text:
            raise make_error()
        return _batch("Alice", "Acme") if "Alice" in text else _batch("Bob", "Beta")

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_parse)
    return client


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(_context_length_error, id="context-length-400"),
        pytest.param(_length_finish_error, id="truncated-output"),
        pytest.param(_malformed_output_error, id="malformed-structured-output"),
        pytest.param(_timeout_error, id="timeout"),
    ],
)
async def test_size_related_failure_halves_and_merges(make_error):  # type: ignore[no-untyped-def]
    """Context-length, truncated output, malformed structured output and (post-retry) timeouts
    halve the input; the halves' results are merged with their edges intact."""
    client = make_halving_client(make_error)
    settings = ExtractionSettings(max_retries=0, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)

    merged = await extractor.extract(make_two_paragraph_doc())

    assert client.chat.completions.parse.call_count == 3  # whole text (fails) + two halves
    nodes, edges = _split_by_kind(merged, SCHEMA)
    by_name = {getattr(n.payload, "full_name", None) or getattr(n.payload, "name", None): n.mention_id for n in nodes}
    assert set(by_name) == {"Alice", "Acme", "Bob", "Beta"}
    pairs = {(e.payload.refs.source_mention_id, e.payload.refs.target_mention_id) for e in edges}  # type: ignore[attr-defined]
    assert pairs == {(by_name["Alice"], by_name["Acme"]), (by_name["Bob"], by_name["Beta"])}


@pytest.mark.anyio
async def test_timeout_halves_only_after_shared_retries_are_exhausted(no_sleep: list[float]):
    client = make_halving_client(_timeout_error)
    settings = ExtractionSettings(max_retries=1, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)

    merged = await extractor.extract(make_two_paragraph_doc())

    assert client.chat.completions.parse.call_count == 4  # 2 attempts on the whole text, then 2 halves
    assert len(no_sleep) == 1
    assert len(merged) == 6


@pytest.mark.anyio
async def test_non_size_failure_in_one_half_propagates():
    """Halving was triggered by a size failure; a 401 inside a half is not swallowed or re-halved."""
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)

    async def _parse(**kwargs):  # type: ignore[no-untyped-def]
        text = kwargs["messages"][1]["content"]
        if "Alice" in text and "Bob" in text:
            raise _length_finish_error()
        if "Bob" in text:
            raise _auth_error()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=graph_batch_cls()))])

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_parse)
    settings = ExtractionSettings(max_retries=0, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)

    with pytest.raises(openai.AuthenticationError):
        await extractor.extract(make_two_paragraph_doc())
    assert client.chat.completions.parse.call_count == 3


# ---------------------------------------------------------------------------
# 12. Gleaning shows the model its first answer and de-duplicates the merge
# ---------------------------------------------------------------------------

_FIRST_ANSWER: tuple[list, list] = (
    [
        {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
        {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
    ],
    [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}, "role": "Engineer"}],
)


@pytest.mark.anyio
async def test_gleaning_call_contains_first_answer_as_assistant_turn():
    """The second call must be a continuation of the first — [system, user, assistant, user] —
    with the first answer in the assistant turn. An independent [system, user] re-extraction
    cannot know what it 'missed' and tends to duplicate everything."""
    import json

    client = make_kg_client(SCHEMA, [_FIRST_ANSWER, ([], [])])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    await extractor.extract(make_doc("Alice works at Acme as an engineer."))

    first_call, second_call = (c.kwargs["messages"] for c in client.chat.completions.parse.call_args_list)
    assert [m["role"] for m in first_call] == ["system", "user"]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    # Same conversation prefix: the gleaning call replays the original request verbatim.
    assert second_call[:2] == first_call
    assert "Alice works at Acme as an engineer." in second_call[1]["content"]

    previous_answer = json.loads(second_call[2]["content"])
    assert [n["local_id"] for n in previous_answer["node_mentions"]] == ["n0", "n1"]
    assert previous_answer["node_mentions"][0]["node"] == {"kind": "Person", "full_name": "Alice"}
    assert previous_answer["edge_mentions"][0]["refs"] == {"source_mention_id": "n0", "target_mention_id": "n1"}
    assert previous_answer["edge_mentions"][0]["role"] == "Engineer"

    instruction = second_call[3]["content"]
    assert "previous answer" in instruction
    assert "n2" in instruction  # tells the model where to continue numbering
    assert "Alice works at Acme" not in instruction  # the text is not pasted a second time


@pytest.mark.anyio
async def test_gleaning_merge_deduplicates_repeated_nodes_and_edges():
    """A gleaning answer that repeats the first answer (plus one genuinely new node + edge) adds
    only what is new; an edge through a repeated node re-points at the original node."""
    client = make_kg_client(
        SCHEMA,
        [
            _FIRST_ANSWER,
            (
                [
                    # Repeats (same payloads) under fresh ids, as a model that ignores "only new" would.
                    {"local_id": "n2", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n3", "node": {"kind": "Company", "name": "Acme"}},
                    # Genuinely new.
                    {"local_id": "n4", "node": {"kind": "Person", "full_name": "Bob"}},
                ],
                [
                    # Repeat of the first answer's edge, expressed through the repeated nodes.
                    {
                        "kind": "Employment",
                        "refs": {"source_mention_id": "n2", "target_mention_id": "n3"},
                        "role": "Engineer",
                    },
                    # New edge: new node -> repeated node (must land on the ORIGINAL Acme).
                    {"kind": "Employment", "refs": {"source_mention_id": "n4", "target_mention_id": "n3"}},
                ],
            ),
        ],
    )
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    merged = await extractor.extract(make_doc())

    nodes, edges = _split_by_kind(merged, SCHEMA)
    names = [getattr(n.payload, "full_name", None) or getattr(n.payload, "name", None) for n in nodes]
    assert names == ["Alice", "Acme", "Bob"]  # no duplicate Alice / Acme
    by_name = dict(zip(names, (n.mention_id for n in nodes), strict=True))
    pairs = [(e.payload.refs.source_mention_id, e.payload.refs.target_mention_id) for e in edges]  # type: ignore[attr-defined]
    assert pairs == [(by_name["Alice"], by_name["Acme"]), (by_name["Bob"], by_name["Acme"])]


@pytest.mark.anyio
async def test_gleaning_empty_second_answer_changes_nothing():
    client = make_kg_client(SCHEMA, [_FIRST_ANSWER, ([], [])])
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    merged = await extractor.extract(make_doc())
    assert len(merged) == 3
    assert [m.ordinal for m in merged] == [0, 1, 2]


@pytest.mark.anyio
async def test_halving_merge_keeps_identical_payloads_from_different_halves():
    """De-duplication is a gleaning concern only: the two halves are different text, so the same
    entity appearing in both is two mentions (the prompt asks for one record per occurrence)."""
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)
    alice = graph_batch_cls.model_validate(
        {"node_mentions": [{"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}}], "edge_mentions": []}
    )

    async def _parse(**kwargs):  # type: ignore[no-untyped-def]
        text = kwargs["messages"][1]["content"]
        if "first" in text and "second" in text:
            raise _length_finish_error()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=alice.model_copy(deep=True)))])

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_parse)
    settings = ExtractionSettings(max_retries=0, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    doc = Document(elements=[Paragraph(html="<p>Alice, first.</p>"), Paragraph(html="<p>Alice, second.</p>")])

    merged = await extractor.extract(doc)
    assert len(merged) == 2
    assert merged[0].mention_id != merged[1].mention_id


# ---------------------------------------------------------------------------
# 13. Dangling edge references are dropped (with a warning), never fatal, never mis-attached
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dangling_edge_is_dropped_with_warning(caplog: pytest.LogCaptureFixture):
    """One edge naming a node the model never emitted must not fail the whole source."""
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [
                    {"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n9"}},
                    {"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}},
                    {"kind": "Employment", "refs": {"source_mention_id": "n7", "target_mention_id": "n1"}},
                ],
            )
        ],
    )
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m")
    doc = make_doc()
    doc.source_id = "src-1"
    with caplog.at_level(logging.WARNING, logger="ragdoc.extraction.kg"):
        merged = await extractor.extract(doc)

    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].payload.refs.source_mention_id == nodes[0].mention_id  # type: ignore[attr-defined]
    assert edges[0].payload.refs.target_mention_id == nodes[1].mention_id  # type: ignore[attr-defined]
    assert [m.ordinal for m in merged] == [0, 1, 2]  # ordinals stay contiguous

    warnings = [r for r in caplog.records if "dangling" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2 edge(s)" in message
    assert "src-1" in message
    assert "n9" in message and "n7" in message


@pytest.mark.anyio
async def test_dangling_edge_dropped_even_in_pattern_error_mode():
    """`on_pattern_violation="error"` is about illegal triples; a dangling ref is still just dropped."""
    client = make_kg_client(
        SCHEMA,
        [
            (
                [{"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}}],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n9"}}],
            )
        ],
    )
    settings = ExtractionSettings(on_pattern_violation="error")
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)
    merged = await extractor.extract(make_doc())
    assert [m.payload.kind for m in merged] == ["Person"]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_dangling_edge_in_primary_cannot_attach_to_gleaned_node():
    """A primary edge pointing at a non-existent 'n2' must not silently resolve to the gleaned
    node that the merge renumbers to 'n2'."""
    client = make_kg_client(
        SCHEMA,
        [
            (
                [
                    {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                    {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
                ],
                [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n2"}}],
            ),
            ([{"local_id": "n2", "node": {"kind": "Company", "name": "Beta"}}], []),
        ],
    )
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", gleaning=True)
    merged = await extractor.extract(make_doc())

    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 3
    assert edges == []  # dropped, not re-pointed at Beta


@pytest.mark.anyio
async def test_dangling_edge_in_one_half_cannot_attach_to_other_half():
    """Halves are independent calls: 'n0' in the second half is not the first half's 'n0'."""
    graph_batch_cls, _ = build_graph_batch_model(SCHEMA)

    def _response(nodes: list, edges: list) -> SimpleNamespace:
        batch = graph_batch_cls.model_validate({"node_mentions": nodes, "edge_mentions": edges})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])

    async def _parse(**kwargs):  # type: ignore[no-untyped-def]
        text = kwargs["messages"][1]["content"]
        if "Alice" in text and "Bob" in text:
            raise _length_finish_error()
        if "Alice" in text:
            return _response([{"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}}], [])
        # Second half: its only node is "c0"; the edge's source "n0" exists only in the OTHER half.
        return _response(
            [{"local_id": "c0", "node": {"kind": "Company", "name": "Beta"}}],
            [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "c0"}}],
        )

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_parse)
    settings = ExtractionSettings(max_retries=0, halving_min_chars=0)
    extractor = KnowledgeGraphExtractor(SCHEMA, client=client, model="m", settings=settings)

    merged = await extractor.extract(make_two_paragraph_doc())
    nodes, edges = _split_by_kind(merged, SCHEMA)
    assert len(nodes) == 2
    assert edges == []  # Alice does NOT acquire an Employment at Beta


# ---------------------------------------------------------------------------
# 14. The LLM-facing schema is unchanged by the typed base models
# ---------------------------------------------------------------------------


def test_graph_batch_schema_has_no_base_class_leakage():
    """The typed bases exist for static typing only; nothing of them may reach the JSON schema."""
    graph_batch_cls, node_cls = build_graph_batch_model(SCHEMA)
    schema = graph_batch_cls.model_json_schema()
    assert schema["title"] == "GraphBatch"
    assert "description" not in schema
    assert list(schema["properties"]) == ["node_mentions", "edge_mentions"]
    node_schema = node_cls.model_json_schema()
    assert node_schema["title"] == "NodeWithLocalId"
    assert "description" not in node_schema
    assert list(node_schema["properties"]) == ["local_id", "node"]
    assert node_schema["required"] == ["local_id", "node"]
    assert node_schema["properties"]["node"]["discriminator"]["propertyName"] == "kind"
