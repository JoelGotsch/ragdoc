# Extraction & Knowledge Graphs

> Run interactively: `marimo edit docs/notebooks/extraction.py`

Structured extraction turns documents into **typed records** — events, people,
organizations, or a full knowledge graph — using the same incremental-sync machinery
that keeps a vector store up to date. It is a first-class pipeline stage, not a
processor: extractors *return* typed objects, and nothing rides on
`document.metadata`.

## Installation

```bash
uv add "ragdoc[extraction,llm]"     # extractors + resolution (edtf, numpy) + OpenAI client
uv add "ragdoc[qdrant]"             # optional: Qdrant-backed stores
```

The data models (`Mention`, `Entity`, `GraphSchema`, `FuzzyDate`) and the local stores
import without any extra; the LLM extractors need `extraction` (and a configured client,
usually via the `llm` extra).

## The three layers

| Layer | Object | Produced by | Stored in |
|---|---|---|---|
| 1 — Mentions | [`Mention`](../api/extraction.md#mention) — one raw, provenance-tagged occurrence | an [`Extractor`](../api/extraction.md#extractor) via [`MentionStorePipeline`](../api/extraction.md#mentionstorepipeline) | [`MentionStore`](../api/extraction.md#mentionstore) |
| 2 — Entities | [`Entity`](../api/extraction.md#entity) — canonical, deduplicated record | [`EntityResolutionPipeline`](../api/extraction.md#entityresolutionpipeline) | [`EntityStore`](../api/extraction.md#entitystore) |
| 3 — Graph | node + edge entities against a [`GraphSchema`](../api/extraction.md#graphschema) | [`KnowledgeGraphResolutionPipeline`](../api/extraction.md#knowledgegraphresolutionpipeline) | [`GraphStore`](../api/extraction.md#graphstore) |

Mentions are cheap and append-per-source (synced like chunks — one set per
`source_id`); entities are the expensive, corpus-global product recomputed by
resolution. Keeping the layers separate means re-running resolution never re-runs
extraction LLM calls.

## Layer 1 — Extracting mentions

### The `Extractor` protocol

Anything with `async extract(document) -> list[Mention[PayloadT]]` is an
[`Extractor`](../api/extraction.md#extractor). The contract: `extract` is a pure query
over the document — it never reads or writes `document.metadata` keys of its own
(each mention's locator *copies* the document metadata, so `split_sequence` and your
custom keys arrive on every mention).

Because the protocol is structural, a stub is a legal extractor — the
[extraction notebook](../notebooks/extraction/) runs the entire workflow with one, no
API key required:

```python
from ragdoc.document import Document
from ragdoc.extraction import Mention, mint_mention_id
from pydantic import BaseModel, Field

class Person(BaseModel):
    """A person mentioned in the text."""

    name: str = Field(description="Full name as written in the text.")
    role: str | None = Field(default=None, description="Role or title, when stated.")

class KeywordPersonExtractor:
    """Toy Extractor: no LLM — matches a fixed name list against the rendered text."""

    payload_model = Person

    def __init__(self, known: dict[str, str | None]) -> None:
        self._known = known

    async def extract(self, document: Document) -> list[Mention[Person]]:
        text = " ".join(el.text for el in document.elements)
        mentions: list[Mention[Person]] = []
        for ordinal, (name, role) in enumerate(self._known.items()):
            if name in text:
                payload = Person(name=name, role=role)
                mentions.append(
                    Mention(
                        mention_id=mint_mention_id(document.source_id or document.id, None, ordinal, payload),
                        source_id=document.source_id or document.id,
                        source_hash=document.source_hash or "",
                        ordinal=ordinal,
                        metadata=dict(document.metadata),
                        payload=payload,
                    )
                )
        return mentions
```

### `StructuredExtractor` — one payload model, LLM-driven

For real extraction, [`StructuredExtractor`](../api/extraction.md#structuredextractor)
asks an LLM for every instance of a caller-supplied Pydantic model. **Your model is the
schema**: its docstring and `Field(description=...)` strings are serialized into the JSON
schema the LLM is constrained to.

```python
from ragdoc.extraction import FuzzyDate, StructuredExtractor
from pydantic import BaseModel, Field

class Event(BaseModel):
    """A dated event described in the document."""

    title: str = Field(description="Short name of the event.")
    date: FuzzyDate = Field(description="When the event happened (verbatim + canonical).")
    location: str | None = Field(default=None, description="Where, if stated.")

extractor = StructuredExtractor(Event)   # client from configure(openai_client=...)
mentions = await extractor.extract(document)
```

[`FuzzyDate`](../api/extraction.md#fuzzydate) is a reusable payload building block for
partial/uncertain dates ("Q3 2024", "the nineties"): it keeps the verbatim text, an
EDTF canonical form, and derives a comparable `(start, end)` interval with a
[`Precision`](../api/extraction.md#precision).

Behavior is configured via [`ExtractionSettings`](../api/extraction.md#extractionsettings)
(env prefix `EXTRACTION_`): `system_prompt` / `system_prompt_file`, `model_name`,
`min_tokens` (skip tiny splits without an LLM call), `max_retries`, `request_timeout`.

### `MentionStorePipeline` — incremental extraction sync

Extraction LLM calls are expensive, so mentions are synced exactly like chunks:
[`MentionStorePipeline`](../api/extraction.md#mentionstorepipeline) composes an
[`IngestPipeline`](../api/pipeline.md#ingestpipeline) (parse → process), a splitter
(extraction runs per context-sized split), and your extractor, with per-source change
detection — an unchanged file is **skipped with no LLM call**.

```python
from pathlib import Path
from ragdoc.extraction import LocalMentionStore, MentionStorePipeline
from ragdoc.pipeline import IngestPipeline

pipeline = MentionStorePipeline(
    ingest=IngestPipeline(),                       # parser + processors + source_id_fn
    extractor=extractor,
    mention_store=LocalMentionStore("mentions/", Event),
)
result = await pipeline.run(Path("corpus").glob("**/*.html"))
print(f"extracted={len(result.processed)} unchanged={len(result.skipped)}")
```

Like every sync pipeline, it exposes `plan()` / `apply()` for human-in-the-loop review —
`plan()` returns a `ChangeSet[Mention[Event]]` and never touches the store (see the
[Sync Engine Guide](sync-engine.md)). Boundary-2 mode
(`MentionStorePipeline.from_document_store(...)`) extracts from Documents already synced
into a [`DocumentStore`](../api/pipeline.md#documentstore) — sharing the parse-once
substrate with the vector-store workflow.

## Layer 2 — Resolving mentions into entities

The same person extracted from ten documents is ten mentions but one entity.
[`EntityResolutionPipeline`](../api/extraction.md#entityresolutionpipeline) clusters a
`MentionStore` into an `EntityStore`:

1. **Seed** — group mentions whose identity text matches exactly.
2. **Block** — embed each cluster's representative text and propose nearest-neighbour
   merge candidates above `cosine_threshold`.
3. **Review** — an LLM reviewer confirms or rejects each candidate group
   ([`make_llm_reviewer`](../api/extraction.md#make_llm_reviewer); inject a fake in tests).
   Confident groups merge; uncertain ones are surfaced as `pending`, never silently merged.
4. **Canonicalize** — each final cluster becomes one
   [`Entity`](../api/extraction.md#entity) carrying its member `mention_ids` and
   contributing `source_ids`.

```python
from ragdoc.extraction import EntityResolutionPipeline, LocalEntityStore, make_llm_reviewer

resolution = EntityResolutionPipeline(
    mention_store=mention_store,
    entity_store=LocalEntityStore("entities/", Event),
    embed=my_embed_fn,                             # async list[str] -> list[vector]
    reviewer=make_llm_reviewer(client, "gpt-4.1"),
)
result = await resolution.resolve()
print(f"{len(result.entities)} entities, {len(result.pending)} pending review")
```

Both collaborators are injected: `embed` is any `async (list[str]) -> list[list[float]]`
(wrap it with [`build_entity_embedder`](../api/extraction.md#build_entity_embedder) when a
store needs to embed payloads directly), and `reviewer` is any
`async (list[str]) -> ReviewResult` — pass deterministic fakes in tests and notebooks.

Entities can be queried with [`EntityQuery`](../api/extraction.md#entityquery) /
[`filter_entities`](../api/extraction.md#filter_entities) — including date-range overlap
via `FuzzyDate`.

## Layer 3 — Typed knowledge graphs

A knowledge graph adds **multiple node types, typed edges, and legal patterns**:

```python
from ragdoc.extraction import EdgeRef, GraphSchema, KnowledgeGraphExtractor
from pydantic import BaseModel, Field
from typing import Literal

class Company(BaseModel):
    """A company."""

    kind: Literal["company"] = "company"
    name: str = Field(description="Company name.")

class PersonNode(BaseModel):
    """A person."""

    kind: Literal["person"] = "person"
    name: str = Field(description="Full name.")

class WorksAt(BaseModel):
    """Employment relationship."""

    kind: Literal["works_at"] = "works_at"
    refs: EdgeRef = Field(description="Endpoints: person -> company.")
    title: str | None = Field(default=None, description="Job title, when stated.")

schema = GraphSchema(
    node_types=(PersonNode, Company),
    edge_types=(WorksAt,),
    patterns=((PersonNode, WorksAt, Company),),
)
kg_extractor = KnowledgeGraphExtractor(schema)
```

Every node/edge model declares a `kind: Literal[...]` discriminator first; every edge
carries a `refs: EdgeRef` field. The schema validates itself at construction (unknown
pattern types, missing discriminators, and edge types that appear in no pattern are
`ValueError`s).

**Patterns are enforced**, not advisory: legal triples are injected into the system
prompt, and every extracted edge is validated against
[`allowed_pattern_kinds`](../api/extraction.md#allowed_pattern_kinds) — violations are
dropped (counted + warned) or raise, per `ExtractionSettings.on_pattern_violation`.
The extractor recovers from LLM failures on large splits by recursively halving the text
(`halving_max_depth`, `halving_min_chars`) and can run an optional `gleaning` pass for
missed entities.

`KnowledgeGraphExtractor` is just another `Extractor` — the same `MentionStorePipeline`
syncs its node + edge mentions. Resolution then runs per node type, rewrites edge
endpoints from mention ids to entity ids, and (optionally) deduplicates edges:

```python
from ragdoc.extraction import KnowledgeGraphResolutionPipeline

kg_resolution = KnowledgeGraphResolutionPipeline(
    schema=schema,
    mention_store=mention_store,
    node_entity_store=node_store,
    edge_entity_store=edge_store,
    embed=embed, reviewer=reviewer,
)
result = await kg_resolution.resolve()
# result.node_entities / result.edge_entities / result.orphan_edges (audited, not dropped)
```

The resolved graph lands in a [`GraphStore`](../api/extraction.md#graphstore)
([`LocalGraphStore`](../api/extraction.md#localgraphstore) for local development,
[`QdrantGraphStore`](../api/integrations.md#qdrantgraphstore) with the `qdrant` extra).

## Store implementations

| Protocol | Local (no extra) | Qdrant (`qdrant` extra) |
|---|---|---|
| `MentionStore` | `LocalMentionStore` | [`QdrantMentionStore`](../api/integrations.md#qdrantmentionstore) |
| `EntityStore` | `LocalEntityStore` | [`QdrantEntityStore`](../api/integrations.md#qdrantentitystore) |
| `GraphStore` | `LocalGraphStore` | [`QdrantGraphStore`](../api/integrations.md#qdrantgraphstore) |

The local stores are JSON-per-file on disk — inspectable and hand-editable, ideal for
development and review.

## Escape hatch: `as_processor`

If you genuinely want mentions serialized *inside* a Document (e.g. dumped into a
`DocumentStore`), the explicit [`as_processor`](../api/extraction.md#as_processor)
adapter wraps any extractor as a `DocumentProcessor` writing to a metadata key. This
deliberately violates the library's metadata rule, which is why it is opt-in and never
used by `MentionStorePipeline`.

## See Also

- [Extraction API reference](../api/extraction.md)
- [Extraction notebook](../notebooks/extraction/) — runnable end-to-end with a stub extractor
- [Sync Engine Guide](sync-engine.md) — the plan/apply/run machinery all sync pipelines share
- [Pipeline Guide](pipeline.md) — `IngestPipeline` and the two sync boundaries
