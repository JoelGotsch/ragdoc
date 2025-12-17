# ragdoc Change Log

Pre-1.0: breaking changes land at will and are documented here.

## Unreleased

> **⚠ MIGRATION — every stored `content_hash` changes once.** `Document.content_hash()` is now
> a versioned canonical-JSON hash (`ragdoc.content_hash.v1`) over `(title, elements)` — pure
> Python, pandoc-free. Its values differ from the old renderer-based hashes, so the **first
> Boundary-2 sync after upgrading re-chunks and re-embeds the full corpus once**. Direct-path
> syncs are unaffected for unchanged files (`source_hash`, the file-byte hash, short-circuits
> before any content hashing). `SimpleChunker`'s default chunk ids (which *are* the content
> hash) also change once. Ships together with Phase 6 (element-model normalization) so the
> corpus migrates a single time.

### Added (Phase 4 — first-class `Extractor` stage)

- **`ragdoc.extraction.extractor` — the `Extractor[PayloadT]` protocol** (Decision D3-A):
  `async extract(document) -> list[Mention[PayloadT]]` is the typed extraction channel.
  Extraction results are **returned**, never written to `document.metadata`;
  `MentionStorePipeline` consumes them directly (the serialize→`model_validate` round trip
  through metadata is gone) and rejects non-conforming extractors with `TypeError` at
  construction. `payload_model` remains a documented attribute on the concrete classes (store
  queries, `ChangeSet[Mention[...]]` parametrization), deliberately outside the protocol.
- **`as_processor(extractor, metadata_key="mentions", overwrite=False)`** — explicit, opt-in
  adapter wrapping an `Extractor` into a `DocumentProcessor` that dumps serialized mentions into
  `document.metadata` (the one legitimate document-dump use case, e.g. a DocumentStore payload).
  Carries the old idempotency guard; with `overwrite=True` it strips the stale key *before*
  extracting so a mention's metadata locator never embeds a previous mention list.
- **`GraphSchema.patterns` are now enforced** (previously validated but never consumed), in two
  layers: `render_patterns_prompt(schema)` appends the legal `(source)-[edge]->(target)` triples
  to every KG system prompt, and each extracted edge's `(source_kind, edge_kind, target_kind)`
  is validated against `allowed_pattern_kinds(schema)`. Violations follow
  `ExtractionSettings.on_pattern_violation`: `"drop"` (default — edge withheld, counted, one
  WARNING per document) or `"error"` (`ValueError` naming the illegal triple). New schema rule:
  every edge type must appear in at least one pattern (dead configuration fails at declaration
  time). New helpers exported from `ragdoc.extraction`: `kind_of`, `allowed_pattern_kinds`,
  `render_patterns_prompt`.
- **`ExtractionSettings` new fields** (all KG-facing knobs now reachable from settings):
  `gleaning` (was an ad-hoc `EXTRACTION_GLEANING` env read), `max_union_size` (was env
  `KG_MAX_UNION_SIZE`; now `EXTRACTION_MAX_UNION_SIZE` via the prefix), `halving_max_depth` /
  `halving_min_chars` (were hard-coded method defaults), `on_pattern_violation`.

### Changed (Phase 4)

- **`StructuredExtractionProcessor` → `StructuredExtractor`** (module
  `ragdoc.extraction.processor` → `ragdoc.extraction.structured`) and
  **`KnowledgeGraphProcessor` → `KnowledgeGraphExtractor`** (`kg_processor` → `kg`). Both
  classes no longer subclass `DocumentProcessor`: `process()`, `metadata_key`, `overwrite`, and
  the `document.metadata` writes are gone — `extract()` is the only entry point. Both are now
  exported from `ragdoc.extraction` (alongside `Extractor`, `as_processor`,
  `ExtractionSettings`).
- **`ExtractionSettings.system_prompt` and `request_timeout` are honored by both extractors**
  (previously each honored a different one). The settings validator no longer injects the
  generic built-in prompt when unset — `system_prompt` stays `None` and each extractor resolves
  its own default (`EXTRACTION_SYSTEM_PROMPT` / `KG_EXTRACTION_SYSTEM_PROMPT`), so a user-set
  prompt finally reaches the KG extractor.
- `GraphSchema` validation is environment-independent: the union-size check moved from the model
  validator (which read `KG_MAX_UNION_SIZE` from the environment) to
  `KnowledgeGraphExtractor.__init__` against `settings.max_union_size`.
- The duplicated client/model/renderer/tokenizer fallbacks and the structured-output retry loop
  now live once in `ragdoc.extraction._llm` (`resolve_*`, `parse_with_retry`, `build_messages`);
  `build_extraction_messages` / `build_kg_messages` remain as thin wrappers documenting the
  per-extractor defaults.

### Removed (Phase 4)

- `KG_MAX_UNION_SIZE` environment variable (use `EXTRACTION_MAX_UNION_SIZE` /
  `ExtractionSettings.max_union_size`), the internal gleaning env read, and the extractors'
  `metadata_key` / `overwrite` constructor parameters (relocated to `as_processor`).

### Changed (Phase 5 — canonical `content_hash`)

- **`Document.content_hash()` is canonical-JSON and pandoc-free** (Decision D4-A):
  `sha256("ragdoc.content_hash.v1" + canonical JSON of (title, elements))` with per-element-type
  payload builders that **fail loudly** (`TypeError`) on unknown element types. Inline
  `<ref id=...>` uuids are normalized to document-order ordinals (dangling refs →
  `"unresolved"`), so a re-parse of an unchanged file hashes identically. Excluded, as before
  but now documented: document/element `metadata` (metadata-only edits do **not** re-chunk),
  provenance fields (`id`, `source_path`, `source_id`, `source_hash`, `parser`,
  `parser_version`, `external_refs`), element `page`/`bounding_box`. Newly participating
  (more-sensitive deltas): element type (a `RawText` vs `Paragraph` with identical text now
  differ) and Image `image`/`image_type`/`width`/`height`. No caching — the hash is pure
  Python and cheap; new standalone helpers `document_content_payload`,
  `element_content_payload`, `normalize_ref_ids` in `ragdoc.document`.
- **`content_hash()`'s `renderer` parameter is removed** (breaking). Customization is
  subclass-and-override, composing over `super().content_hash()`. `ragdoc.document` no longer
  imports from `ragdoc.rendering`.
- `VectorStorePipeline.plan()` over an unchanged Boundary-2 corpus now makes **zero** pandoc
  invocations (previously one pandoc subprocess per stored document per plan).

### Added

- **`ragdoc.pipeline.sync` — generic `SyncEngine[T]`** (Decision D2-A). One streaming
  plan/apply/run core now drives all three sync pipelines (`VectorStorePipeline`,
  `DocumentStorePipeline`, `MentionStorePipeline`), which became thin compositions owning only
  constructor validation, source resolution, and a per-source producer. New public names in
  `ragdoc.pipeline`: `SyncEngine`, `SourceSyncStore`, `SyncSource`, `SyncPlanInput`,
  `SourceOutcome`, `file_hash` (the file-byte SHA-256 hash, now public; `_file_hash` removed).
  Pointing the engine at a new sink is configuration: any store with
  `upsert`/`delete_by_source`/`list_source_state` plus a producer coroutine.
- Concurrent, threaded file hashing: the shared current-map builder
  (`ragdoc.pipeline.sync.build_current_map`) fans hashing out via `asyncio.to_thread` under the
  concurrency semaphore, and detects `source_id` collisions **before** any hashing.

### Changed

- **Unified sync semantics across all three pipelines** (formerly drifted per copy):
  - A Boundary-2 target `source_id` missing from the DocumentStore, or a document that vanishes
    between `list_source_state` and `get_document`, is **logged and skip-counted** everywhere
    (previously uncounted; `MentionStorePipeline.plan()` was fully silent).
  - A **filtered source** (processor returns `None`) yields an empty change: counted
    `processed` and its stale stored artifacts are **deleted**. This fixes
    `DocumentStorePipeline` leaving a stale stored Document behind forever when its source file
    becomes filtered.
  - `apply()` performs delete-then-upsert with per-source error isolation everywhere: a source
    whose stale-delete failed is recorded in `errors` and **not** upserted (previously the
    mention path could mix old and new mentions; the docstore path did no stale-delete).
  - Error handling is `except Exception` at every per-source site (was `except BaseException`):
    `asyncio.CancelledError`, `KeyboardInterrupt`, `SystemExit`, and custom `BaseException`
    subclasses now always propagate. `UpdateResult.errors` is
    `list[tuple[str, Exception]]` (was `BaseException`).
- **`DocumentStorePipeline.run()` is now streaming** (was literally `apply(plan(...))`):
  each source is written as soon as it is produced (per-source durability on abort); updates go
  delete-then-upsert. Same end state.
- `UpdateResult` moved to `ragdoc.pipeline.sync` (still re-exported from `ragdoc.pipeline`).
- `MentionStorePipeline.plan()/apply()` payload type is now `ChangeSet[Mention[P]]`
  (load with `ChangeSet[Mention[P]].load(path)`).
- `ChangeSet`'s type parameter loosened from `TypeVar("T", Document, Chunk)` to
  `TypeVar("T", bound=BaseModel)` — any Pydantic model payload.

### Removed

- **`VectorStore.get_source_hash`** (protocol + `QdrantVectorStore` implementation) — superseded
  by the bulk `list_source_state`. `list_source_ids` stays (consumer-facing; the sync engine
  does not use it).
- **`ragdoc.extraction.changeset`** (`MentionChangeSet`, `MentionSourceChange`) — use
  `ChangeSet[Mention[P]]` / `SourceChange[Mention[P]]` from `ragdoc.pipeline.changeset`.

## 0.1.0 — 2026-07-04

### Added

- **`ragdoc.extraction` — typed knowledge-graph / structured extraction** _(new, `extraction`
  extra)_. Multi-node-type + typed-relationship extraction from Documents:
  - `FuzzyDate` (EDTF-backed arbitrary-precision dates), `Entity`, `Mention`, `GraphSchema`
    (closed ontology of node/edge types + legal patterns).
  - `StructuredExtractionProcessor` and `KnowledgeGraphProcessor` (single `.parse()` over a
    built discriminated-union batch model, gleaning, and a recursive `_extract_with_halving`
    fallback on LLM failures).
  - `MentionStore`/`LocalMentionStore`, `EntityStore`/`LocalEntityStore`,
    `GraphStore`/`LocalGraphStore`, `MentionStorePipeline` (plan/apply/run, Boundary-1 and
    Boundary-2), and `EntityResolutionPipeline` / `KnowledgeGraphResolutionPipeline`.
  - Qdrant adapters (`qdrant` extra): `QdrantMentionStore` (with a `payload_type` filter),
    `QdrantEntityStore` (server-side date-range filtering), `QdrantGraphStore`.
  - Env prefixes: `EXTRACTION_*` and `KG_MAX_UNION_SIZE`.

- **`DocumentSummarizerProcessor`** — recursive map-reduce whole-document summarization into
  `document.metadata["summary"]`, with `DocumentSummarizerSettings` (env prefix
  `DOCUMENT_SUMMARIZER_`) and `DocumentSummary` result model.
- **`OpenAIEmbedder`** — concrete `Embedder` wrapping an OpenAI-compatible embeddings endpoint
  (typed via a structural client Protocol; no `Any`). Exported from `ragdoc.pipeline`.
- **`QdrantDocumentStore`** (`ragdoc.integrations.document_stores`) — Qdrant-backed
  `DocumentStore` for Boundary-1 sync, with `DocumentTooLargeError` for oversized payloads.
- **Streaming per-source `VectorStorePipeline.run()`** — the standard sync path now processes
  each source independently (parse → chunk → embed → delete-then-upsert) with flat peak memory
  and per-source atomicity, instead of `apply(plan(...))`. `plan()`/`apply()` remain the
  reviewable whole-corpus path.
- **Boundary predicates on `DocumentPipeline`** — `has_splitter`, `has_custom_chunker`,
  `has_processors`, `has_custom_parser` — used by the sync pipelines to fail loudly on
  misconfiguration.

### Changed

- **`DocumentPipeline.chunk_document()` no longer processes** — processing now runs exactly once,
  at parse time (`_process_one`) or at Boundary 1 (`DocumentStorePipeline`). Re-chunking a stored
  Document (Boundary 2 / Mode 2) splits and chunks only, fixing destructive double-processing by
  non-idempotent processors (`LLMHeadingResolver`, `FootnoteProcessor`).
- **Boundary construction guards** — `DocumentStorePipeline` rejects a splitter/non-default
  chunker; `VectorStorePipeline` (with a `document_store`) rejects processors/a custom parser.
- **`FootnoteProcessor` `only_orphaned` defaults to `True`** — re-running is now idempotent
  (never appends duplicate refs). Pass `only_orphaned=False` for the legacy behaviour.
- **`LLMHeadingResolver` short-circuits when `document.title` is already set** — prevents a second
  pass from re-detecting a different title and re-stripping elements.

- **`ragdoc.pipeline` module** — high-level facade over parse → process → split → chunk
  (Scenarios A and D from `PLAN_PIPELINE.md`).
  - `DocumentPipeline` — linear pipeline with async `run()`, `run_many()`, `stream()`,
    and `run_sync()` entry points.  Supports configurable concurrency via
    `asyncio.Semaphore` and per-file error handling (`on_error="raise"|"skip"`).
  - `TokenSplitter` — `Splitter`-Protocol-compatible wrapper around `split_document()`.
  - `AutoParser` / `Parser` Protocol — selects the right parser from a file extension.
  - `PipelineResult` dataclass — aggregates chunks and errors from `run_many()`.

- **`Document.content_hash()`** — content-stable SHA-256 hash of a document.
  Uses `MARKDOWN + render_for_prompt` by default.  Subclass and override to
  incorporate provenance (e.g. file path) or switch renderer.

### Changed

- **`SimpleChunker` default `id_fn`** _(breaking)_ — changed from `uuid.uuid4()` to
  `document.content_hash()`.  The same document content now always produces the same
  chunk ID, making vector-store upserts idempotent.  Pass `id_fn=lambda _: str(uuid4())`
  to restore the previous behaviour.