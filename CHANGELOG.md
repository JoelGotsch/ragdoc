# ragdoc Change Log

Pre-1.0: breaking changes land at will and are documented here.

## Unreleased

> **⚠ MIGRATION — every stored `content_hash` and every chunk id changes once.**
> `Document.content_hash()` is now a versioned canonical-JSON hash (`ragdoc.content_hash.v1`)
> over `(title, elements)` — pure Python, pandoc-free. Its values differ from the old
> renderer-based hashes, so the **first Boundary-2 sync after upgrading re-chunks and re-embeds
> the full corpus once**. Direct-path syncs are unaffected for unchanged files (`source_hash`,
> the file-byte hash, short-circuits before any content hashing). Chunk ids move to a new
> deterministic scheme minted by the pipeline (`mint_chunk_id` over
> `(source_id, split_sequence, chunk_ordinal, content_hash)`), so all stored chunk ids change
> once too. Phases 5 and 6 (canonical hash + element-model normalization) ship in **this one
> release** so the corpus migrates a single time.

### Added (Phase 7 — shared LLM reliability layer)

- **`ragdoc.llm` — one module for every LLM call**: structural client protocols
  (`ChatClient` for `chat.completions.parse`, `EmbeddingsClient` for `embeddings.create`, and
  the `LLMClient` intersection that `RagdocConfig.openai_client` now holds — `AsyncOpenAI` /
  `AsyncAzureOpenAI` satisfy all three structurally); `resolve_openai_client()` (the single
  fallback policy: explicit → `get_config().openai_client` → `LLMNotConfiguredError`, fail-loud
  at construction); `call_structured()` (one structured-output call on the **non-beta**
  `client.chat.completions.parse`); and the generic retry engine `retry_llm()` (shared with
  `OpenAIEmbedder`). `ChatClient`/`EmbeddingsClient`/`LLMClient` are exported from the `ragdoc`
  root. The module never imports `openai` at module level (exception classes load lazily),
  ready for the Phase 8 extras split.
- **One retry policy everywhere** (transport errors only): 429 (honoring a parseable
  `Retry-After` header), connection/timeout errors, and 5xx are retried with full-jitter
  exponential backoff (`min(backoff_max, backoff_base * 2**attempt) * uniform(0.5, 1.5)`);
  other 4xx, `pydantic.ValidationError`, and non-openai errors raise immediately. A model
  refusal (`parsed is None` or empty `choices`) raises **`LLMRefusalError`** and is never
  retried — a refusal is deterministic for a given input.
- **`OpenAIEmbedder`** gains `timeout` and `max_retries` and runs `embeddings.create` under
  `retry_llm` (previously no retry at all).
- **`LLMChunker`** gains `request_timeout`, `max_prompt_tokens`, and `tokenizer` parameters:
  when the rendered document exceeds `max_prompt_tokens`, the **LLM input** is truncated (one
  WARNING per document) while the emitted chunks keep the full text as `prompt_content`.
- **`request_timeout` settings** on `LLMHeadingResolverSettings` and
  `DocumentSummarizerSettings` (both forwarded per request); `ExtractionSettings.request_timeout`
  is now honored by *both* extractors.

### Changed (Phase 7)

- **All eight LLM call sites migrated off the deprecated `beta.chat.completions.parse`
  namespace** onto `ragdoc.llm.call_structured` (footnote resolver, heading resolver, document
  summarizer, image summarizer, `LLMChunker`, both extractors via `parse_with_retry`, and the
  entity-resolution reviewer). Per-site degrade semantics are deliberately preserved: heading →
  `[]`, image summary → skip-with-warning, footnote → `None`, reviewer → `ReviewResult()`;
  summarizer/chunker/extraction raise.
- **`RagdocConfig.openai_client` is now typed `LLMClient | None`** (was `Any | None`); invalid
  objects fail validation at `configure()` time. There is no `client: Any` left in the library.
- **Default models bumped**: `default_llm_model` / `default_image_llm_model` are now `gpt-4.1`
  (were the two-year-old `gpt-4o-2024-08-06` snapshot). Kept within the gpt-4.x family because
  every call site pins `temperature=0.0`, which reasoning-family models reject.
- **`LLMFootnoteResolver`** moved from string-parsing (`"NONE"` / int-parse of free text) to a
  structured `_FootnoteSelection` response model; its client is resolved fail-loud at
  construction (`LLMNotConfiguredError` instead of a silent `None` that crashed at resolve
  time), and `model=None` now falls back to `get_config().default_llm_model` (was a hard-coded
  `gpt-4o-mini` default).
- **`LLMChunker` fails loudly at `__init__`** when no client is resolvable (was: silent `None`,
  `AttributeError` at chunk time); a refusal at chunk time raises `LLMRefusalError` (was
  `ValueError`).
- **`LLMHeadingResolver`** no longer retries deterministic 4xx errors (previously its manual
  loop retried everything with no backoff); it still degrades to `[]` on final failure. The
  settings-based client factory (base_url + api_key) survives as a documented layer above
  `resolve_openai_client`; `from openai import AsyncOpenAI` is now a lazy import inside it.
- **`ImageSummaryProcessor`** resolves a missing client fail-loud (`LLMNotConfiguredError`
  instead of an unchecked `None`); per-image refusals skip with a WARNING (image keeps
  `text_representation=None`).
- **`make_llm_reviewer`** (entity resolution) gains retry it never had; refusal still degrades
  to `ReviewResult()`.
- **openai dependency floor raised to `>=1.92.0`** — the first release shipping the non-beta
  `chat.completions.parse` namespace.

### Removed (Phase 7)

- The seven-protocol `_SummaryClient` stack in `processing/summary_document.py` and the
  four-protocol `_EmbeddingsClient` stack in `pipeline/embedders.py` (replaced by the shared
  `ragdoc.llm` protocols; `EmbeddingsClient` is re-exported from `ragdoc.pipeline`).
- `ragdoc.extraction._llm.resolve_client` (superseded by `ragdoc.llm.resolve_openai_client`).
- All four hand-rolled retry loops (heading resolver, document summarizer, and the extraction
  `parse_with_retry` internals — its body is now a `call_structured` call) and every
  `beta.chat.completions` reference in the library.
- `LLMFootnoteResolver`'s `"NONE"`-string protocol and int-parsing block.

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

### Changed (Phase 6 — element model normalization; ships with Phase 5, one corpus migration)

- **`BaseElement.html` is a plain stored field on `Heading`, `Paragraph`, `Table`,
  `DocumentList`, and `RawText`** (Decision D5-A), each with a normalizing `field_validator`
  (Heading: first `<h1>`–`<h6>` tag re-serialized by bs4, else text escaped + `<h1>`-wrapped;
  Paragraph/Table/DocumentList: `.strip()`; RawText: normalized to a single outer `<div>`).
  The `html_content`/`innerhtml` dual-storage split, the `@computed_field html`
  property/abstract-setter machinery, `_fields_from_html`, the `_parse_html_input`
  before-validator, and the seven per-subclass `fset` blocks are **gone**. Construct with
  `Heading(html="<h2>…</h2>")` etc.; `Heading(innerhtml=…, level=…)` and `html_content=` no
  longer exist. `Heading.innerhtml` / `RawText.innerhtml` survive as read-only properties;
  `Heading.level` is derived (setter rebuilds html).
- **`Image` and `Footnote` keep *derived* `html`** (property + real setter): Image's html is a
  projection of six structured fields (a stored copy would duplicate base64 and desync);
  Footnote's html embeds the live `id="footnote-{id}"`. `Image(html='<img …/>')` /
  `Footnote(html='<aside …>')` still work via before-validators; the parsing helpers are public
  standalone functions `image_fields_from_html` / `footnote_fields_from_html`.
- **Identity-keyed soup cache**: derived accessors (`text`, `inline_refs`, `footnote_ids`,
  `image_ids`, `level`, `innerhtml`) share one cached BeautifulSoup per element, invalidated by
  string-object identity on assignment — at most one parse per element per pipeline pass
  (previously a fresh parse per access). `html_tag` deliberately stays fresh-parse (it returns
  a mutable `Tag`). `BaseElement` now has `validate_assignment=True`: every field assignment
  re-validates and re-normalizes.
- **`from_html` removed; `from_markdown` is typed** — `cls(html=…)` replaces `from_html(…)`;
  `from_markdown(markdown_text, *, page=0)` lost its `**kwargs`.
- `BaseElement.inline_refs` skips refs with unknown `rel` values with a WARNING (once per
  value) instead of a blind `except Exception: pass`.
- **Serialization shape**: `model_dump()` now emits `html` as a real field for the five
  stored-html classes (no `html_content`/`innerhtml` keys) and **no** `html` key for
  Image/Footnote. Old dumps happen to load under the new model (the old computed `"html"` key
  feeds the new field) — pinned by a test but **not promised**; new dumps are the format.
- **Splits own their metadata**: every splitter output `Document` copies the parent's metadata
  dict at construction (previously shared by reference and compensated downstream), and the
  `split_document` no-split path **no longer mutates the input document** — it returns a
  shallow copy carrying `split_sequence`/`split_total`. Elements remain shared across splits
  by design.
- **Chunk ids are minted by `DocumentPipeline.chunk_document`** over
  `(source_id, split_sequence, chunk_ordinal, content_hash)` via the new
  `ragdoc.chunking.provenance.mint_chunk_id` (override with `DocumentPipeline(chunk_id_fn=…)`,
  type `ChunkIdFn`). This fixes the collision where two identical-content splits of one source
  produced one chunk id, and gives `LLMChunker` stable ids. The chunkers' `id_fn` constructor
  parameters are **removed**; standalone chunker use yields uuid4 ids.
- **`Chunk.source_hash` is `str | None` and honest** — it is the file-byte hash from the sync
  pipelines' `hash_fn` or `None`; the silent `content_hash` fallback at every stamping site is
  gone (use `content_hash` for content change detection). The shared fallback chain lives in
  `ragdoc.chunking.provenance.resolve_chunk_provenance`.
- **`merge_documents(first, second, *, metadata_policy="first"|"second"|"strict")` replaces
  `Document.__or__`** with an explicit, documented field policy (title bridge heading is now
  HTML-escaped; `parser`/`parser_version` kept iff identical; metadata always a fresh dict;
  `"strict"` raises on conflicting keys). `join_documents` gained the same `metadata_policy`
  keyword. (Not to be confused with `ragdoc.merging.merge_documents`, which aligns two parses
  of the *same* source.)
- **Parser provenance is stamped centrally in `ragdoc.parsing.load()`** via the new
  `stamp_provenance(document, source, parser_name)` (fills `parser`, `source_path`,
  `metadata["filename"]` only when unset). The standalone loaders (`load_html`, `load_pandoc`,
  `load_excel`, `load_azure_json`, mineru) no longer stamp these fields themselves — call
  them through `load()` (or stamp manually) when you need file provenance.
- **Escaping fixes** (four injection sites): Heading attribute values (via bs4
  re-serialization), `Image.html` alt text, the renderer's metadata `<head>`
  (title + `<meta>` tags), and xlsx sheet-name headings.
- HTML parser: `<h7>`+ tags (a pandoc docx artifact) are clamped to level 6.

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