# ragdoc Change Log

Pre-1.0: breaking changes land at will and are documented here.

## Unreleased

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