# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Important:** Always use `uv run python` (or `uvx python`) instead of bare `python` to ensure the correct virtual environment and dependencies are used.

**Install dependencies:**
```bash
uv sync --all-extras --group docs
```

**Run tests:**
```bash
uv run pytest                        # all tests
uv run pytest tests/test_document.py # single file
uv run pytest tests/ -k "test_name"  # single test by name
uv run pytest-watcher tests/         # watch mode
```

**Lint, format, type-check (ruff + basedpyright):**
```bash
uv run ruff check src tests          # lint
uv run ruff check src tests --fix    # lint + autofix
uv run ruff format src tests         # format (black-compatible, line-length 120)
uv run basedpyright                  # type check (standard mode)
```
These mirror the CI jobs; tox wraps them as `tox -e lint` and `tox -e type`.

**Pre-commit hooks:**
```bash
uv run pre-commit install            # set up the git hook
uv run pre-commit run --all-files    # run ruff + hygiene hooks on everything
```

**Build docs (MkDocs + marimo):**
```bash
make html     # export marimo notebooks to WASM, then build the MkDocs site
make serve    # live-reload preview
```

## Architecture

### Core philosophy

Four principles govern all design decisions:

1. **`Document` is the single source of truth.** Every stage operates on structured `Document` objects. Never pass rendered strings between stages. The `Document` model is the unifying interface throughout parsing, processing, splitting, and chunking.

2. **The library is async-only.** All public entry points are `async`. Do not add sync convenience wrappers. Callers use `asyncio.run(...)` or their own event loop.

3. **Chunkers own the embedding content strategy.** The library produces two text representations per chunk (`prompt_content` and `embedding_content`), but the chunker decides how:
   - **`SimpleChunker`** — sets `embedding_content = prompt_content` (no LLM calls). Use when a single rendering suffices.
   - **`LLMChunker`** — generates N distinct `embedding_content` strings (one per topic) via LLM, all sharing the same `prompt_content`.

   Rendering uses `render_for_prompt` (full-fidelity structured text) or `render_raw` (with base64 images). There is no separate embedding renderer — chunkers handle that concern.

4. **Source provenance and metadata propagate through the entire pipeline.** `Document.source_path` is an explicit field set by parsers. Parsers also write `document.metadata["filename"] = path.name` — the bare filename lives in metadata, not as a first-class field. Provenance propagates through splitting and chunking into `Chunk.source_path` (optional on Chunk). `Document.metadata` is a dict that carries both **user-defined custom data** and **library-written informational fields** (e.g. `filename` set by parsers, `split_sequence`/`split_total` set by `split_document`). The rule for library-written metadata keys is: **no other part of the library may read or branch on them** — they exist purely for external consumers (retrieval clients, self-query filters). Keys that drive internal library behavior (change detection, deletion) belong as first-class fields on `Chunk` instead (`source_id`, `source_hash`). Any key/value present in `document.metadata` will arrive on the final chunks. Processors must not drop or overwrite metadata. Splitters copy the parent document's `source_path` and `metadata` to each split. Chunkers forward both into `Chunk`. **`VectorStorePipeline` never injects into `chunk.metadata`** — provenance (`source_id`, `source_hash`) lives in dedicated first-class `Chunk` fields, not in metadata.

### Pipeline

```
PARSING → PROCESSING ─┐
                       ↓
                   SPLITTING ← (uses rendering to measure token budget)
                       ↓
              (post-split PROCESSING, e.g. summarizers)
                       ↓
                   CHUNKING ← (chunker owns embedding strategy)
                       ↓
               Chunk
```

The pipeline classes are split at the sync boundaries: **`IngestPipeline`** (parser + processors + `source_id_fn`; Boundary 1) and **`ChunkPipeline`** (splitter + chunker + `chunk_id_fn` + `metadata_type`; Boundary 2). **`DocumentPipeline`** composes both for the direct path (`pipeline.ingest` / `pipeline.chunk`); its flat kwargs (`DocumentPipeline(splitter=…)`) delegate into freshly built sub-pipelines. A stage on the wrong side is a `TypeError` at construction (there is no parameter for it) — do not add boundary-validation predicates or runtime rejections.

The pipeline is **not strictly linear**:
- Splitting uses rendering internally to measure token budgets → splitting is rendering-aware.
- Post-split processors run after splitting on focused sub-documents, because splits are still `Document` objects.
- Chunking materializes `prompt_content` and `embedding_content`. `SimpleChunker` reuses prompt content; `LLMChunker` generates distinct embedding content via LLM.
- `Chunk` is the final materialization point — keep everything in `Document` form until then.

### Core Model

`Document` is the central model. It contains a list of `BaseElement` subclasses (Heading, Paragraph, Table, Image, DocumentList, Footnote, RawText). Elements hold content as `html` — the full HTML including the outer tag — with visual properties stored as **inline CSS** (e.g., `font-size`, `font-weight`, `text-align`). This CSS-in-HTML convention enables universal processors that work across all parsers without parser-specific fields. `html` is a **stored field with a normalizing validator** on Heading/Paragraph/Table/DocumentList/RawText, and a **derived property (with setter)** on Image and Footnote (their html is a projection of structured fields). Derived accessors (`text`, `inline_refs`, `level`, `innerhtml`, …) share one identity-keyed cached BeautifulSoup per element; `html_tag` is deliberately fresh-parse (it returns a mutable `Tag`). Elements use `validate_assignment=True`, so `element.html = value` re-normalizes. `merge_documents(first, second, *, metadata_policy)` / `join_documents` concatenate documents (there is no `|` operator).

Cross-document relationships use `ExternalRef` (parent/child/related); within-document references (images, footnotes, tables embedded in text) use `InlineRef` with `<ref id='...'/>` placeholders in HTML. The `Renderer` resolves these placeholders during rendering.

### Stage 1: Parsing (`src/ragdoc/parsing/`: `html/`, `pandoc/`, `xlsx/`, `azure_di/`, `mineru/`, `ragdoc_json/`)

Each parser converts a format → `Document` and sets parser-specific fields (e.g. `document.parser = "mineru"`). File provenance is stamped **centrally** by `parsing.load()` via `stamp_provenance` (`parser` if unset, `source_path`, `metadata["filename"]`) — individual loaders do not stamp it. Processors check `document.parser` to adjust behavior.

### Stage 2: Processing (`src/ragdoc/processing/`)

- `DocumentProcessor` (async ABC) — all processors subclass this; pure-sync processors simply don't `await`
- `ProcessingPipeline` — chains processors in sequence

Built-in processors: `HeadingLevelProcessor`, `TitleDetectionProcessor`, `LLMHeadingResolver`, `FootnoteProcessor`, `EmptyDocumentFilter`. Pluggable strategies use `Protocol` (e.g., `FootnoteResolver`).

Processors may return `None` to drop a document. `ProcessingPipeline` short-circuits on `None`; `IngestPipeline.run()` returns `None` and `DocumentPipeline.run()` returns `[]` chunks for filtered documents.

### Stage 3: Rendering (`src/ragdoc/rendering/`)

`Renderer` converts `Document` → formatted string. Element renderers use `singledispatch`. Two rendering strategies: `render_for_prompt` (full-fidelity text for LLM context) and `render_raw` (full fidelity with base64 images). Output formats: HTML, MARKDOWN, GFM, RST, PLAIN (via pypandoc).

### Stage 4–5: Splitting & Chunking (`src/ragdoc/splitting/`, `src/ragdoc/chunking/`)

`split_document()` splits by heading hierarchy and token budget. Every split copies the parent's `metadata` dict at construction (mutation-isolated); elements are shared across splits by reference (by design). The no-split path returns a shallow copy — the input document is never mutated. Output of chunking is `Chunk` (id, source_path, source_id, source_hash, content_hash, prompt_content, embedding_content, metadata) for loading into vector stores.

**Chunkers** (`src/ragdoc/chunking/`): `SimpleChunker` — pure rendering, one chunk per document (`embedding_content = prompt_content`). `LLMChunker` — calls LLM to produce N topic summaries, returns N chunks with the same `prompt_content` but distinct `embedding_content` per topic. Both implement the `Chunker` ABC.

### Stage 6: Sync (`src/ragdoc/pipeline/`)

Incremental synchronisation follows a **plan → apply** shape across two boundaries plus a direct path. Every sync pipeline exposes `plan()` (compute a reviewable `ChangeSet`, no store writes), `apply()` (write it), and `run = apply(plan(...))`.

**Two change-detection hashes** (first-class, never in `chunk.metadata`):
- `source_hash` — SHA-256 of the **raw source-file bytes** (Boundary 1 / direct path).
- `content_hash` — `Document.content_hash()`, a canonical-JSON hash over `(title, elements)` (pandoc-free, dependency-stable; Boundary 2). `None` ⇒ treated as "always changed".

**Provenance ownership.** `IngestPipeline` owns `source_id_fn` (`Path -> str`, default `p.name`); it stamps `document.source_id`, and `chunking/provenance.py` (`resolve_chunk_provenance`) is the single fallback chain stamping `source_id`/`source_hash`/`content_hash` onto every `Chunk`. `source_id` is never None (falls back `source_id → source_path → doc.id`); `source_hash` is **honestly optional** (`str | None`) — the file-byte hash from the sync pipelines' `hash_fn`, never faked from the content hash. **Chunk ids are minted by `ChunkPipeline.run`** (not by chunkers) via `mint_chunk_id` over `(source_id, split_sequence, chunk_ordinal, content_hash)` — deterministic and collision-free across identical-content splits; override with `ChunkPipeline(chunk_id_fn=…)`. The file-byte `hash_fn` is a sync concern living on the sync pipelines. **There is no `ProvenanceProcessor`** (do not add one).

`source_id_fn` strategies: `lambda p: p.name` (default), `str(p)` (full path), `str(p.relative_to(base_dir))` (portable). Duplicate source_ids raise `ValueError` before any processing.

**Pipelines** (each takes the boundary-appropriate pipeline type, so a misplaced stage is a `TypeError` at construction — unconstructible, not runtime-rejected):
- `VectorStorePipeline(pipeline: DocumentPipeline, vector_store, embedders=None, hash_fn=…, concurrency=10)` — **direct mode**: `plan(paths)` hashes files, compares `source_hash` via `vector_store.list_source_state()`, chunks changed files. `apply()` embeds **all** chunks first (non-destructive), then per source delete-then-upsert.
- `VectorStorePipeline.from_document_store(chunk: ChunkPipeline, vector_store, document_store, embedders=None, concurrency=10)` — **Boundary 2**: `plan(source_ids=None)` reads Documents from the store, compares `content_hash`, re-chunks changed ones via `ChunkPipeline.run` (a `ChunkPipeline` cannot carry a parser or processors).
- `DocumentStorePipeline(ingest: IngestPipeline, document_store, …)` — Boundary 1: parses/processes files into Documents and syncs them into a `DocumentStore` (payload is `Document`, no embedding; an `IngestPipeline` cannot carry a splitter or chunker). Uses `IngestPipeline.run`.

`delete_orphans` defaults to **`False`** (footgun guard: in direct mode only pass your complete corpus). `UpdateResult` (`processed`/`skipped`/`deleted`/`errors`) is keyed on `source_id`.

**Stores** (`src/ragdoc/pipeline/stores.py`):
- `VectorStore` protocol — `upsert` / `delete` / `delete_by_source` / `list_source_ids` / `list_source_state() -> {source_id: SourceState}`. (`get_source_hash` is retained but unused — superseded by the bulk `list_source_state`.) Index on `chunk.source_id`/`chunk.source_hash`, not metadata.
- `DocumentStore` protocol — `upsert` / `delete_by_source` / `get_document` / `list_source_ids` / `list_source_state`.
- `SourceState(source_hash, content_hash)` — bulk change-detection state.
- `LocalDocumentStore` — filesystem-backed `DocumentStore` (one JSON file per source); index-free, so manual edits to stored documents are detected at Boundary 2.

**`ChangeSet[T]`** (`changeset.py`) — serializable plan artifact (`to_add`/`to_update`/`to_delete`); `save()`/`load()` (call `load` on the concrete type, e.g. `ChangeSet[Chunk].load(path)`). Edit it between `plan()` and `apply()` for human-in-the-loop review.

### Extraction (`src/ragdoc/extraction/`)

Structured extraction is a **first-class typed stage**, not a processor. The `Extractor[PayloadT]` protocol (`extractor.py`) is `async extract(document) -> list[Mention[PayloadT]]` — extractors **return** mentions and never read or write `document.metadata` keys of their own (each mention's locator *copies* the document metadata). Do not route extraction results through metadata; the explicit opt-in `as_processor(extractor, metadata_key=...)` adapter exists solely for document-dump consumers.

- **`StructuredExtractor(payload_model, ...)`** (`structured.py`) — extracts one caller-supplied Pydantic model; the model's docstring + `Field` descriptions are the LLM schema.
- **`KnowledgeGraphExtractor(schema, ...)`** (`kg.py`) — multi-type node + edge extraction against a `GraphSchema`; rewrites chunk-local edge refs to real mention ids; recursive halving fallback on LLM failure; optional gleaning pass. **`GraphSchema.patterns` are enforced**: legal triples are injected into the system prompt (`render_patterns_prompt`) and every extracted edge is validated against `allowed_pattern_kinds(schema)` — violations drop (counted + warned) or raise per `ExtractionSettings.on_pattern_violation`. Every edge type must appear in ≥1 pattern (declaration-time rule); union sizes are checked at extractor construction against `settings.max_union_size`.
- **`ExtractionSettings`** (env prefix `EXTRACTION_`) — `system_prompt` and `request_timeout` are honored by both extractors; the validator never injects a built-in default prompt (each extractor resolves its own fallback). KG knobs: `gleaning`, `max_union_size`, `halving_max_depth`, `halving_min_chars`, `on_pattern_violation`.
- Shared extraction plumbing (`resolve_model`/`resolve_renderer`/`resolve_tokenizer`, `parse_with_retry`, `build_messages`) lives once in `extraction/_llm.py`; `parse_with_retry` is a thin adapter mapping `ExtractionSettings` onto `ragdoc.llm.call_structured` — do not inline retry loops in extractors.
- **`MentionStorePipeline(ingest: IngestPipeline, extractor, mention_store, ...)`** — the mention analogue of the sync pipelines (same `plan`/`apply`/`run`); Boundary-2 mode is the separate constructor `MentionStorePipeline.from_document_store(extractor, mention_store, document_store, ...)` (no ingest/parser/processors parameter exists there). Consumes `extract()` directly and rejects non-`Extractor` arguments with `TypeError`.

### LLM reliability layer (`src/ragdoc/llm.py`)

**Every LLM call in the library goes through `ragdoc.llm` — never call an OpenAI client directly, and never use the deprecated `beta.chat.completions` namespace.**

- **Protocols:** `ChatClient` (`chat.completions.parse`) and `EmbeddingsClient` (`embeddings.create`) are narrow structural protocols — call sites demand only what they use. `LLMClient` is their intersection and is what `RagdocConfig.openai_client` holds (typed, not `Any`; one configured `AsyncOpenAI` serves both). All three are exported from the `ragdoc` root.
- **Client resolution:** `resolve_openai_client(explicit)` = explicit → `get_config().openai_client` → `LLMNotConfiguredError`. Call it in `__init__` (fail-loud at construction), never lazily at request time. Exception: `LLMHeadingResolver` keeps its settings-based factory (base_url + api_key → `AsyncOpenAI`) as a documented layer above this chain.
- **One retry policy** (`retry_llm`, used by `call_structured` and `OpenAIEmbedder`): retry 429 (honoring `Retry-After`) / connection / timeout / 5xx with full-jitter exponential backoff; never retry other 4xx, `ValidationError`, non-openai errors, or `LLMRefusalError` (`parsed is None` — a refusal is deterministic, retrying burns tokens).
- **Degrade semantics are per-site, not unified** — the shared policy governs *transport* only. Preserve these failure products: heading resolver → `[]`, image summary → skip-with-warning, footnote resolver → `None`, entity reviewer → `ReviewResult()`; summarizer / `LLMChunker` / extractors → raise (KG halving catches and halves on top).
- No module-level `import openai` in `ragdoc/llm.py` (exception classes load lazily) — keep it that way for the Phase 8 extras split.
- All call sites pin `temperature=0.0`, so default models must stay in the gpt-4.x family (reasoning models reject non-default temperature).

## Collaboration Guidelines

- **Drive work to completion autonomously.** When the user has authorized a task, do every step you have the means to do — run the tests, install missing deps, fix the venv, capture snapshot values, amend commits. Don't hand the user shell commands to run when you can run them yourself. Don't gate-keep with "want me to X?" prompts on follow-through. Reserve confirmation for actions you genuinely cannot reverse on your own (pushing to a shared remote when no credentials are available, sending messages, destructive ops). The devcontainer has `.venv-devcontainer/` (set via `UV_PROJECT_ENVIRONMENT`); use it. If a tool isn't installed, install it. If the environment legitimately blocks you (no GPU, no network, no creds), state that in one sentence and structure the hand-off as one clean recipe — not multiple round-trips.
- **Before making structural model changes** (adding/removing fields on `Document`, `BaseElement` subclasses, or other core Pydantic models), stop and ask the user with all implications noted. Do not proceed autonomously.
- **When changing any public API** (signatures, imports, field names, class names), check all notebooks in `docs/notebooks/` and update them. A notebook that crashes or produces wrong output is a bug.

## Notebook Freshness

Notebooks in `docs/notebooks/` are living documentation and must execute correctly against the current codebase. Four layers keep them honest:

**1. Static check (`marimo check`)** — catches undefined names and reactive-graph errors before running anything:
```bash
uv run marimo check docs/notebooks/*.py
```

**2. pytest execution** — marimo notebooks are pure Python; pytest runs them and discovers `test_*` cells natively (no plugin needed):
```bash
uv run pytest docs/notebooks/
```

**3. `test_*` cells inside notebooks** — name assertion cells `test_*`. These serve dual purpose: pytest runs them in CI, and readers see exactly what outputs to expect without running the notebook. Good test cells are readable documentation:
```python
@app.cell
def test_document_parsed(doc):
    assert len(doc.elements) > 0, "document must have elements after parsing"
    assert doc.filename is not None
```

**4. `tests/test_notebooks.py`** — wrap every notebook with `app.run()` for assertions in the main test suite:
```python
from docs.notebooks.pipeline_walkthrough import app

@pytest.mark.anyio
async def test_pipeline_walkthrough():
    outputs, defs = app.run()
    assert "chunks" in defs
    assert len(defs["chunks"]) > 0
```

CI (`ci.yml`) runs all four layers. See `/documentation-guidelines` for the full documentation workflow.

## Plan Guidelines

Planning follows two phases:

### Phase 1: Design (`designs/DESIGN-*.md`)

When a non-trivial feature or change is requested, first write a design document to `designs/DESIGN-<topic>.md`. The document structure is:

1. **Problem statement** — what is being solved and why.
2. **Options** (1–4) — each option covers: approach summary, all touchpoints, and implications (performance, maintainability, compatibility, etc.).
3. **Summary** — pros/cons table and a concrete proposal.

Discuss the design with the user until a specific option is chosen.

### Phase 2: Implementation Plan (`plans/PLAN-*.md`)

Once an option is chosen, create `plans/PLAN-<topic>.md` with:

1. **Status** — one of: `ready` / `implementation phase N` / `complete` / `disregarded`. Update this as work progresses.
2. **Design reference** — link to the corresponding `designs/DESIGN-*.md`.
3. **Problem statement** — repeat the reason/motivation (plans must be self-contained).
4. **Implementation phases** — ordered phases, each with:
   - Tests first (TDD): use `/test-writing-guidelines` before writing implementation code.
   - Implementation steps.
5. **Documentation phase** — final phase to update CLAUDE.md, docstrings, and API docs.

## Code Conventions

- **Models:** Pydantic `BaseModel` with `Field(description="...")` on every field. Use `@property` (not `@computed_field`) for computed values.
- **Functions:** Extract reusable logic into standalone functions (not private methods), so they can be tested independently.
- **Type hints:** Strict — no `Any`, no `**kwargs`. Use `Callable` type aliases for injectable functions (e.g., `SizeToLevelMapper`).
- **CSS extraction functions:** Always return `None` when a property is absent (not a default value).
- **Naming:** Processors = `*Processor`, LLM-based async resolvers = `*Resolver`, Protocols describe capability, standalone functions = `verb_noun`.
- **Formatter / linter:** ruff (line-length=120; lint + import-sort + format). Type checking: basedpyright (standard mode). Config lives in `pyproject.toml`.

### Async-only conventions

The library is **async-only**. ABCs expose a single `async` method. Do not add sync wrappers. Callers are responsible for running the event loop (`asyncio.run(...)`).

- **Async tests:** Use `@pytest.mark.anyio`.
- **New ABCs:** expose only an `async` method; do not add a `_sync` variant.

## Dependency Management

```bash
uv add <package>           # production
uv add --dev <package>     # dev only
```

Optional extras: `llm` (openai + pillow), `tokenizers` (transformers), `xlsx` (pandas + openpyxl), `pdf` (pymupdf), `azure-di`, `pdf-mineru`, `qdrant`, `extraction`. The base install is lean — optional imports are lazy and fail with an actionable 'pip install ragdoc[<extra>]' message.
