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
uv run basedpyright                  # type check (strict)
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

The pipeline is **not strictly linear**:
- Splitting uses rendering internally to measure token budgets → splitting is rendering-aware.
- Post-split processors run after splitting on focused sub-documents, because splits are still `Document` objects.
- Chunking materializes `prompt_content` and `embedding_content`. `SimpleChunker` reuses prompt content; `LLMChunker` generates distinct embedding content via LLM.
- `Chunk` is the final materialization point — keep everything in `Document` form until then.

### Core Model

`Document` is the central model. It contains a list of `BaseElement` subclasses (Heading, Paragraph, Table, Image, DocumentList, Footnote, RawText). Elements hold content as `innerhtml` (HTML string) with visual properties stored as **inline CSS** (e.g., `font-size`, `font-weight`, `text-align`). This CSS-in-HTML convention enables universal processors that work across all parsers without parser-specific fields.

Cross-document relationships use `ExternalRef` (parent/child/related); within-document references (images, footnotes, tables embedded in text) use `InlineRef` with `<ref id='...'/>` placeholders in HTML. The `Renderer` resolves these placeholders during rendering.

### Stage 1: Parsing (`src/ragdoc/parsing/`: `html/`, `pandoc/`, `xlsx/`, `azure_di/`, `textract/`, `mineru/`)

Each parser converts a format → `Document`, sets `document.parser` (provenance string, e.g., `"mineru"`, `"azure_di"`), sets `document.source_path`, and writes `document.metadata["filename"] = path.name`. Processors check `document.parser` to adjust behavior.

### Stage 2: Processing (`src/ragdoc/processing/`)

- `DocumentProcessor` (async ABC) — all processors subclass this; pure-sync processors simply don't `await`
- `ProcessingPipeline` — chains processors in sequence

Built-in processors: `HeadingLevelProcessor`, `TitleDetectionProcessor`, `LLMHeadingResolver`, `FootnoteProcessor`, `EmptyDocumentFilter`. Pluggable strategies use `Protocol` (e.g., `FootnoteResolver`).

Processors may return `None` to drop a document. `ProcessingPipeline` short-circuits on `None`; `DocumentPipeline._process_one()` returns `[]` chunks for filtered documents.

### Stage 3: Rendering (`src/ragdoc/rendering/`)

`Renderer` converts `Document` → formatted string. Element renderers use `singledispatch`. Two rendering strategies: `render_for_prompt` (full-fidelity text for LLM context) and `render_raw` (full fidelity with base64 images). Output formats: HTML, MARKDOWN, GFM, RST, PLAIN (via pypandoc).

### Stage 4–5: Splitting & Chunking (`src/ragdoc/splitting/`, `src/ragdoc/chunking/`)

`split_document()` splits by heading hierarchy and token budget. Output of chunking is `Chunk` (id, source_path, source_id, source_hash, prompt_content, embedding_content, metadata) for loading into vector stores.

**Chunkers** (`src/ragdoc/chunking/`): `SimpleChunker` — pure rendering, one chunk per document (`embedding_content = prompt_content`). `LLMChunker` — calls LLM to produce N topic summaries, returns N chunks with the same `prompt_content` but distinct `embedding_content` per topic. Both implement the `Chunker` ABC.

### Stage 6: Vector Store Update (`src/ragdoc/pipeline/`)

`VectorStorePipeline` wraps a `DocumentPipeline` with incremental updates. It sets two first-class `Chunk` fields before upsert:

- `chunk.source_id` — source identity key derived from the source `Path` via `source_id_fn` (default: `path.name`).
- `chunk.source_hash` — SHA-256 hex digest of the raw file bytes.

These are **not injected into `chunk.metadata`**. The `metadata` dict remains exclusively for user-defined data.

**`source_id_fn` strategies** (passed to `VectorStorePipeline`):

| Strategy | Expression | Use case |
|---|---|---|
| Filename (default) | `lambda p: p.name` | Simple, flat folders |
| Full path | `lambda p: str(p)` | Multi-dir, single machine |
| Relative path | `lambda p: str(p.relative_to(base_dir))` | Portable multi-dir |

If two `Path` objects map to the same `source_id`, `run()` raises `ValueError` **before any processing begins**.

**`VectorStore` protocol** (`src/ragdoc/pipeline/stores.py`) — typed, source-aware methods:
- `upsert(chunks)` / `delete(ids)` — CRUD by chunk ID.
- `get_source_hash(source_id) -> str | None` — returns stored hash, or `None` if source unknown.
- `delete_by_source(source_id)` — removes all chunks for a source.
- `list_source_ids() -> set[str]` — enumerates all known sources.

Implementations must index on `chunk.source_id` / `chunk.source_hash`, not `chunk.metadata`.

### Refactor Status

The codebase is mid-refactor (see `REFACTOR_TODO.md`). Phases 1–3 are complete. Phase 4 (updating parsers to set `document.parser` and store CSS) and Phase 5 (folder restructuring to `parsing/`, `splitting/`, `chunking/`) are in progress. `mineru/middleware/` is deprecated in favor of `processing/`.

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

CI (`tests.yaml`) runs all four layers. See `/documentation-best-practices` for the full documentation workflow.

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
- **Formatter / linter:** ruff (line-length=120; lint + import-sort + format). Type checking: basedpyright (strict). Config lives in `pyproject.toml`.

### Async-only conventions

The library is **async-only**. ABCs expose a single `async` method. Do not add sync wrappers. Callers are responsible for running the event loop (`asyncio.run(...)`).

- **Async tests:** Use `@pytest.mark.anyio`.
- **New ABCs:** expose only an `async` method; do not add a `_sync` variant.

## Dependency Management

```bash
uv add <package>           # production
uv add --dev <package>     # dev only
```

Optional extras: `azure-di`, `pdf-mineru`, `qdrant`. Local dependency: `aa-utils` at `C:/DEV/aa-utils`.
