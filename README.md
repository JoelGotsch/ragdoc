# Ragdoc

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Ragdoc is an **ingestion library** for retrieval-augmented generation. It parses common document formats (DOCX, HTML, XLSX, PDF) into structured `Document` objects, processes and splits them, and chunks them into dual-representation retrieval units. `DocumentPipeline` wires the stages together; `VectorStorePipeline` adds plan/apply **incremental sync** into a vector store (Qdrant supported out of the box), with typed metadata carried end-to-end. Ragdoc is ingestion-only — retrieval/query is out of scope; your retrieval code reads the documented payload fields.

## Installation

```bash
uv add ragdoc        # or: pip install ragdoc
```

The first PyPI release is 0.1.0. Optional extras:

- `azure-di` — Azure Document Intelligence PDF parsing
- `pdf-mineru` — MinerU-based PDF parsing
- `qdrant` — Qdrant vector-store integration
- `extraction` — typed knowledge-graph / structured extraction

Full documentation: <https://joelgotsch.github.io/ragdoc/>

## Quickstart

Sync a folder of HTML files into a local Qdrant collection. Requires `ragdoc[qdrant]`, the `openai` package, and an OpenAI API key in the environment.

```python
import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from ragdoc import DocumentPipeline, TokenSplitter, VectorStorePipeline
from ragdoc.pipeline import EmbedderConfig, OpenAIEmbedder
from ragdoc.integrations.vector_stores import QdrantVectorStore

async def main() -> None:
    store = await QdrantVectorStore.create(AsyncQdrantClient(url="http://localhost:6333"), "docs", vector_size=1536)
    embedders = {"dense": EmbedderConfig(OpenAIEmbedder(AsyncOpenAI(), model="text-embedding-3-small"))}
    sync = VectorStorePipeline(DocumentPipeline(splitter=TokenSplitter(max_tokens=4000)), store, embedders=embedders)
    result = await sync.run(Path("corpus").glob("**/*.html"))   # re-run any time: unchanged files are skipped
    print(f"synced={len(result.processed)} unchanged={len(result.skipped)}")

asyncio.run(main())
```

Re-runs are cheap: ragdoc hashes each source file and skips unchanged ones (hash-based incremental sync).

## Core Philosophy

The library is built around four fundamental ideas:

**1. `Document` is the single source of truth.**
Every stage operates on structured `Document` objects containing typed elements (`Heading`, `Paragraph`, `Table`, …). Raw formats are parsed into `Document` once; all subsequent work — enrichment, splitting, rendering — operates on this structured representation. Never pass rendered strings between stages.

**2. The library is async-only.**
All public entry points are `async`. There are no sync convenience wrappers. Use `asyncio.run(...)` or an existing event loop to call them. This keeps the API surface small and avoids the hidden cost of spawning threads to run an event loop inside a sync wrapper.

**3. Chunkers own the embedding content strategy.**
Retrieval-augmented pipelines need two text representations per chunk: `prompt_content` (full-fidelity text for LLM context / BM25) and `embedding_content` (compact semantic text for dense vector search). The chunker decides how to produce them:

- **`SimpleChunker`** — sets `embedding_content = prompt_content`. Use when a single rendering suffices.
- **`LLMChunker`** — generates N distinct `embedding_content` strings (one per topic) via LLM, all sharing the same `prompt_content`.

Rendering uses `render_for_prompt` (full-fidelity structured text) or `render_raw` (with base64 images). There is no separate embedding renderer.

**4. Source provenance and metadata propagate through the entire pipeline.**
Parsers set `Document.source_path` and write the bare filename into `document.metadata["filename"]`. Metadata carries both user-defined custom data and library-written informational fields; processors must not drop or overwrite it, splitters copy it to every split, and chunkers forward it into `Chunk` — anything you put in `document.metadata` arrives on the final chunks. Fields that drive library behavior (change detection, deletion) are first-class on `Chunk` instead (`source_id`, `source_hash`, `content_hash`), never metadata keys, and `VectorStorePipeline` never injects into `chunk.metadata`.

## Architecture

The library stages are:

```
PARSING → PROCESSING ─┐
                       ↓
                   SPLITTING ← (rendering used to measure token budget)
                       ↓
              (post-split PROCESSING, e.g. summarizers)
                       ↓
                   CHUNKING ← (chunker owns embedding strategy)
                       ↓
               Chunk
```

> The pipeline is **not strictly linear**. Splitting relies on rendering to measure token budgets. Processors such as document summarizers typically run *after* splitting so they operate on focused sub-documents. Chunking produces `prompt_content` and `embedding_content` — `SimpleChunker` reuses prompt content, while `LLMChunker` generates distinct embedding content via LLM. The `Document` model is the unifying interface throughout.

### Stage 1: Parsing

Convert source files into `Document` objects. Each parser sets `document.parser` for provenance.

```python
from ragdoc import load

document = await load("report.docx")
document = await load("page.html")
document = await load("data.xlsx")
```

Supported formats: `.html`, `.docx`/`.doc`, `.xlsx`, `.json` (Azure DI output), PDF.

PDF parsing currently requires the `azure-di` extra (plus Azure credentials) or the `pdf-mineru` extra — there is no zero-config PDF path yet; a local zero-config PDF parser is planned. With Azure DI:

```python
from ragdoc import RagdocConfig, configure, load

async with configure(RagdocConfig(azure_key="...", azure_endpoint="...")):
    document = await load("report.pdf")
```

### Stage 2: Processing

`DocumentProcessor` (async ABC) transforms documents.
Chain processors with `ProcessingPipeline`.

```python
from ragdoc.processing import (
    FootnoteProcessor,
    HeadingLevelProcessor,
    ProcessingPipeline,
    TitleDetectionProcessor,
)

pipeline = ProcessingPipeline([
    HeadingLevelProcessor(),
    TitleDetectionProcessor(),
    FootnoteProcessor(),
])
document = await pipeline.process(document)
```

LLM-based processors (`LLMHeadingResolver`, `LLMFootnoteResolver`, …) require an `AsyncOpenAI` client (or set `openai_client` in `RagdocConfig`).

### Stage 3: Rendering

`Renderer` converts a `Document` to a formatted string. Element renderers are pluggable via `singledispatch`.

```python
from ragdoc.rendering import Renderer, OutputFormat, render_for_prompt

# For LLM prompts (full-fidelity content)
renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
text = renderer.render(document)
```

Output formats: `HTML`, `MARKDOWN` (`md`), `GFM`, `RST`, `PLAIN`.

### Stage 4: Splitting

Split large documents by heading hierarchy or token count. `split_document()` is the main entry point: it splits by heading hierarchy, then enforces a token budget.

```python
from ragdoc import split_document
from ragdoc.rendering import Renderer

sections = split_document(document, renderer=Renderer(), max_tokens=4000)
```

`split_by_headings` and `split_hierarchical` (in `ragdoc.splitting`) expose the structural strategies directly. Parent-child relationships between split documents are tracked via `ExternalRef`.

### Stage 5: Chunking

Create `Chunk` objects ready for vector store loading.

```python
from ragdoc import Chunk
```

`Chunk` holds `id`, `source_path`, `source_id`, `source_hash`, `content_hash`, `prompt_content`, `embedding_content`, `named_embeddings`, and `metadata`.

## Incremental Sync

`VectorStorePipeline` (and `DocumentStorePipeline`) follow a **plan → apply** shape: `plan()` computes a reviewable `ChangeSet` without touching the store, `apply()` writes it, and `run()` does both.

```python
plan = await sync.plan(paths)       # ChangeSet: to_add / to_update / to_delete — no writes
plan.save("changeset.json")         # optionally review or edit before applying
result = await sync.apply(plan)     # embed, then per-source delete-and-upsert
```

Change detection uses two hashes: `source_hash` (SHA-256 of the raw file bytes) when syncing from files, and `content_hash` (renderer-stable `Document.content_hash()`) when syncing from a `DocumentStore`. Unchanged sources are skipped; changed ones are re-chunked, re-embedded, and replaced atomically per source. Orphan deletion is off by default (`delete_orphans=False`) — enable it only when you pass your complete corpus. The stored payload schema (the `Chunk` fields above plus your metadata) is documented at <https://joelgotsch.github.io/ragdoc/> for retrieval clients to read.

## Core Model

`Document` contains a list of `BaseElement` subclasses:
`Heading`, `Paragraph`, `Table`, `Image`, `DocumentList`, `Footnote`, `RawText`.

Visual properties are stored as **inline CSS** in each element's `html` (e.g. `font-size`, `text-align`).
Cross-document relationships use `ExternalRef` (parent/child/related).
Within-document inline references use `InlineRef` with `<ref id='...'/>` placeholders.

## Design Implications

The `Document`-centric design, combined with chunkers owning the embedding strategy, has direct consequences for how the library should be extended:

- **Processors should write back into `Document` structure, not into strings.** An LLM image summarizer stores its output in `image.text_representation`, not as a rendered string. Enrichment results stay in `Document` form until chunking materializes them.

- **Splits are still `Document` objects.** This is deliberate — it allows processors (e.g. a section summarizer, a topic classifier) to run after splitting on focused sub-documents, then chunking renders the enriched result. Processors do not need to know whether they are processing a full document or a split.

- **Token-budget splitting must render first.** Because `split_document()` measures prompt-content length in tokens, it internally invokes the prompt renderer. Splitting is therefore rendering-aware and belongs after the main processing pipeline, not before.

- **`Chunk` is the materialization point.** Keep everything in structured `Document` form for as long as possible. Only at the final chunking step are `prompt_content` and `embedding_content` strings produced. This maximizes reusability: the same `Document` can be re-chunked with different chunkers without reprocessing.

## Configuration

```python
from ragdoc import RagdocConfig, configure

async with configure(RagdocConfig(
    azure_key="...",
    azure_endpoint="...",
    download_images=True,
    openai_client=AsyncOpenAI(...),
)):
    document = await load("file.pdf")
```

`configure()` uses `contextvars.ContextVar` and is safe for both sync and async use.

## CLI

The `ragdoc` command wraps parsing and rendering:

```bash
# Parse files to Document JSON
ragdoc parse file.docx report.html --output ./out

# Parse and render to text (formats: html, md, gfm, rst, plain)
ragdoc chunk file.html --format md --output ./out
```

## Terminology

- **Document**: Structured representation of content (paragraphs, headings, images, tables, …).
- **DocumentProcessor**: Transforms a Document (heading levels, footnotes, summaries, …).
- **Parser**: Converts raw input into a Document. Sets `document.parser` for provenance.
- **Renderer**: Converts a Document to a formatted string (HTML, Markdown, plain text, …).
- **Prompt-Content**: Text included in the LLM prompt after retrieval — full context, narrow topic.
- **Embedding-Content**: Text embedded for semantic search — typically a short summary.
- **Splitter**: Splits a Document into smaller Documents by heading hierarchy or token budget.
- **Chunker**: Combines a prompt renderer and an embedding renderer to produce `Chunk` objects.
- **Chunk**: A chunk ready to load into a vector store — holds both prompt and embedding content.

### Comparison with other RAG libraries

Most RAG frameworks (LangChain, LlamaIndex, …) use **"chunking"** to mean splitting a document into smaller text pieces — what ragdoc calls **splitting**. Ragdoc reserves the term **chunking** for a distinct, later stage that those frameworks typically do not expose as a separate concept:

| Stage | ragdoc term | Typical industry term |
|---|---|---|
| Subdivide document by structure / token budget | **splitting** | chunking, text splitting |
| Materialize retrieval units with dual representations | **chunking** | *(not a distinct stage)* |

The key distinction: in most libraries a "chunk" is just a slice of text — one string, used for both embedding and retrieval. In ragdoc, a `Chunk` always carries **two independent text representations** (`prompt_content` and `embedding_content`) rendered from the same structured `Document`. This separation is a first-class design choice, not an afterthought.

`LLMChunker` pushes this further: it produces **N `Chunk`s from a single document**, one per LLM-generated topic summary. Each chunk shares the same `prompt_content` (the full section text) but has a distinct `embedding_content` (a focused topic summary). This is analogous to what some frameworks call *multi-vector indexing* or *summary-augmented retrieval*, but here it is modelled explicitly as N chunks rather than a side-channel index.
