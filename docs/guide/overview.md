# Architecture Overview

ragdoc is built around a single unifying principle: **`Document` is the single source of truth.**
Every stage of the pipeline operates on structured `Document` objects — never on rendered strings.

## Pipeline

```
Raw File
   ↓
PARSING  ──────────────────────→  Document
   ↓                               (elements: Heading, Paragraph, Table, Image, ...)
PROCESSING  ───────────────────→  Document  (enriched)
   ↓                               (headings resolved, images described, footnotes linked)
SPLITTING  ─────────────────────→  [Document, ...]
   ↓                               (each section is still a Document with ExternalRef links)
CHUNKING  ──────────────────────→  [Chunk, ...]
                                   (prompt_content + embedding_content materialized)
```

The pipeline is **not strictly linear**:

- Splitting uses rendering internally to measure token budgets.
- Post-split processors (e.g., `ImageSummaryProcessor`) run after splitting on focused sub-documents.
- `Chunk` is the final materialization point — keep everything as `Document` until then.
- A processor can return `None` to **drop the document** entirely. `ProcessingPipeline` short-circuits on `None` (no subsequent processors run) and `DocumentPipeline` returns zero chunks for that source file. Use `EmptyDocumentFilter` for the common case of skipping empty parses.

## Rendering and chunking

Every `Chunk` carries two text representations of the same content:

| Field | Source | Purpose |
|-------|--------|---------|
| `prompt_content` | `render_for_prompt` | Full-fidelity text for LLM context windows and BM25 keyword search |
| `embedding_content` | varies by chunker | Compact semantic text for dense vector search |

`SimpleChunker` sets `embedding_content = prompt_content` — a single renderer is used for both.
`LLMChunker` produces distinct `embedding_content` by generating N topic summaries per document,
each becoming a separate chunk with unique `embedding_content` but shared `prompt_content`.

ragdoc provides two element renderers: `render_for_prompt` (the primary renderer) and
`render_raw` (for debugging).

## CSS-in-HTML convention

Document elements store content as `html` (full HTML including the outer tag) with visual properties stored as
**inline CSS** — for example `font-size`, `font-weight`, `text-align`. This convention enables
processors to work uniformly across all parsers without parser-specific fields.

## Terminology note: splitting vs. chunking

Most RAG frameworks (LangChain, LlamaIndex, …) use **"chunking"** to mean splitting a document into smaller text pieces — what ragdoc calls **splitting**. Ragdoc reserves the term **chunking** for a distinct, later stage:

| Stage | ragdoc term | Typical industry term |
|---|---|---|
| Subdivide document by structure / token budget | **splitting** | chunking, text splitting |
| Materialize retrieval units with dual representations | **chunking** | *(not a distinct stage)* |

In most libraries a "chunk" is a single string used for both embedding and retrieval. In ragdoc a `Chunk` always carries **two content fields** — `prompt_content` for LLM context windows and `embedding_content` for dense vector search. With `SimpleChunker`, both fields contain the same rendered text. With `LLMChunker`, `embedding_content` is a distinct LLM-generated topic summary.

`LLMChunker` extends this further by producing **N `Chunk`s from one document**, one per LLM-generated topic summary, each with the same `prompt_content` but a distinct `embedding_content`. This is analogous to *multi-vector indexing* in other frameworks, but modelled explicitly as N chunks.

## Async-only API

All public entry points in ragdoc are `async`. There are no sync convenience wrappers exposed to library users. Call them with `asyncio.run(...)` or from within an existing event loop:

```python
import asyncio

document = asyncio.run(pipeline.process(document))
chunks = asyncio.run(chunker.chunk(document))
```

This keeps the API surface small and unambiguous. Sync wrappers that spawn a thread just to call `asyncio.run` are hidden complexity — the library does not provide them.

## Guide reading order

The guide is structured to follow the pipeline:

1. **[Document Model](document-model.md)** — The `Document` structure and element types
2. **[Parsing](parsing.md)** — Converting files into `Document` objects
3. **[Rendering](rendering.md)** — Converting `Document` objects into text
4. **[Splitting](splitting.md)** — Dividing a `Document` into sections
5. **[Chunking](chunking.md)** — Materializing `Chunk` objects for vector stores
