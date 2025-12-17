# ragdoc

**ragdoc** is a Python library for chunking and preprocessing documents for LLM use cases.
It converts documents from various formats into structured, semantically-rich chunks ready for
vector stores and retrieval-augmented generation (RAG) pipelines.

## What ragdoc does

ragdoc provides a complete pipeline from raw document files to vector-store-ready chunks:

```
Raw Files (.docx, .html, .xlsx, .pdf, ...)
         ↓
     Parsing  →  Document (structured model)
         ↓
    Processing  →  Document (enriched: headings, summaries, footnotes)
         ↓
     Splitting  →  [Document, Document, ...]  (by heading or token budget)
         ↓
     Chunking  →  [Chunk, ...]  (prompt_content + embedding_content)
```

Every stage operates on the [`Document`](api/document.md) model — the single source of truth
throughout the pipeline.

## Rendering

ragdoc provides two element renderers:

| Renderer | Purpose |
|----------|---------|
| `render_for_prompt` | Full-fidelity text for LLM context windows and BM25 search |
| `render_raw` | Full fidelity, no transformation — for debugging |

`SimpleChunker` sets `embedding_content = prompt_content`. Only `LLMChunker` produces
distinct `embedding_content` by generating topic summaries via an LLM.

## Quick example

```python
from ragdoc.parsing import load
from ragdoc.rendering import Renderer, render_for_prompt, OutputFormat
from ragdoc.splitting import split_by_headings
from ragdoc.chunking import Chunk

# 1. Parse
document = await load("report.docx")

# 2. Split
sections = split_by_headings(document)

# 3. Chunk
renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
chunks = [
    Chunk(
        prompt_content=renderer.render(doc),
        metadata=doc.metadata,
    )
    for doc in sections
]
```

## Navigate the docs

- **[Getting Started](getting-started/installation.md)** — Install and run your first pipeline
- **[Guide](guide/overview.md)** — Understand the architecture and each pipeline stage
- **[API Reference](api/document.md)** — Complete reference for all public classes and functions
- **[Tutorials](tutorials.md)** — interactive marimo notebooks with worked examples
