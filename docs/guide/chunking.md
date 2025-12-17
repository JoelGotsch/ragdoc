# Chunking

> Run interactively: `marimo edit docs/notebooks/chunking.py`

Chunking is the final stage of the pipeline. It materializes a [`Document`](document-model.md)
(or split section) into a [`Chunk`](../api/chunking.md#chunk) — a
flat, self-contained record ready to be loaded into a vector store.

## What is a Chunk?

A [`Chunk`](../api/chunking.md#chunk) is a Pydantic model with:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (UUID string; minted deterministically by `ChunkPipeline.run`) |
| `prompt_content` | Full-fidelity text for LLM context windows and BM25 search |
| `embedding_content` | Compact semantic text for dense vector search |
| `source_path` | Full path to the source file (propagated from `Document.source_path`) |
| `source_id` | Stable source identity used for incremental sync (from `IngestPipeline`'s `source_id_fn`) |
| `source_hash` | SHA-256 of the raw source-file bytes (`None` outside the sync pipelines) |
| `content_hash` | Canonical-JSON hash of the parsed document content |
| `metadata` | Key-value metadata from the source document (includes `metadata["filename"]`, set by parsers) |
| `named_embeddings` | Embedding vectors keyed by embedder name (populated by `VectorStorePipeline`) |
| `created_at` | Timestamp of creation |

For `SimpleChunker`, `embedding_content` equals `prompt_content`. Only `LLMChunker` produces distinct `embedding_content` per topic.

## Creating chunks

Combine the [`Renderer`](../api/rendering.md#renderer) with `Chunk`:

```python
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
from ragdoc.chunking import Chunk
from ragdoc.splitting import split_by_headings

sections = split_by_headings(document)

renderer = Renderer(
    format=OutputFormat.MARKDOWN,
    element_renderer=render_for_prompt,
    include_title=True,
    metadata_keys=["source", "date"],
)

# SimpleChunker does exactly this: embedding_content = prompt_content
chunks = [
    Chunk(
        prompt_content=(rendered := renderer.render(section)),
        embedding_content=rendered,
        source_path=section.source_path,
        metadata=section.metadata,
    )
    for section in sections
]
```

In practice, prefer `SimpleChunker` (or a `ChunkPipeline`, which also mints deterministic
chunk ids and stamps provenance) over constructing `Chunk` objects by hand:

```python
from ragdoc.chunking import SimpleChunker

chunker = SimpleChunker(prompt_renderer=renderer)
chunks = [chunk for section in sections for chunk in await chunker.chunk(section)]
```

## Provenance fields

`source_path`, `source_id`, `source_hash`, and `content_hash` are explicit first-class
fields on `Chunk`; the bare filename lives in `metadata["filename"]` (written by parsers
onto the `Document` and propagated automatically through splitting):

```python
# Parsers set these; they propagate through splits automatically:
print(document.metadata["filename"])  # "annual_report_2024.docx"
print(document.source_path)          # "/data/annual_report_2024.docx"

# When constructing chunks manually, forward them explicitly:
chunk = Chunk(
    prompt_content=...,
    embedding_content=...,
    source_path=section.source_path,
    metadata=section.metadata,  # carries metadata["filename"]
)
```

## Metadata inheritance

When you split a document, child sections inherit parent metadata. Set metadata on the
source document before parsing and it will flow through to all chunks:

```python
document.metadata["department"] = "Finance"
document.metadata["year"] = 2024

# All chunks will have department and year in their metadata
```

## Loading into a vector store

`Chunk` is designed to map directly to vector store document schemas. Use
[`VectorStorePipeline`](../api/pipeline.md#vectorstorepipeline) to embed and
upsert chunks with incremental change detection.

## Prompt vs embedding content

A `Chunk` carries two content fields:

- `prompt_content` is what you pass to the LLM in RAG retrieval (full-fidelity text).
- `embedding_content` is what you encode with your embedding model.

With `SimpleChunker`, `embedding_content` equals `prompt_content` — one renderer is used
for both. Only `LLMChunker` produces distinct `embedding_content` by generating N topic
summaries per document, each becoming a separate chunk with unique `embedding_content`
but shared `prompt_content`.

## See Also

- [API Reference: Chunking](../api/chunking.md)
- [API Reference: Integrations](../api/integrations.md) — vector store integrations
- [Rendering Guide](rendering.md) — understanding the rendering system
- [Splitting Guide](splitting.md) — preparing documents before chunking
