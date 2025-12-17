# Pipeline

> Run interactively: `marimo edit docs/notebooks/pipeline.py`

The `ragdoc.pipeline` module provides a high-level facade that wires together
all four stages — **parse → process → split → chunk** — into a single call.

## When to use it

Use `DocumentPipeline` when you want to go from a file on disk to
[`Chunk`](chunking.md) objects in one step, without manually threading
`Document` objects through each stage yourself.

Use the individual stage APIs ([Parsing](parsing.md), [Processing](../api/processing.md),
[Splitting](splitting.md), [Chunking](chunking.md)) when you need fine-grained
control — custom parsers that don't fit a `Path → Document` signature, partial
re-processing, or stage-level debugging.

---

## Scenario A — Linear pipeline

### Minimal example

```python
from pathlib import Path
from ragdoc.pipeline import DocumentPipeline

pipeline = DocumentPipeline()
chunks = await pipeline.run(Path("report.docx"))
```

`DocumentPipeline` is the composition of two sub-pipelines, split at the sync boundaries:

- **`IngestPipeline`** (`pipeline.ingest`) — parse → process → stamp `source_id`. Boundary 1.
- **`ChunkPipeline`** (`pipeline.chunk`) — split → chunk → mint chunk ids. Boundary 2.

Flat kwargs (`DocumentPipeline(splitter=..., processors=...)`) delegate into freshly built
sub-pipelines; you can equivalently pass pre-built ones
(`DocumentPipeline(ingest=IngestPipeline(...), chunk=ChunkPipeline(...))`). A stage on the
wrong side is a `TypeError` at construction — an `IngestPipeline` has no splitter/chunker
parameter, a `ChunkPipeline` has no parser/processors parameter.

`DocumentPipeline` defaults:

| Stage | Default |
|-------|---------|
| Parser | [`load`](../api/parsing.md#load) — picks parser from file extension via the registry |
| Processors | none (identity pass-through) |
| Splitter | none — whole document → one chunk |
| Chunker | `SimpleChunker` — one chunk per document |

### Adding a splitter

To split documents into retrieval-sized chunks, pass a
[`TokenSplitter`](../api/pipeline.md#TokenSplitter).
Chunk size is independent of your LLM's context window — chunks are sized
for retrieval quality, typically 1/50th to 1/10th of the context window.
Test different `max_tokens` values for your use case:

```python
from ragdoc.pipeline import DocumentPipeline, TokenSplitter

pipeline = DocumentPipeline(
    splitter=TokenSplitter(max_tokens=4000),
)
chunks = await pipeline.run(Path("annual_report.docx"))
```

Each split section becomes an independent chunk.

### Adding processors

Pass processors as a list or a `ProcessingPipeline`:

```python
from ragdoc.pipeline import DocumentPipeline, TokenSplitter
from ragdoc.processing import HeadingLevelProcessor, TitleDetectionProcessor

pipeline = DocumentPipeline(
    processors=[
        HeadingLevelProcessor(),
        TitleDetectionProcessor(),
    ],
    splitter=TokenSplitter(max_tokens=4000),
)
```

#### Metadata: extend, don't replace

When a processor sets metadata, always write individual keys; never assign a new dict:

```python
# correct — preserves filename and any keys set by earlier processors
document.metadata["document_name"] = _extract_name(document)
document.metadata["document_date"] = _extract_date(document)

# wrong — silently discards filename and all prior processor output
document.metadata = {"document_name": _extract_name(document)}
```

`document.metadata["filename"]` is set by every parser and is the primary provenance
key used throughout the pipeline. Replacing the dict loses it silently.

### Custom parser

Any callable `(Path) -> Document` satisfies the `Parser` protocol:

```python
from pathlib import Path
from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline

async def my_parser(path: Path) -> Document:
    # e.g. call an internal API, parse a custom format, …
    return Document(...)

pipeline = DocumentPipeline(parser=my_parser)
```

### Deterministic chunk IDs

Chunk ids are minted by `ChunkPipeline.run` — the single id authority —
using `ragdoc.chunking.provenance.mint_chunk_id` over
`(source_id, split_sequence, chunk_ordinal, content_hash)`.  The same file content always
produces the same chunk IDs (idempotent vector-store upserts), and two identical-content
splits of one source still get distinct ids:

```python
chunks1 = await pipeline.run(Path("report.docx"))
chunks2 = await pipeline.run(Path("report.docx"))

assert chunks1[0].id == chunks2[0].id  # always True for the same content
```

Override the scheme with `ChunkPipeline(chunk_id_fn=...)` (or the delegating
`DocumentPipeline(chunk_id_fn=...)`)
(`(source_id, split_sequence, chunk_ordinal, content_hash) -> str`).  Chunkers used
standalone (outside a pipeline) leave `Chunk.id` at its uuid4 default.

---

## Scenario D — Concurrent processing and streaming

### Processing many files

`run_many` fans out across a list of paths and returns a
[`PipelineResult`](../api/pipeline.md#PipelineResult) with all chunks
aggregated:

```python
from pathlib import Path
from ragdoc.pipeline import DocumentPipeline, TokenSplitter

pipeline = DocumentPipeline(splitter=TokenSplitter())
paths = list(Path("docs/").glob("**/*.docx"))

result = await pipeline.run_many(paths, concurrency=8)
print(f"Produced {len(result.chunks)} chunks from {len(paths)} files")
```

### Error handling

By default errors propagate immediately (`on_error="raise"`).
Use `on_error="skip"` to collect failures and continue:

```python
pipeline = DocumentPipeline(
    splitter=TokenSplitter(),
    on_error="skip",
)
result = await pipeline.run_many(paths)

for path, exc in result.errors:
    print(f"  FAILED {path}: {exc}")
```

### Streaming (memory-efficient)

`stream` yields a batch of chunks per file as it completes, so you can hand
off each batch (e.g. to a vector store) before the next file starts:

```python
async for batch in pipeline.stream(paths, concurrency=4):
    await vector_store.upsert(batch)
```

---

## TokenSplitter reference

`max_tokens` controls retrieval chunk size, not LLM context-window fit.
Chunks should typically be 1/50th to 1/10th of the context window — the right
value depends on your retrieval quality requirements and should be tested
empirically.

```python
TokenSplitter(
    max_tokens=7000,    # token budget per split (retrieval-quality sizing, not context-window fit)
    overlap_tokens=200, # overlap between consecutive splits
    renderer=None,      # Renderer for token measurement; defaults to MARKDOWN + render_for_prompt
    tokenizer=None,     # tiktoken cl100k_base by default
)
```

---

## Scenario B — Incremental update

`VectorStorePipeline` wraps `DocumentPipeline` and adds hash-based change detection
so that a corpus of documents can be kept in sync with a vector store
incrementally — only changed or new files are re-processed.

The **vector store is the single source of truth** for provenance.  Each chunk
carries `source_id` and `source_hash` as first-class fields, so the pipeline
can detect unchanged sources, clean up stale chunks, and discover orphans —
all without any local state file.

### Setup

```python
from pathlib import Path
from ragdoc.pipeline import VectorStorePipeline, TokenSplitter

vs_pipeline = VectorStorePipeline(
    pipeline=DocumentPipeline(splitter=TokenSplitter()),
    vector_store=my_vector_store,   # any VectorStore implementation
)
```

### First run — all files processed

```python
paths = list(Path("docs/").glob("**/*.docx"))

result = await vs_pipeline.run(paths)
print(f"processed: {len(result.processed)}, skipped: {len(result.skipped)}")
# processed: 42, skipped: 0
```

### Subsequent runs — unchanged files skipped

If the raw file bytes have not changed since the last run, the file is skipped
without parsing or uploading:

```python
result = await vs_pipeline.run(paths)
# processed: 0, skipped: 42   (nothing changed)
```

### Handling modifications and deletions

`VectorStorePipeline.run` handles new and modified files automatically on every call:

| Case | What happens |
|------|-------------|
| **New file** | Parsed, chunked, upserted; `source_id` / `source_hash` / `content_hash` set on each chunk. |
| **Modified file** (byte hash changed) | Old chunks deleted by `source_id`, file re-processed, new chunks upserted. |
| **Removed file** (path no longer in source list) | Kept by default; deleted only with `delete_orphans=True`. |

Orphan deletion is **opt-in** and must be requested explicitly:

```python
# ⚠ delete_orphans=True deletes every source NOT in `paths`. Only safe when `paths`
# is your COMPLETE corpus — an incremental subset would wipe everything else.
result = await vs_pipeline.run(all_paths, delete_orphans=True)
print(result.deleted)  # ["old_report.docx"]  (source IDs, not paths)
```

### plan() / apply()

`run()` is `apply(plan(...))`. Call them separately to review changes before they reach the
store — `plan()` returns a serializable `ChangeSet` and does not touch the store:

```python
changeset = await vs_pipeline.plan(paths)
changeset.save(Path("review/changes.json"))
# ... inspect / edit the JSON ...
approved = ChangeSet[Chunk].load(Path("review/changes.json"))
result = await vs_pipeline.apply(approved)
```

### Running against a directory

Use `run_directory` to discover files automatically:

```python
result = await vs_pipeline.run_directory(Path("docs/"), glob="**/*.docx")
```

### Error handling

Per-file exceptions are collected in `UpdateResult.errors` rather than propagated:

```python
result = await vs_pipeline.run(paths)
for path, exc in result.errors:
    print(f"FAILED {path}: {exc}")
```

---

## Scenario C — Qdrant vector store

[`QdrantVectorStore`](../api/integrations.md#QdrantVectorStore) is the built-in
concrete implementation of `VectorStore`, backed by `qdrant-client`'s
`AsyncQdrantClient`.

Install the extra first:

```bash
uv add "ragdoc[qdrant]"
```

### Create a store and run incremental sync

```python
from pathlib import Path
from qdrant_client import AsyncQdrantClient
from ragdoc.integrations.vector_stores import QdrantVectorStore
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline, TokenSplitter

client = AsyncQdrantClient("http://localhost:6333")

# create() creates the collection (and a source_id payload index) if it doesn't exist.
# Safe to call on every startup — it is idempotent.
store = await QdrantVectorStore.create(client, "my_docs", vector_size=1536)

pipeline = DocumentPipeline(splitter=TokenSplitter(max_tokens=4000))
vs_pipeline = VectorStorePipeline(pipeline=pipeline, vector_store=store)

result = await vs_pipeline.run(list(Path("docs/").glob("**/*.docx")))
print(f"processed: {len(result.processed)}, skipped: {len(result.skipped)}")
```

### Payload schema

Each Qdrant point stores the full `Chunk` as payload.  The provenance fields
(`source_id`, `source_hash`) live at the top level — they are **not** injected
into the user-facing `metadata` dict:

| Payload key | Chunk field | Purpose |
|---|---|---|
| `source_id` | `chunk.source_id` | Groups chunks by source for cleanup |
| `source_hash` | `chunk.source_hash` | Detects whether the source has changed |
| `source_path` | `chunk.source_path` | Full path to the source file |
| `prompt_content` | `chunk.prompt_content` | Full-fidelity text for LLM context |
| `embedding_content` | `chunk.embedding_content` | Text used to generate the embedding |
| `metadata` | `chunk.metadata` | User-defined key/value data |
| `created_at` | `chunk.created_at` | ISO 8601 creation timestamp |

### Performance notes

- `create()` adds a Qdrant payload index on `source_id`, turning filter operations
  in `list_source_state` and `delete_by_source` from O(n) scans into O(log n) lookups.
- `upsert()` batches points in groups of 100 to avoid gRPC message size limits.
- `list_source_ids()` uses scroll pagination (1000 points per page) so it works
  correctly at any collection size.

For a full production example including processors and `LLMChunker`, see the
[Qdrant Pipeline notebook](../notebooks/qdrant_pipeline/).

---

### Implementing a custom VectorStore

Any object implementing the `VectorStore` protocol satisfies the requirements.
The protocol requires five async methods — implementations should index on
`chunk.source_id` and `chunk.source_hash` (first-class fields, not
`chunk.metadata`):

```python
class MyVectorStore:
    async def upsert(self, chunks: list[Chunk]) -> list[str]:
        ids = [await self._db.insert(c) for c in chunks]
        return ids

    async def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            await self._db.remove(id_)

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks whose source_id field equals source_id."""
        await self._db.delete(source_id=source_id)

    async def list_source_ids(self) -> set[str]:
        """Return all distinct source_id values stored in this vector store."""
        return await self._db.distinct_source_ids()

    async def list_source_state(self) -> dict[str, SourceState]:
        """Bulk-read each source's hashes in one round-trip (used by plan())."""
        return {
            row.source_id: SourceState(row.source_hash, row.content_hash)
            for row in await self._db.distinct_sources()
        }
```

### Provenance fields

Each chunk upserted by `VectorStorePipeline` has two first-class fields set
before upsert — these are **not** injected into `chunk.metadata`:

| Field | Value | Purpose |
|-------|-------|---------|
| `chunk.source_id` | `source_id_fn(path)` (default: `path.name`) | Groups chunks by source for cleanup |
| `chunk.source_hash` | SHA-256 hex digest of file bytes | Detects whether the source has changed |

To customise the source identity key, set `source_id_fn` on the **`DocumentPipeline`** (the
single source of truth — the `VectorStorePipeline` reads it from there for collision/orphan
checks):

```python
doc_pipeline = DocumentPipeline(
    splitter=TokenSplitter(),
    source_id_fn=lambda p: str(p.relative_to(base_dir)),  # portable multi-dir
)
vs_pipeline = VectorStorePipeline(pipeline=doc_pipeline, vector_store=my_vector_store)
```

A third field, `chunk.content_hash` (`Document.content_hash()`), is also set — it drives
re-chunking when a stored Document is edited in the two-stage `DocumentStore` workflow
(`DocumentStorePipeline` → editing → `VectorStorePipeline.from_document_store(...)`).

---

## Scenario E — two-stage with a `DocumentStore`

Parse once, chunk many ways. `DocumentStorePipeline` syncs parsed+processed `Document`s into a
`DocumentStore` (Boundary 1); you can then edit them and re-chunk into the vector store
(Boundary 2) without re-parsing — which matters when parsing is expensive (OCR, LLM heading
resolution).

| Implementation | Backing | Use case | Extra |
|---|---|---|---|
| `LocalDocumentStore` | one JSON file per source on disk | local dev, single machine, hand-editing | — |
| `QdrantDocumentStore` | one Qdrant point per source | shared across workers/containers | `qdrant` |

Each stage takes the boundary-appropriate pipeline type, so a misplaced stage is a
`TypeError` at construction: `DocumentStorePipeline` takes an `IngestPipeline` (no
splitter/chunker parameter exists), and the Boundary-2 vector sync is a separate
constructor — `VectorStorePipeline.from_document_store` — taking a `ChunkPipeline`
(no parser/processors parameter exists).

```python
from ragdoc.pipeline import ChunkPipeline, DocumentStorePipeline, IngestPipeline, VectorStorePipeline
from ragdoc.integrations.document_stores import QdrantDocumentStore

doc_store = await QdrantDocumentStore.create(client, "my_documents")

# Stage 1: parse + process → DocumentStore (no chunking)
await DocumentStorePipeline(
    ingest=IngestPipeline(processors=[...]),
    document_store=doc_store,
).run(paths)

# ... edit stored Documents (any worker) ...

# Stage 2: chunk + embed only the documents whose content_hash changed
vs = VectorStorePipeline.from_document_store(
    chunk=ChunkPipeline(splitter=..., chunker=...),
    vector_store=store,
    document_store=doc_store,
)
await vs.run()   # sources=None → all documents in the store
```

`QdrantDocumentStore` stores each Document as a single point with a **throwaway 1-dim vector**
(Documents aren't embedded) — the collection is a key-value store keyed on `source_id`, not
semantically searchable. A Document whose payload exceeds Qdrant's size limit raises
`DocumentTooLargeError` (carrying the `source_id` and byte size) rather than silently
splitting or dropping image data.

---

## See Also

- [API Reference: Pipeline](../api/pipeline.md)
- [Chunking Guide](chunking.md) — `SimpleChunker`, `LLMChunker`
- [Splitting Guide](splitting.md) — lower-level splitting API
