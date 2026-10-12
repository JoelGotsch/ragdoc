# Custom Metadata

ragdoc's metadata system lets you declare your own document fields as a typed
`TypedDict`, propagate them through the pipeline with full static-type checking,
and use them for filtered retrieval in vector stores.

---

## 1. Declare a TypedDict

Subclass [`BaseMetadata`](../api/metadata.md#basemetadata) and list your fields:

```python
from typing import Annotated
from typing_extensions import Required
import ragdoc
from ragdoc.integrations.vector_stores import QdrantIndex
from qdrant_client.http import models

class MyMetadata(ragdoc.BaseMetadata, total=False):
    # Required — a processor MUST set this; the pipeline raises otherwise.
    # The QdrantIndex annotation creates a payload index at collection setup (section 5).
    document_name: Annotated[Required[str], QdrantIndex(models.PayloadSchemaType.KEYWORD)]
    # Optional — may be absent.
    document_date: Annotated[str | None, QdrantIndex(models.PayloadSchemaType.DATETIME)]
    # Optional, not indexed — a plain annotation is enough.
    department: str | None
```

The `Annotated[..., QdrantIndex(...)]` wrapper is only needed for fields you want
indexed in Qdrant; without the `qdrant` extra, declare plain types
(`document_name: Required[str]`).

`BaseMetadata` already declares `filename: Required[str]` — it is stamped
automatically when a file is loaded (`load()` / the pipelines), so you never need to
add it yourself.

Use `Required[T]` (from `typing_extensions`) to mark fields that a processor must
populate.  When you pass the type as `metadata_type` (section 3), the pipeline checks
the required keys at runtime — after all processors have run, before splitting — and
raises `ValueError` naming any that are missing.

---

## 2. Write a metadata processor

Processors receive a `Document` and write to `document.metadata`.  Use the same
key names you declared in your TypedDict:

```python
from ragdoc.processing import DocumentProcessor
from ragdoc.document import Document

class DocumentNameProcessor(DocumentProcessor):
    """Copies the detected document title into metadata["document_name"]."""

    async def process(self, document: Document) -> Document:
        # document.title is set by TitleDetectionProcessor (or LLMHeadingResolver),
        # so this processor must run after one of them.
        if document.title:
            document.metadata["document_name"] = document.title
        return document
```

Processors must not drop or overwrite keys set by earlier processors.

---

## 3. Build a typed pipeline

Pass your `TypedDict` as `metadata_type` to
[`DocumentPipeline`](../api/pipeline.md#documentpipeline):

```python
from ragdoc.chunking import SimpleChunker
from ragdoc.pipeline import DocumentPipeline, TokenSplitter
from ragdoc.processing import TitleDetectionProcessor

pipeline: DocumentPipeline[MyMetadata] = DocumentPipeline(
    processors=[TitleDetectionProcessor(), DocumentNameProcessor()],
    splitter=TokenSplitter(),
    chunker=SimpleChunker(),
    metadata_type=MyMetadata,
)
```

The type parameter propagates: `pipeline.run(path)` returns
`list[Chunk[MyMetadata]]`, so `chunk.metadata` is typed as `MyMetadata` in
your IDE.

---

## 4. Read typed metadata from chunks

After running the pipeline you can access metadata with full type support:

```python
from pathlib import Path

chunks = await pipeline.run(Path("report.docx"))
for chunk in chunks:
    name: str = chunk.metadata["document_name"]   # typed — IDE knows this is str
    date: str | None = chunk.metadata.get("document_date")
```

---

## 5. Register Qdrant payload indexes

When using [`QdrantVectorStore`](../api/integrations.md#qdrantvectorstore), pass
`metadata_type` to `create` so indexes are registered automatically:

```python
from ragdoc.integrations.vector_stores import QdrantVectorStore

store = await QdrantVectorStore.create(
    client,
    "my_collection",
    vector_size=1536,
    metadata_type=MyMetadata,   # registers payload indexes for QdrantIndex fields
)
```

This calls [`register_indexes_from_type`](../api/integrations.md#register_indexes_from_type)
internally.  Against a Qdrant server, `409 Conflict` responses (index already
exists) are ignored, so it is safe to call on every application start.

You can also call `register_indexes_from_type` directly — useful when the
collection already exists:

```python
from ragdoc.integrations.vector_stores import register_indexes_from_type

await register_indexes_from_type(MyMetadata, client, "my_collection")
```

---

## 6. Generate a JSON schema for LLM self-query

[`metadata_json_schema`](../api/metadata.md#metadata_json_schema) generates a
JSON Schema dict from your TypedDict, suitable for injecting into an LLM prompt
to enable self-query retrieval:

```python
import json
from ragdoc.metadata import metadata_json_schema

schema = metadata_json_schema(MyMetadata)
system_prompt = (
    "You are a document retrieval assistant. "
    "Filter results using these metadata fields:\n"
    + json.dumps(schema, indent=2)
)
```

`Annotated` extras that Pydantic does not recognise (such as `QdrantIndex`) are
automatically stripped — the schema contains only field names and types.

---

## Antipatterns

**Do not inject metadata inside a processor using keys not declared in your TypedDict.**
The TypedDict is the contract; undeclared keys are invisible to the type checker
and will not be indexed in Qdrant.

**Do not put provenance in `chunk.metadata`.**
`source_id`, `source_hash` and `content_hash` are first-class `Chunk` fields
stamped by the pipelines (`source_id` at ingest, `source_hash` by the sync
pipelines, `content_hash` at chunking); the sync pipelines read those fields, never
`metadata`.  Copies in `metadata` are duplication that change detection ignores.

**Do not use `total=True` on your subclass.**
All fields in `BaseMetadata` subclasses should be optional by default
(`total=False`).  Mark individual required fields with `Required[T]` instead —
this lets you mix required and optional fields precisely.
