# Splitting

> Run interactively: `marimo edit docs/notebooks/splitting.py`

Splitting divides a single [`Document`](document-model.md) into multiple smaller documents,
each representing a logical section. Splitting is the fourth stage of the pipeline, between
processing and chunking.

## Why split?

LLM context windows and vector store chunks have size limits. Splitting breaks a large
document into sections that are:

- Small enough to fit in a context window
- Semantically coherent (bounded by headings)
- Still structured `Document` objects (not plain text)

Each resulting document carries [`ExternalRef`](../api/document.md#ExternalRef) entries
that record the parent-child relationship back to the source document.

## Split functions

ragdoc provides two splitting strategies in [`ragdoc.splitting`](../api/splitting.md):

### `split_by_headings`

Splits at every heading boundary, producing a flat list of documents:

```python
from ragdoc.splitting import split_by_headings

sections = split_by_headings(document)
# Each section: one heading + following content, until next heading
# Result: flat list[Document]
```

Use this when you want uniform granularity — one section per heading regardless of depth.

### `split_hierarchical`

Splits recursively by the lowest heading level that produces at least two parts:

```python
from ragdoc.splitting import split_hierarchical

sections = split_hierarchical(document)
# Result: list[Document] where each document may contain sub-headings
```

Use this when you want larger, more self-contained sections that preserve sub-structure.

## Parent-child relationships

After splitting, each child document has an `external_refs` entry pointing to its parent,
and the parent has entries pointing to all children:

```python
sections = split_by_headings(document)

for section in sections:
    parent_refs = [r for r in section.external_refs if r.rel_type == "external-parent"]
    print(f"Section '{section.title}' has parent: {parent_refs}")
```

These relationships can be used by the renderer's
[`ExternalRefProvider`](rendering.md#cross-document-references) to include
context from sibling or parent sections in the rendered output.

## Post-split processing

Because split results are still `Document` objects, you can run processors on them after
splitting. For example, enrich each section independently with any `DocumentProcessor`:

```python
from ragdoc.splitting import split_by_headings
from ragdoc.processing import ImageSummaryProcessor
from ragdoc.processing.summary_image import openai_image_summarizer

sections = split_by_headings(document)

processor = ImageSummaryProcessor(summarize=openai_image_summarizer(openai_client))
sections = [await processor.process(section) for section in sections]
```

## Reading order

`split_document` writes `metadata["split_sequence"]` (1-based position) and
`metadata["split_total"]` (total split count) into every document it produces —
including documents that fit within the budget (`split_sequence=1, split_total=1`).
The list it returns is already in reading order, but these keys let you restore
that order after documents have been stored, retrieved, or reordered:

```python
splits = split_document(doc, renderer=renderer, max_tokens=5000)

# sorting is a no-op here — splits already in order — but it is safe and explicit:
ordered = sorted(splits, key=lambda d: d.metadata["split_sequence"])
```

Both keys propagate automatically to [`Chunk.metadata`](../api/chunking.md#Chunk)
via the chunker's `metadata_fn`, so no extra work is needed after chunking.

`split_total` makes each chunk self-describing: a retrieval client can determine
whether the chunk has predecessors (`split_sequence > 1`) or successors
(`split_sequence < split_total`) without fetching sibling chunks.

`split_by_headings`, `split_hierarchical`, and `split_by_elements` do *not* set
these metadata keys — they are lower-level primitives. Only `split_document` writes
them.

## See Also

- [API Reference: Splitting](../api/splitting.md)
- [Chunking Guide](chunking.md) — materialize split documents into `Chunk`
- [Document Model Guide](document-model.md#cross-document-references) — ExternalRef details
