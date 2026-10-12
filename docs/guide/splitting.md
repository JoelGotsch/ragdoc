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

Each resulting document carries [`ExternalRef`](../api/document.md#externalref) entries
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

Splits once, at the lowest heading level that occurs at least twice; parent headings are
prepended to each part as context (`split_document` applies it recursively):

```python
from ragdoc.splitting import split_hierarchical

sections = split_hierarchical(document)
# Result: list[Document] where each document may contain sub-headings
```

Use this when you want larger, more self-contained sections that preserve sub-structure.

## Token-budget splitting: `split_document`

[`split_document`](../api/splitting.md#split_document) (what
[`TokenSplitter`](../api/pipeline.md#tokensplitter) calls) combines the heading strategies
with a token budget. A document that fits is returned as one split; otherwise it is split
hierarchically by headings, and sections that still do not fit are packed element by element.

An **element that alone exceeds the budget** — a 300-row table, a very long list — is cut on
its own structural boundaries first:

| Element | Cut at | Each fragment is |
|---|---|---|
| Table | rows (including rows in `<tbody>` / `<tfoot>`) | a well-formed `Table`; caption, `<colgroup>` and header are repeated |
| List | items (`<dt>`/`<dd>` pairs stay together) | a `DocumentList`; an `<ol>` continues its numbering via `start` |
| Other container | block children | a `RawText` wrapped in the original outer tag |

Fragments keep the outer tag with its attributes and inherit `page`, `bounding_box` and
`metadata`; referenced elements (`<ref id="…"/>`) follow the fragment that contains their
reference. Every fragment is measured through the renderer and halved on unit boundaries until
it fits. Only a **single** row or item that is still too large falls through to the text
tiers — sentence splitting, then a token slice with `overlap_tokens` of overlap — which work
on rendered text and also verify the rendered result against `max_tokens`. An element with
bare text between its child tags is not cut structurally (the text would be lost); it goes to
the text tiers directly.

## Parent-child relationships

After splitting, each split carries one `external-parent` entry in `external_refs` pointing
to the source document. The source document is not modified — no `external-child` entries
are written to it; add them yourself if you need them:

```python
sections = split_by_headings(document)

for section in sections:
    parent_refs = [r for r in section.external_refs if r.rel_type == "external-parent"]
    print(f"Section '{section.title}' has parent: {parent_refs}")
```

A custom element renderer can look related documents up through the renderer's
[`ExternalRefProvider`](rendering.md#cross-document-references). The built-in renderers do
not use it, so these relationships do not change the rendered output by themselves.

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
including documents that fit within the budget (`split_sequence=1, split_total=1`). (One gap:
a document that is over budget but cannot be split any further comes back as a single split
without these keys.)
The list it returns is already in reading order, but these keys let you restore
that order after documents have been stored, retrieved, or reordered:

```python
splits = split_document(doc, renderer=renderer, max_tokens=5000)

# sorting is a no-op here — splits already in order — but it is safe and explicit:
ordered = sorted(splits, key=lambda d: d.metadata["split_sequence"])
```

Both keys propagate automatically to [`Chunk.metadata`](../api/chunking.md#chunk)
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
