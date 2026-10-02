# Document Model

> Run interactively: `marimo edit docs/notebooks/document_model.py`

The [`Document`](../api/document.md#document) is the central data structure in ragdoc.
Every pipeline stage — parsing, processing, splitting, and chunking — operates on `Document`
objects. It is the single source of truth throughout the pipeline.

## Structure

A `Document` is a Pydantic model containing:

- **`id`** — Unique identifier (UUID string)
- **`title`** — Optional document title
- **`elements`** — Ordered list of [`BaseElement`](../api/document.md#baseelement) subclasses
- **`source_path`** — Full path to the source file (e.g. `"/data/report.pdf"`)
- **`source_id`** / **`source_hash`** — Sync identity and file-byte hash, stamped by the pipelines (`None` on a hand-built document)
- **`metadata`** — Key-value metadata (dict): your own custom data plus a few library-written
  informational keys such as `filename` (the bare source filename, e.g. `"report.pdf"`)
- **`external_refs`** — List of [`ExternalRef`](../api/document.md#externalref) for cross-document links
- **`parser`** — Name of the parser that produced the document (e.g. `"azure_di"`, `"pandoc"`)

`source_path` is a first-class field; the bare filename lives in `metadata["filename"]`.
Both, and `parser`, are stamped by [`load()`](../api/parsing.md#load) (and by the pipelines
when you pass a custom parser); they propagate through splitting and into `Chunk`.

```python
from ragdoc.document import Document, Heading, Paragraph

doc = Document(
    title="Annual Report",
    source_path="/data/annual_report.docx",
    elements=[
        Heading(html="<h1>Introduction</h1>"),
        Paragraph(html="<p>This report covers...</p>"),
    ],
    metadata={"year": 2024, "filename": "annual_report.docx"},
)
```

## Element types

All elements inherit from [`BaseElement`](../api/document.md#baseelement):

| Type | Description |
|------|-------------|
| [`Heading`](../api/document.md#heading) | Section heading; stored `html` carries the outer `<h1>`–`<h6>` tag (`level` is derived) |
| [`Paragraph`](../api/document.md#paragraph) | Body text with HTML content |
| [`Table`](../api/document.md#table) | Table with HTML content (`<table>...</table>`) |
| [`Image`](../api/document.md#image) | Image with base64 data, alt text, and optional `text_representation` |
| [`DocumentList`](../api/document.md#documentlist) | Ordered or unordered list |
| [`Footnote`](../api/document.md#footnote) | Footnote definition with number and text |
| [`RawText`](../api/document.md#rawtext) | Raw HTML content (pre-formatted blocks), always inside one outer `<div>` |

A few rules hold for every element:

- **Unknown keywords raise.** Elements and `Document` are strict (`extra="forbid"`): a
  misspelled or outdated constructor keyword — or an unknown key in JSON passed to
  `model_validate` — is a `ValidationError`, never silently dropped. The pre-refactor
  spellings `html_content=` and `RawText(innerhtml=…)` are not aliases; the error names the
  `html` field. Custom element subclasses inherit this, so declare every field you pass.
- **`RawText` only guarantees its wrapper.** A value that is exactly one root `<div>…</div>`
  is stored unchanged; anything else (plain text, several sibling tags, a single `<table>` or
  `<pre>`) is wrapped verbatim in a `<div>`. The content is never unwrapped or re-serialized,
  and `RawText.innerhtml` returns it byte for byte. It is still HTML, so escape literal text
  yourself: `RawText(html=f"<pre><code>{html.escape(code)}</code></pre>")`.
- **Equality is by value.** Two elements are equal when they have the same concrete type and
  equal field values. Internal caches (the parsed HTML behind `.text` / `.inline_refs`) never
  affect equality, so an element equals its own `model_copy()` and its JSON round-trip.

### Image `text_representation`

`Image` elements have an optional `text_representation` field. This is populated by
`ImageSummaryProcessor` and is used by `render_for_prompt` as a text fallback when
the image cannot be displayed. The `ImageSummary` model also includes a `decorative`
boolean field that allows the processor to flag worthless images (e.g. decorative
borders) so they can be skipped during rendering.

```python
# ImageSummaryProcessor writes here:
image_element.text_representation = "Bar chart showing revenue growth from 2020 to 2024."

# render_for_prompt uses text_representation as a text fallback for images
```

## Inline references

Within a document, elements can reference each other using `<ref id="..." rel="..."/>` tags
embedded in HTML content. This is how images embedded in paragraphs, footnote markers,
and inline tables are represented.

```python
# A paragraph referencing an image inline:
para = Paragraph(
    html='<p>As shown in <ref id="fig-1" rel="image"/>, revenue grew...</p>'
)
# The renderer resolves <ref> tags to the rendered content of the referenced element.
```

The [`InlineRef`](../api/document.md#inlineref) model is derived from the HTML — it is
not stored as a field. The HTML is the single source of truth.

## Cross-document references

[`ExternalRef`](../api/document.md#externalref) records relationships between documents —
typically parent/child links created by the splitter:

```python
from ragdoc.document import ExternalRef

# After splitting, child documents carry a reference to their parent:
child_doc.external_refs = [
    ExternalRef(target_id=parent_doc.id, rel_type="external-parent")
]
# And the parent carries references to all its children:
parent_doc.external_refs = [
    ExternalRef(target_id=child_doc.id, rel_type="external-child")
]
```

## Convenience properties

`Document` exposes typed property accessors for each element type:

```python
doc.headings      # list[Heading]
doc.paragraphs    # list[Paragraph]
doc.tables        # list[Table]
doc.images        # list[Image]
doc.footnotes     # list[Footnote]

doc.main_heading  # Heading | None  (the first h1)
```

## Concatenating documents

`concat_documents(first, second)` appends `second`'s elements after `first`'s and returns a
new `Document` (inputs are not mutated); `join_documents([...])` folds a list left-to-right.
There is no `|` operator on `Document`.

```python
from ragdoc.document import concat_documents, join_documents

combined = concat_documents(part_one, part_two)                          # metadata: first wins
combined = concat_documents(part_one, part_two, metadata_policy="strict")  # raise on conflicting keys
whole = join_documents([part_one, part_two, appendix])
```

When both documents have a title, an `<h1>` carrying the second title is inserted between
the two element lists. The result has a new `id` and no `source_id` / `source_hash`.
This is plain concatenation — to *align* two parses of the same source, see the
[Merging Guide](merging.md).

## See Also

- [API Reference: Document](../api/document.md)
- [Parsing Guide](parsing.md) — how parsers produce Documents
- [Processing](../api/processing.md) — processors that enrich documents
