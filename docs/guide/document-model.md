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
- **`filename`** — Name of the source file (e.g. `"report.pdf"`), set by parsers
- **`source_path`** — Full path to the source file (e.g. `"/data/report.pdf"`), set by parsers
- **`metadata`** — Arbitrary key-value metadata (dict) for user-defined custom data
- **`external_refs`** — List of [`ExternalRef`](../api/document.md#externalref) for cross-document links
- **`parser`** — Provenance string set by the parser (e.g. `"azure_di"`, `"pandoc"`)

`filename` and `source_path` are explicit first-class fields — not stored in `metadata`.
Parsers set them automatically; they propagate through splitting and into `Chunk`.

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
| [`RawText`](../api/document.md#rawtext) | Raw HTML content (pre-formatted blocks) |

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

## See Also

- [API Reference: Document](../api/document.md)
- [Parsing Guide](parsing.md) — how parsers produce Documents
- [Processing](../api/processing.md) — processors that enrich documents
