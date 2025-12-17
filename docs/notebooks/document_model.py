import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Document Model

    `Document` is the central data structure in ragdoc. Every pipeline stage — parsing,
    processing, splitting, and chunking — operates on `Document` objects. It is the single
    source of truth throughout the pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Structure

    A `Document` is a Pydantic model containing:

    - **`id`** — Unique identifier (UUID string)
    - **`title`** — Optional document title
    - **`elements`** — Ordered list of `BaseElement` subclasses
    - **`metadata`** — Arbitrary key-value metadata (dict)
    - **`external_refs`** — List of `ExternalRef` for cross-document links
    - **`parser`** — Provenance string set by the parser (e.g. `"azure_di"`, `"pandoc"`)
    """)
    return


@app.cell
def _():
    from ragdoc.document import Document, Heading, Paragraph

    doc = Document(
        title="Annual Report",
        elements=[
            Heading(innerhtml="Introduction", level=1),
            Paragraph(html_content="<p>This report covers...</p>"),
        ],
        metadata={"source": "annual_report.docx", "year": 2024},
    )

    print(f"id={doc.id!r}")
    print(f"title={doc.title!r}")
    print(f"parser={doc.parser!r}")
    print(f"elements={len(doc.elements)}")
    print(f"metadata={doc.metadata}")
    return (Document, Heading, Paragraph, doc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Element types

    All elements inherit from [`BaseElement`][ragdoc.document.BaseElement]:

    | Type | Description |
    |------|-------------|
    | `Heading` | Section heading with `level` (1–6) and `innerhtml` |
    | `Paragraph` | Body text with HTML content |
    | `Table` | Table with HTML content (`<table>...</table>`) |
    | `Image` | Image with base64 data, alt text, and optional `text_representation` |
    | `DocumentList` | Ordered or unordered list |
    | `Footnote` | Footnote definition with number and text |
    | `RawText` | Raw HTML content (pre-formatted blocks) |
    """)
    return


@app.cell
def _():
    from ragdoc.document import DocumentList, Footnote, Image, RawText, Table

    # Construct examples of each element type
    table = Table(html_content="<table><tr><th>A</th></tr></table>")
    fn = Footnote(innerhtml="<p>See IAEA Statute Art. IV.</p>", number=1)
    raw = RawText(innerhtml="<pre>code block</pre>")

    print(f"Table element_type: {table.element_type!r}")
    print(f"Footnote number: {fn.number}")
    print(f"RawText innerhtml: {raw.innerhtml}")
    return (DocumentList, Footnote, Image, RawText, Table, fn, raw, table)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image `text_representation`

    `Image` elements have an optional `text_representation` field. This is populated by
    `ImageSummaryProcessor` and is used by `render_for_prompt` as a text fallback when the
    image cannot be displayed. The `ImageSummary` model also includes a `decorative` boolean
    field for flagging worthless images that can be skipped during rendering.
    """)
    return


@app.cell
def _(Image, Table):
    # Demonstrate text_representation field on Image
    image_el = Image(alt="Revenue chart")
    image_el.text_representation = "Bar chart showing revenue growth from 2020 to 2024."

    # render_for_prompt uses text_representation as a text fallback for images
    print(f"image.text_representation={image_el.text_representation!r}")
    return (image_el,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inline references

    Within a document, elements can reference each other using `<ref id="..." rel="..."/>`
    tags embedded in HTML content. This is how images embedded in paragraphs, footnote
    markers, and inline tables are represented.

    The `Renderer` resolves these `<ref>` tags inline when generating prompt content.
    """)
    return


@app.cell
def _(Document, Image, Paragraph):
    img = Image(alt="Revenue chart")

    doc_with_ref = Document(
        elements=[
            Paragraph(html_content=f'<p>As shown in <ref id="{img.id}" rel="image"/>, revenue grew...</p>'),
            img,
        ],
    )

    # The renderer replaces <ref> tags with the rendered content of the referenced element
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    print(renderer.render(doc_with_ref))
    return (Document, Image, OutputFormat, Paragraph, Renderer,
            doc_with_ref, img, render_for_prompt, renderer)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-document references

    [`ExternalRef`][ragdoc.document.ExternalRef] records relationships between documents —
    typically parent/child links created by the splitter:
    """)
    return


@app.cell
def _(Document, Heading, Paragraph):
    from ragdoc.document import ExternalRef

    parent_doc = Document(
        title="Full Report",
        elements=[Heading(innerhtml="Full Report", level=1)],
    )
    child_doc = Document(
        title="Introduction",
        elements=[Paragraph(html_content="<p>Introduction text.</p>")],
    )

    # After splitting, relationships are set automatically by split_by_headings / split_document
    child_doc.external_refs = [
        ExternalRef(target_id=parent_doc.id, rel_type="external-parent")
    ]
    parent_doc.external_refs = [
        ExternalRef(target_id=child_doc.id, rel_type="external-child")
    ]

    print(f"Child refs: {[r.rel_type for r in child_doc.external_refs]}")
    print(f"Parent refs: {[r.rel_type for r in parent_doc.external_refs]}")
    return (ExternalRef, child_doc, parent_doc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convenience properties

    `Document` exposes typed property accessors for each element type:
    """)
    return


@app.cell
def _(Document, Heading, Paragraph, Table):
    demo_doc = Document(
        elements=[
            Heading(innerhtml="Title", level=1),
            Paragraph(html_content="<p>Text.</p>"),
            Table(html_content="<table></table>"),
            Heading(innerhtml="Section", level=2),
        ]
    )

    print(f"headings:    {[h.level for h in demo_doc.headings]}")
    print(f"paragraphs:  {len(demo_doc.paragraphs)}")
    print(f"tables:      {len(demo_doc.tables)}")
    print(f"main_heading: {demo_doc.main_heading.text if demo_doc.main_heading else None!r}")
    return (demo_doc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [API Reference: Document][ragdoc.document]
    - [Parsing notebook](parsing.py) — how parsers produce Documents
    - [Processing API][ragdoc.processing] — processors that enrich documents
    """)
    return


if __name__ == "__main__":
    app.run()
