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
    # Rendering

    Rendering converts a `Document` into a formatted string. ragdoc provides two element
    renderers and multiple output formats.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The two element renderers

    | Renderer function | Intent |
    |-------------------|--------|
    | [`render_for_prompt`][ragdoc.rendering.render_for_prompt] | Full-fidelity text for LLM context windows and BM25 search |
    | [`render_raw`][ragdoc.rendering.render_raw] | Full fidelity, no transformation — for debugging |

    `render_for_prompt` is the primary renderer used throughout the pipeline. For `Image`
    elements, it uses `text_representation` as a text fallback when the image cannot be
    displayed directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using the Renderer

    [`Renderer`][ragdoc.rendering.Renderer] combines an element renderer with a target
    output format. The cells below create a small sample document to demonstrate rendering.
    """)
    return


@app.cell
def _():
    from ragdoc.document import Document, Heading, Image, Paragraph, Table
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

    # Build a minimal sample document
    sample_doc = Document(
        title="Sample Document",
        elements=[
            Heading(html="<h1>Introduction</h1>"),
            Paragraph(html="<p>This document explains the rendering system.</p>"),
            Table(html="<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"),
        ],
        metadata={"source": "sample.docx", "date": "2024-01-01"},
    )

    # For LLM context windows — full content as Markdown
    prompt_renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
    )

    prompt_text = prompt_renderer.render(sample_doc)

    print("--- PROMPT ---")
    print(prompt_text)
    return (
        Document, Heading, Image, OutputFormat, Paragraph, Renderer, Table,
        prompt_renderer, prompt_text,
        render_for_prompt, sample_doc,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Output formats

    [`OutputFormat`][ragdoc.rendering.OutputFormat] controls the final text format.
    Element renderers always output HTML internally; the `Renderer` handles conversion:

    | Format | Description |
    |--------|-------------|
    | `OutputFormat.HTML` | Raw HTML output (no conversion) |
    | `OutputFormat.MARKDOWN` | Standard Markdown |
    | `OutputFormat.GFM` | GitHub Flavored Markdown (tables rendered) |
    | `OutputFormat.RST` | reStructuredText |
    | `OutputFormat.PLAIN` | Plain text (all markup stripped) |

    Format conversion is done via [pypandoc](https://github.com/NicklasTegner/pypandoc).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metadata in output

    Use `metadata_keys` to include selected metadata fields as a header in the rendered output:
    """)
    return


@app.cell
def _(OutputFormat, Renderer, render_for_prompt, sample_doc):
    renderer_with_meta = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        metadata_keys=["source", "date"],
        include_title=True,
    )
    text_with_meta = renderer_with_meta.render(sample_doc)
    print(text_with_meta)
    return (renderer_with_meta, text_with_meta)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image text_representation

    When `ImageSummaryProcessor` sets `image.text_representation`, `render_for_prompt` uses
    it as a text fallback when the image cannot be displayed directly.
    """)
    return


@app.cell
def _(Document, Heading, Image, prompt_renderer):
    # Demonstrate text_representation on an Image element
    img = Image(alt="Chart")
    img.text_representation = "A simple bar chart showing quarterly revenue."

    img_doc = Document(elements=[
        Heading(html="<h1>Report</h1>"),
        img,
    ])

    print("--- PROMPT (uses image.text_representation as fallback) ---")
    print(prompt_renderer.render(img_doc))
    return (img, img_doc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-document references

    Pass an [`ExternalRefProvider`][ragdoc.rendering.ExternalRefProvider] to access parent
    documents and siblings during rendering. A plain `dict[str, Document]` works:
    """)
    return


@app.cell
def _(Document, Heading, OutputFormat, Paragraph, Renderer, render_for_prompt):
    all_documents = [
        Document(title="Parent", elements=[Heading(html="<h1>Parent</h1>")]),
        Document(title="Child", elements=[Paragraph(html="<p>Child content.</p>")]),
    ]
    all_docs = {doc.id: doc for doc in all_documents}

    renderer_with_refs = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        external_refs=all_docs,
    )
    print("Renderer with external ref provider created.")
    return (all_docs, all_documents, renderer_with_refs)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extending with custom element renderers

    Element renderers are `singledispatch` functions. Register a custom renderer for any
    element type — see the [Custom Elements notebook](custom_elements.py) for a full example.
    """)
    return


@app.cell
def _(Image, render_for_prompt):
    from ragdoc.rendering.base import RenderContext

    @render_for_prompt.register(Image)
    def render_image_custom(element: Image, ctx: RenderContext, inline: bool = False) -> str:
        alt = element.alt or ""
        return f"<figure><img alt='{alt}'/></figure>"

    print("Custom Image renderer registered for render_for_prompt.")
    return (RenderContext, render_image_custom)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [API Reference: Rendering][ragdoc.rendering]
    - [Chunking notebook](chunking.py) — how renderers are used to produce `Chunk`
    - [Processing API][ragdoc.processing] — LLM processors that enrich documents
    """)
    return


if __name__ == "__main__":
    app.run()
