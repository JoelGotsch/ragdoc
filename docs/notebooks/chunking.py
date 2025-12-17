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
    # Chunking

    Chunking is the final stage of the pipeline. It materializes a `Document` (or split
    section) into a [`Chunk`][ragdoc.chunking.Chunk] — a flat, self-contained record ready
    to be loaded into a vector store.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is a Chunk?

    A [`Chunk`][ragdoc.chunking.Chunk] is a Pydantic model with:

    | Field | Description |
    |-------|-------------|
    | `id` | Unique identifier (UUID string) |
    | `source_id` | Sync identity key of the source document (required) |
    | `source_hash` | SHA-256 of the source file bytes (required; chunkers fall back to `Document.content_hash()`) |
    | `prompt_content` | Full-fidelity text for LLM context windows and BM25 search |
    | `embedding_content` | Compact semantic text for dense vector search |
    | `metadata` | Key-value metadata from the source document |
    | `embedding` | Optional pre-computed embedding vector (list[float]) |
    | `created_at` | Timestamp of creation |

    The two content fields come from the two rendering paths.
    """)
    return


@app.cell
def _():
    # Build a sample document to demonstrate chunking
    from ragdoc.document import Document, Heading, Paragraph, Table

    sample_doc = Document(
        title="Q3 Report",
        elements=[
            Heading(innerhtml="Q3 Report", level=1),
            Paragraph(html_content="<p>Revenue increased by 12% year-over-year.</p>"),
            Heading(innerhtml="Financial Summary", level=2),
            Table(html_content=(
                "<table>"
                "<tr><th>Metric</th><th>Q3 2023</th><th>Q3 2024</th></tr>"
                "<tr><td>Revenue</td><td>$10M</td><td>$11.2M</td></tr>"
                "<tr><td>EBITDA</td><td>$2M</td><td>$2.5M</td></tr>"
                "</table>"
            )),
        ],
        metadata={"source": "q3_report.docx", "department": "Finance"},
    )

    print(f"Sample document: {len(sample_doc.elements)} elements")
    return (Document, Heading, Paragraph, Table, sample_doc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating chunks

    Combine [`Renderer`][ragdoc.rendering.Renderer] with `Chunk`. `SimpleChunker` sets
    `embedding_content = prompt_content` — a single renderer is used for both fields.
    """)
    return


@app.cell
def _(sample_doc):
    from ragdoc.chunking import Chunk
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
    from ragdoc.splitting import split_by_headings

    sections = split_by_headings(sample_doc)

    renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        include_title=True,
        metadata_keys=["source", "department"],
    )

    # SimpleChunker sets embedding_content = prompt_content
    chunks = [
        Chunk(
            prompt_content=renderer.render(section),
            embedding_content=renderer.render(section),
            # source_id / source_hash are required: DocumentPipeline stamps them from the
            # source path; when building chunks by hand, fall back to document identity.
            source_id="q3_report.docx",
            source_hash=section.content_hash(),
            metadata=section.metadata,
        )
        for section in sections
    ]

    print(f"Created {len(chunks)} chunks")
    return (
        Chunk, OutputFormat, Renderer, chunks,
        renderer, render_for_prompt,
        sections, split_by_headings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prompt vs embedding content

    With `SimpleChunker`, both fields contain the same rendered text. Only `LLMChunker`
    produces distinct `embedding_content` per topic.
    """)
    return


@app.cell
def _(chunks):
    # With SimpleChunker, both fields are the same
    sample_chunk = chunks[0]

    print("PROMPT CONTENT:")
    print(sample_chunk.prompt_content)
    print()
    print("EMBEDDING CONTENT (same as prompt_content with SimpleChunker):")
    print(sample_chunk.embedding_content)
    return (sample_chunk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metadata inheritance

    When you split a document, child sections inherit parent metadata. Set metadata on the
    source document before parsing and it will flow through to all chunks.
    """)
    return


@app.cell
def _(chunks):
    for chunk in chunks:
        print(f"chunk {chunk.id[:8]}… metadata={chunk.metadata}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading into a vector store

    `Chunk` is designed to map directly to vector store document schemas. Use
    `VectorStorePipeline` to embed and upsert chunks with incremental change detection.
    """)
    return


@app.cell
def _(chunks):
    print(f"Chunks ready for upsert: {len(chunks)}")
    print(f"First chunk id: {chunks[0].id}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [API Reference: Chunking][ragdoc.chunking]
    - [API Reference: Integrations][ragdoc.integrations] — vector store integrations
    - [Rendering notebook](rendering.py) — understanding the two rendering paths
    - [Splitting notebook](splitting.py) — preparing documents before chunking
    """)
    return


if __name__ == "__main__":
    app.run()
