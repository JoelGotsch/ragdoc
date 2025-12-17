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
    # Splitting

    Splitting divides a single `Document` into multiple smaller documents, each representing
    a logical section. It is the fourth stage of the pipeline, between processing and chunking.

    ## Why split?

    LLM context windows and vector store chunks have size limits. Splitting breaks a large
    document into sections that are:

    - Small enough to fit in a context window
    - Semantically coherent (bounded by headings)
    - Still structured `Document` objects (not plain text)

    Each resulting document carries [`ExternalRef`][ragdoc.document.ExternalRef] entries
    that record the parent-child relationship back to the source document.
    """)
    return


@app.cell
def _():
    # Build a sample document with headings to demonstrate splitting
    from ragdoc.document import Document, Heading, Paragraph

    sample_doc = Document(
        title="Annual Report",
        elements=[
            Heading(innerhtml="Introduction", level=1),
            Paragraph(html_content="<p>Overview paragraph.</p>"),
            Heading(innerhtml="Section A", level=2),
            Paragraph(html_content="<p>Section A content.</p>"),
            Heading(innerhtml="Section B", level=2),
            Paragraph(html_content="<p>Section B content.</p>"),
            Heading(innerhtml="Subsection B1", level=3),
            Paragraph(html_content="<p>Subsection B1 content.</p>"),
        ],
        metadata={"source": "annual_report.docx"},
    )
    print(f"Sample document: {len(sample_doc.elements)} elements, {len(sample_doc.headings)} headings")
    return (Document, Heading, Paragraph, sample_doc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split functions

    ### `split_by_headings`

    Splits at every heading boundary, producing a flat list of documents. Use this when you
    want uniform granularity — one section per heading regardless of depth.
    """)
    return


@app.cell
def _(sample_doc):
    from ragdoc.splitting import split_by_headings

    sections = split_by_headings(sample_doc)
    print(f"split_by_headings: {len(sections)} sections")
    for _i, _sec in enumerate(sections):
        print(f"  [{_i}] title={_sec.title!r}  elements={len(_sec.elements)}")
    return (sections, split_by_headings)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `split_hierarchical`

    Splits recursively by the lowest heading level that produces at least two parts. Use this
    when you want larger, more self-contained sections that preserve sub-structure.
    """)
    return


@app.cell
def _(sample_doc):
    from ragdoc.splitting import split_hierarchical

    hierarchical_sections = split_hierarchical(sample_doc)
    print(f"split_hierarchical: {len(hierarchical_sections)} sections")
    for _i, _sec in enumerate(hierarchical_sections):
        print(f"  [{_i}] title={_sec.title!r}  elements={len(_sec.elements)}")
    return (hierarchical_sections, split_hierarchical)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parent-child relationships

    After splitting, each child document has an `external_refs` entry pointing to its parent,
    and the parent has entries pointing to all children.
    """)
    return


@app.cell
def _(sample_doc, split_by_headings):
    sections_2 = split_by_headings(sample_doc)

    for section in sections_2:
        parent_refs = [r for r in section.external_refs if r.rel_type == "external-parent"]
        child_refs = [r for r in section.external_refs if r.rel_type == "external-child"]
        print(f"Section {section.title!r}: parents={len(parent_refs)} children={len(child_refs)}")
    return (sections_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Post-split processing

    Because split results are still `Document` objects, you can run processors on them after
    splitting. For example, use `ImageSummaryProcessor` to summarize images in each section,
    or any custom `DocumentProcessor` to enrich focused sub-documents before chunking.
    """)
    return


@app.cell
def _(sample_doc, split_by_headings):
    sections_3 = split_by_headings(sample_doc)

    # Example: run a processor on each section after splitting
    # from ragdoc.processing import ImageSummaryProcessor
    # from openai import AsyncOpenAI
    # processor = ImageSummaryProcessor(client=AsyncOpenAI())
    # sections_3 = [await processor.process(s) for s in sections_3]

    print("Post-split processing: run any DocumentProcessor on each section before chunking.")
    print(f"Would process {len(sections_3)} sections.")
    return (sections_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading order after retrieval

    `split_document` writes `metadata["split_sequence"]` (1-based position) and
    `metadata["split_total"]` (total split count) into every document it produces —
    including documents that fit within the budget (`split_sequence=1, split_total=1`).
    The list is already in reading order, but these keys let you restore that order
    after documents have been stored, retrieved, or reordered:
    """)
    return


@app.cell
def _(sample_doc):
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
    from ragdoc.splitting.token import split_document

    _renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    # max_tokens=20 forces the sample document to be split into multiple sections
    splits = split_document(sample_doc, renderer=_renderer, max_tokens=20)
    print(f"split_document produced {len(splits)} split(s)")
    for _i, _d in enumerate(splits):
        print(
            f"  [{_i}] split_sequence={_d.metadata.get('split_sequence')!r}"
            f"  split_total={_d.metadata.get('split_total')!r}"
            f"  elements={len(_d.elements)}"
        )
    return (splits, split_document)


@app.cell
def _(splits):
    restored = sorted(splits, key=lambda d: d.metadata["split_sequence"])
    assert restored == splits  # already in order in this example
    return (restored,)


@app.cell
def test_sequence_assigned(splits):
    assert all("split_sequence" in d.metadata for d in splits), \
        "split_document must set split_sequence in metadata for every split"
    assert [d.metadata["split_sequence"] for d in splits] == list(range(1, len(splits) + 1)), \
        "split_sequence must be contiguous starting from 1"
    assert all(d.metadata["split_total"] == len(splits) for d in splits), \
        "split_total must equal the number of splits for every split"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [API Reference: Splitting][ragdoc.splitting]
    - [Chunking notebook](chunking.py) — materialize split documents into `Chunk`
    - [Document Model notebook](document_model.py) — ExternalRef details
    """)
    return


if __name__ == "__main__":
    app.run()
