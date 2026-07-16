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
    # Document Merging

    When the same document is parsed by two different parsers (e.g. MinerU for layout and
    Azure DI for text accuracy), the outputs typically complement each other. The merging API
    combines both into a single, higher-quality `Document`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two approaches

    | | Approach A | Approach B |
    |---|---|---|
    | Function | `merge_documents` / `compute_patch` | `merge_documents_html` / `compute_html_patch` |
    | Mechanism | Aligns structured elements directly | Render → HTML diff → reparse |
    | Inspectable intermediate | `DocumentPatch` (typed operations) | `DocumentHtmlPatch` (HTML strings) |
    | Page / bounding-box metadata | preserved | lost (architectural) |

    Use **Approach A** when you need to inspect or override individual merge decisions, or
    when page numbers / bounding boxes must be retained.

    Use **Approach B** when you want a simple drop-in merge with no extra configuration.
    """)
    return


@app.cell
def _():
    # Build two sample documents representing the same content from different parsers
    from ragdoc.document import Document, Heading, Paragraph, Table

    doc_a = Document(
        elements=[
            Heading(html="<h1>Introduction</h1>"),
            Paragraph(html="<p>Revenue <em>grew</em> significantly.</p>"),
            Table(html="<table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2024</td><td>$11M</td></tr></table>"),
        ],
        metadata={"source": "report_mineru"},
        parser="mineru",
    )

    doc_b = Document(
        elements=[
            Heading(html="<h1>Introduction</h1>"),
            Paragraph(html="<p>Revenue grew <strong>significantly</strong> in Q3.</p>"),
            Table(html="<table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2024</td><td>$11.2M</td></tr></table>"),
        ],
        metadata={"source": "report_azure_di"},
        parser="azure_di",
    )

    print(f"doc_a: parser={doc_a.parser!r}  elements={len(doc_a.elements)}")
    print(f"doc_b: parser={doc_b.parser!r}  elements={len(doc_b.elements)}")
    return (Document, Heading, Paragraph, Table, doc_a, doc_b)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quick start
    """)
    return


@app.cell
def _(doc_a, doc_b):
    from ragdoc.merging import merge_documents, merge_documents_html

    # Approach A (element-alignment)
    merged_a = merge_documents(doc_a, doc_b)

    # Approach B (render-merge-reparse)
    merged_b = merge_documents_html(doc_a, doc_b)

    print(f"Approach A: parser={merged_a.parser!r}  metadata={merged_a.metadata}")
    print(f"Approach B: parser={merged_b.parser!r}  metadata={merged_b.metadata}")
    return (merge_documents, merge_documents_html, merged_a, merged_b)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Controlling what gets merged

    Both approaches accept the same two control parameters.

    ### `prefer_source`

    Force specific element types to always come from one document:
    """)
    return


@app.cell
def _(doc_a, doc_b, merge_documents_html):
    from ragdoc.document import ElementTypeEnum

    merged_prefer = merge_documents_html(
        doc_a, doc_b,
        prefer_source={
            ElementTypeEnum.HEADING: "b",   # always take headings from doc_b
            ElementTypeEnum.TABLE: "a",     # always take tables from doc_a
        },
    )
    print(f"Merged with prefer_source: {len(merged_prefer.elements)} elements")
    for el in merged_prefer.elements:
        print(f"  {type(el).__name__}: {el.text[:60]}")
    return (ElementTypeEnum, merged_prefer)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `allow_insertions_from_b`

    Control whether elements that appear only in `doc_b` are included:
    """)
    return


@app.cell
def _(doc_a, doc_b, merge_documents_html):
    # Include everything from doc_b (default)
    merged_all = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=True)

    # Suppress all doc_b-only elements
    merged_no_b = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=False)

    print(f"allow_insertions_from_b=True:  {len(merged_all.elements)} elements")
    print(f"allow_insertions_from_b=False: {len(merged_no_b.elements)} elements")
    return (merged_all, merged_no_b)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Approach A — Element-alignment

    `compute_patch()` returns a `DocumentPatch` whose operations you can inspect and override
    before applying:
    """)
    return


@app.cell
def _(doc_a, doc_b):
    from ragdoc.merging import PatchOperationType, compute_patch

    patch_a = compute_patch(doc_a, doc_b)

    print(f"source_parser_a={patch_a.source_parser_a!r}")
    print(f"source_parser_b={patch_a.source_parser_b!r}")
    print(f"\n{len(patch_a.operations)} patch operations:")
    for _op in patch_a.operations:
        print(f"  {_op.op!r:20}  {_op.reason!r}")
    return (PatchOperationType, compute_patch, patch_a)


@app.cell
def _(patch_a):
    # Optionally override a decision before applying
    # patch_a.operations[2].resolved_elements = [Paragraph(html="<p>Custom</p>")]

    merged_from_patch = patch_a.apply()
    print(f"Applied patch: {len(merged_from_patch.elements)} elements, parser={merged_from_patch.parser!r}")

    # DocumentPatch is JSON-serializable
    json_str = patch_a.model_dump_json()
    print(f"Patch JSON length: {len(json_str)} chars")
    return (json_str, merged_from_patch)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Approach B — Render-merge-reparse

    `compute_html_patch()` returns a `DocumentHtmlPatch` with HTML-string operations you can
    inspect and override:
    """)
    return


@app.cell
def _(doc_a, doc_b):
    from ragdoc.merging import compute_html_patch

    patch_b = compute_html_patch(doc_a, doc_b)

    print(f"{len(patch_b.operations)} HTML patch operations:")
    for _op in patch_b.operations:
        _selected_preview = str(_op.selected)[:60]
        print(f"  {_op.opcode!r:10}  {_selected_preview}")
    return (compute_html_patch, patch_b)


@app.cell
def _(patch_b):
    # Override before applying:
    # patch_b.operations[0].selected = ["<h1>Custom Title</h1>"]

    merged_html = patch_b.apply()
    print(f"Applied HTML patch: {len(merged_html.elements)} elements")

    # merge_documents_html is equivalent to compute_html_patch(...).apply()
    json_b = patch_b.model_dump_json()
    print(f"HTML patch JSON length: {len(json_b)} chars")
    return (json_b, merged_html)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validating the merged document

    `validate_inline_refs` returns a list of target IDs for which no matching element exists
    in the document:
    """)
    return


@app.cell
def _(merged_a):
    from ragdoc.merging import validate_inline_refs

    broken = validate_inline_refs(merged_a)
    if broken:
        print("Broken inline refs:", broken)
    else:
        print("All inline refs resolved.")
    return (validate_inline_refs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [API Reference: Merging][ragdoc.merging]
    - [Parsing notebook](parsing.py) — available parsers
    - [Document Model notebook](document_model.py) — element types and inline refs
    """)
    return


if __name__ == "__main__":
    app.run()
