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
    # Parsing

    Parsing is the first stage of the pipeline. It converts a raw file into a structured
    `Document` object. `document.parser` records which parser produced it, so downstream
    processors can adjust behavior.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quick start

    [`load()`][ragdoc.parsing.load] is the primary entry point. It picks the parser from
    the file name. The cell below writes a small HTML file to a temp directory and loads it;
    point `sample_path` at one of your own files to try another format.
    """)
    return


@app.cell
async def _():
    import tempfile
    from pathlib import Path

    from ragdoc.parsing import load

    sample_path = Path(tempfile.mkdtemp(prefix="ragdoc-parsing-")) / "page.html"
    sample_path.write_text(
        "<html><body>"
        "<h1>Annual Report</h1>"
        "<p>Revenue grew in every region.</p>"
        "<h2>Regions</h2>"
        "<table><tr><th>Region</th><th>Growth</th></tr><tr><td>North</td><td>8%</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )

    document = await load(sample_path)
    print(f"parser={document.parser!r}  elements={len(document.elements)}")
    print(f"source_path={document.source_path!r}")
    print(f"metadata={document.metadata}")
    for _el in document.elements:
        print(f"  {type(_el).__name__:<10} {_el.text[:50]!r}")
    return (Path, document, load, sample_path)


@app.cell
def test_load_html(document, sample_path):
    from ragdoc.document import Heading, Paragraph, Table

    assert document.parser == "html"
    assert [type(e) for e in document.elements] == [Heading, Paragraph, Heading, Table]
    assert [h.level for h in document.headings] == [1, 2]
    # load() stamps provenance: the full path as a field, the bare filename in metadata.
    assert document.source_path == str(sample_path)
    assert document.metadata["filename"] == "page.html"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Available parsers

    ### Word documents (`.docx`)

    Uses [Pandoc](https://pandoc.org/) to convert Word documents to HTML, then parses the
    HTML into a `Document`.
    """)
    return


@app.cell
async def _(Path, load):
    doc_docx = None
    if Path("report.docx").exists():
        doc_docx = await load("report.docx")
        # doc_docx.parser == "pandoc"
        print(f"parser={doc_docx.parser!r}")
    else:
        print("report.docx not found — replace the path with a real .docx file to run this cell")
    return (doc_docx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### HTML files (`.html`)

    Parses HTML files directly into a `Document`, preserving heading structure, paragraphs,
    tables, and images. Pass `parser=` to force a parser by its registry name instead of
    relying on the file name:
    """)
    return


@app.cell
async def _(load, sample_path):
    doc_html = await load(sample_path, parser="html")
    print(f"parser={doc_html.parser!r}  tables={len(doc_html.tables)}")
    return (doc_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Excel files (`.xlsx`)

    Converts Excel workbooks into a `Document`. Each sheet becomes a section; tables are
    extracted as `Table` elements.
    """)
    return


@app.cell
async def _(Path, load):
    doc_xlsx = None
    if Path("data.xlsx").exists():
        doc_xlsx = await load("data.xlsx")
        # doc_xlsx.parser == "xlsx"
        print(f"parser={doc_xlsx.parser!r}")
    else:
        print("data.xlsx not found — replace the path with a real .xlsx file to run this cell")
    return (doc_xlsx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Azure Document Intelligence (`.azure.json`)

    Parses a saved JSON result from Azure Document Intelligence into a `Document`.
    """)
    return


@app.cell
async def _(Path, load):
    doc_azure = None
    if Path("result.azure.json").exists():
        doc_azure = await load("result.azure.json")
        # doc_azure.parser == "azure_di"
        print(f"parser={doc_azure.parser!r}")
    else:
        print("result.azure.json not found — replace the path with a real Azure DI output file to run this cell")
    return (doc_azure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PDF files (`.pdf`)

    Several parsers can handle PDFs. `load("report.pdf")` uses the highest-priority one that
    is *available* in the current environment — a parser whose extra or credentials are
    missing is skipped:

    | Parser | Priority | Needs |
    |---|---|---|
    | `azure_di` | 40 | `azure-di` extra + Azure Document Intelligence credentials |
    | `pdf_basic` | 10 | `pdf` extra (local, pymupdf; text + font-size headings) |

    MinerU works on its own output file rather than on the PDF: run MinerU first, then load
    the resulting `*_middle.json` (parser `mineru`, `pdf-mineru` extra).

    ```python
    doc_pdf = await load("report.pdf")                      # best available PDF parser
    doc_pdf = await load("report.pdf", parser="pdf_basic")  # force one by name
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Built-in parser registry

    `load()` resolves parsers from a registry. Built-in parsers are registered at import time:

    | File name ends with | Parser name | Priority |
    |---------|--------|----------|
    | `.html`, `.htm` | `html` | 0 |
    | `.docx` (`.doc` is registered but pandoc cannot read it) | `pandoc` | 0 |
    | `.xlsx` | `xlsx` | 0 |
    | `.azure.json` | `azure_json` | 0 |
    | `.pdf` | `azure_di` | 40 |
    | `.pdf` | `pdf_basic` | 10 |
    | `_middle.json` | `mineru` | 50 |
    | `.ragdoc.json` | `ragdoc_json` | 60 |

    When several parsers match a file name, the highest-priority available one wins.
    Use `describe_registry()` to see all registered parsers:
    """)
    return


@app.cell
def _():
    from ragdoc.parsing import describe_registry

    registry_text = describe_registry()
    print(registry_text)
    return (registry_text,)


@app.cell
def test_registry_lists_builtin_parsers(registry_text):
    for _name in ("html", "pandoc", "xlsx", "azure_json", "azure_di", "pdf_basic", "mineru", "ragdoc_json"):
        assert _name in registry_text, f"{_name} missing from the parser registry"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What parsers produce

    All parsers produce a `Document` where:

    - Elements store content as `html` (full HTML including the outer tag)
    - Visual properties (font size, weight, alignment) are stored as **inline CSS** on the HTML tags
    - Inline references (`<ref id="..." rel="..."/>`) are embedded in HTML for images and footnotes

    Provenance is stamped centrally by `load()` (and by the pipelines when you pass a custom
    parser), not by each parser:

    - `document.parser` — the name of the parser that produced the document
    - `document.source_path` — the path that was loaded, as a string
    - `document.metadata["filename"]` — the bare filename

    The CSS-in-HTML convention is what allows processors like
    [`HeadingLevelProcessor`][ragdoc.processing.HeadingLevelProcessor] to work uniformly
    across all parsers without parser-specific logic.

    ## See Also

    - [API Reference: Parsing][ragdoc.parsing]
    - [Document Model notebook](document_model.py)
    - [Processing API][ragdoc.processing] — refine heading levels, detect titles, summarize
    """)
    return


if __name__ == "__main__":
    app.run()
