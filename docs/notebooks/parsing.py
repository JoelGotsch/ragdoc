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
    `Document` object. Each parser sets `document.parser` to a provenance string so downstream
    processors can adjust behavior.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quick start

    [`load()`][ragdoc.parsing.load] is the primary entry point. It infers the right parser
    from the file extension. Replace the path below with a real file to run this cell.
    """)
    return


@app.cell
async def _():
    from ragdoc.parsing import load

    document = await load("report.docx")
    print(f"parser={document.parser!r}  elements={len(document.elements)}")
    return (document, load)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Available parsers

    ### Word documents (`.docx`, `.doc`)

    Uses [Pandoc](https://pandoc.org/) to convert Word documents to HTML, then parses the
    HTML into a `Document`. `WordFile` is an alias for `PandocFile`.
    """)
    return


@app.cell
async def _(load):
    doc_docx = await load("report.docx")
    # doc_docx.parser == "pandoc"
    print(f"parser={doc_docx.parser!r}")
    return (doc_docx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### HTML files (`.html`)

    Parses HTML files directly into a `Document`, preserving heading structure, paragraphs,
    tables, and images.
    """)
    return


@app.cell
async def _(load):
    doc_html = await load("page.html")
    # doc_html.parser == "html"
    print(f"parser={doc_html.parser!r}")
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
async def _(load):
    doc_xlsx = await load("data.xlsx")
    # doc_xlsx.parser == "xlsx"
    print(f"parser={doc_xlsx.parser!r}")
    return (doc_xlsx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Azure Document Intelligence (`.azure.json`)

    Parses the JSON output from Azure Document Intelligence into a `Document`. This is the
    recommended parser for complex PDFs.
    """)
    return


@app.cell
async def _(load):
    doc_azure = await load("result.azure.json")
    # doc_azure.parser == "azure_di"
    print(f"parser={doc_azure.parser!r}")
    return (doc_azure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also pass an in-memory analyze result directly, without a file on disk:
    """)
    return


@app.cell
async def _(load):
    # For PDFs, load() dispatches directly to Azure Document Intelligence:
    # doc_pdf = await load("report.pdf")

    # You can also force a specific parser by name:
    # doc = await load("result.json", parser="azure_di")

    print("load() dispatches PDFs to Azure DI; pass parser= to force a specific parser")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Built-in parser registry

    `load()` resolves parsers from a registry. Built-in parsers are registered at import time:

    | Pattern | Parser | Priority |
    |---------|--------|----------|
    | `.html`, `.htm` | HTML | — |
    | `.docx`, `.doc` | Pandoc | — |
    | `.xlsx` | Excel | — |
    | `.azure.json` | Azure DI | 40 |
    | `.pdf` | Azure DI | 40 |
    | `_middle.json` | MinerU | 50 |

    Use `describe_registry()` to see all registered parsers:
    """)
    return


@app.cell
def _():
    from ragdoc.parsing import describe_registry
    print(describe_registry())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What parsers produce

    All parsers produce a `Document` where:

    - Elements store content as `innerhtml` (HTML string)
    - Visual properties (font size, weight, alignment) are stored as **inline CSS** on the HTML tags
    - `document.parser` is set to identify the source parser
    - Inline references (`<ref id="..." rel="..."/>`) are embedded in HTML for images and footnotes
    - `document.metadata["filename"]` is set to the source filename (bare name)
    - `document.source_path` is set to the full path as a string

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
