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
    # Quickstart

    This notebook walks through a complete parsing → splitting → chunking pipeline on a
    small sample document. Point `FILE_PATH` at one of your own files (`.docx`, `.html`,
    `.xlsx`, `.pdf`, …) to try it on real data.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    # A small sample document so the notebook runs as-is. Replace with a path to your own file.
    FILE_PATH = Path(tempfile.mkdtemp(prefix="ragdoc-quickstart-")) / "report.html"
    FILE_PATH.write_text(
        "<html><body>"
        "<h1>Quarterly Report</h1>"
        "<p>Revenue grew by twelve percent compared to the previous quarter.</p>"
        "<h2>Highlights</h2>"
        "<p>The new subscription tier accounted for most of the growth.</p>"
        "<h2>Outlook</h2>"
        "<p>We expect moderate growth to continue through the end of the year.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    return FILE_PATH, Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Minimal example

    `DocumentPipeline` wires together parse → process → split → chunk in one call:
    the parser is inferred from the file name, `TokenSplitter` splits at heading
    boundaries within a token budget, and the default `SimpleChunker` renders each
    split into a `Chunk` (with `embedding_content = prompt_content`).

    The library is async-only: in a script, wrap the call in `asyncio.run(...)`; in a
    notebook, `await` works directly.
    """)
    return


@app.cell
async def _(FILE_PATH):
    from ragdoc.pipeline import DocumentPipeline, TokenSplitter

    pipeline = DocumentPipeline(splitter=TokenSplitter(max_tokens=4000))

    chunks = await pipeline.run(FILE_PATH)
    print(f"Produced {len(chunks)} chunks")
    print(chunks[0].prompt_content[:500])
    return DocumentPipeline, TokenSplitter, chunks


@app.cell
def test_minimal_example(chunks):
    # The sample is far below the 4000-token budget, so it stays in one chunk.
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Quarterly Report" in chunk.prompt_content
    assert "subscription tier" in chunk.prompt_content
    assert chunk.embedding_content == chunk.prompt_content  # SimpleChunker
    assert chunk.source_id == "report.html"
    assert chunk.metadata["filename"] == "report.html"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## With LLM enrichment

    Add processing steps between parsing and splitting to enrich the document.
    `ImageSummaryProcessor` writes descriptions into `Image.text_representation`,
    which `render_for_prompt` uses as a text fallback for images.

    This cell needs the `llm` extra and an `OPENAI_API_KEY`; without a key it is skipped.
    """)
    return


@app.cell
async def _(DocumentPipeline, FILE_PATH, TokenSplitter):
    import os

    from openai import AsyncOpenAI

    from ragdoc.processing import (
        HeadingLevelProcessor,
        ImageSummaryProcessor,
        TitleDetectionProcessor,
        openai_image_summarizer,
    )

    enriched_chunks = []
    if os.environ.get("OPENAI_API_KEY"):
        client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        enriched_pipeline = DocumentPipeline(
            processors=[
                HeadingLevelProcessor(),
                TitleDetectionProcessor(),
                ImageSummaryProcessor(summarize=openai_image_summarizer(client, model="gpt-4o")),
            ],
            splitter=TokenSplitter(max_tokens=4000),
        )
        enriched_chunks = await enriched_pipeline.run(FILE_PATH)
        print(f"Produced {len(enriched_chunks)} enriched chunks")
    else:
        print("Set OPENAI_API_KEY in the environment to run this cell")
    return (enriched_chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supported file formats

    | File name ends with | Parser | Notes |
    |-----------|--------|-------|
    | `.html`, `.htm` | `html` | |
    | `.docx`, `.doc` | `pandoc` | Converted via pandoc |
    | `.xlsx` | `xlsx` | Needs the `xlsx` extra |
    | `.pdf` | `azure_di` or `pdf_basic` | Azure Document Intelligence if the `azure-di` extra and credentials are available; otherwise the local `pdf_basic` parser (`pdf` extra) |
    | `.azure.json` | `azure_json` | A saved Azure Document Intelligence result |
    | `_middle.json` | `mineru` | MinerU layout output (`pdf-mineru` extra) |
    | `.ragdoc.json` | `ragdoc_json` | A `Document` previously dumped by ragdoc |

    Use [`load()`][ragdoc.parsing.load] to automatically select the right parser,
    or pass `parser=` to force a specific one.

    ## Next steps

    - See the [Parsing notebook](parsing.py) for a deep dive into each parser
    - See the [Pipeline notebook](pipeline.py) for `DocumentPipeline` and the incremental sync pipelines
    - See the [Rendering notebook](rendering.py) for output format options
    """)
    return


if __name__ == "__main__":
    app.run()
