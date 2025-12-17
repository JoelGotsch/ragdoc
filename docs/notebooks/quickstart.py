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

    This notebook walks through a complete parsing → splitting → chunking pipeline.
    Fill in `FILE_PATH` below with a real document to run the examples end-to-end.
    """)
    return


@app.cell
def _():
    # Set this to a real file on disk to run the examples
    FILE_PATH = "report.docx"
    return (FILE_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Minimal example

    `DocumentPipeline` wires together parse → process → split → chunk in one call:
    the parser is inferred from the file extension, `TokenSplitter` splits at heading
    boundaries within a token budget, and the default `SimpleChunker` renders each
    split into a `Chunk` (with `embedding_content = prompt_content`).
    """)
    return


@app.cell
async def _(FILE_PATH):
    from pathlib import Path

    from ragdoc.pipeline import DocumentPipeline, TokenSplitter

    pipeline = DocumentPipeline(splitter=TokenSplitter(max_tokens=4000))

    chunks = []
    if Path(FILE_PATH).exists():  # noqa: ASYNC240 — placeholder-file guard in a doc example
        chunks = await pipeline.run(Path(FILE_PATH))
        print(f"Produced {len(chunks)} chunks")
        print(chunks[0].prompt_content[:500])
    else:
        print(f"{FILE_PATH!r} not found — set FILE_PATH to a real document to run this cell")
    return (DocumentPipeline, Path, TokenSplitter, chunks, pipeline)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## With LLM enrichment

    Add processing steps between parsing and splitting to enrich the document.
    `ImageSummaryProcessor` writes descriptions into `Image.text_representation`,
    which `render_for_prompt` uses as a text fallback for images.
    """)
    return


@app.cell
async def _(DocumentPipeline, FILE_PATH, Path, TokenSplitter):
    import os

    from openai import AsyncOpenAI

    from ragdoc.processing import (
        HeadingLevelProcessor,
        ImageSummaryProcessor,
        TitleDetectionProcessor,
        openai_image_summarizer,
    )

    enriched_chunks = []
    if Path(FILE_PATH).exists() and os.environ.get("OPENAI_API_KEY"):
        client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        enriched_pipeline = DocumentPipeline(
            processors=[
                HeadingLevelProcessor(),
                TitleDetectionProcessor(),
                ImageSummaryProcessor(summarize=openai_image_summarizer(client, model="gpt-4o")),
            ],
            splitter=TokenSplitter(max_tokens=4000),
        )
        enriched_chunks = await enriched_pipeline.run(Path(FILE_PATH))
        print(f"Produced {len(enriched_chunks)} enriched chunks")
    else:
        print("Set FILE_PATH to a real document and OPENAI_API_KEY in the environment to run this cell")
    return (enriched_chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supported file formats

    | Extension | Parser | Notes |
    |-----------|--------|-------|
    | `.docx`, `.doc` | Pandoc | Requires Pandoc installed |
    | `.html` | HTML parser | |
    | `.xlsx` | Excel parser | |
    | `.azure.json` | Azure DI | Azure Document Intelligence output |
    | `.pdf` | Azure DI | Dispatches to Azure Document Intelligence |
    | `_middle.json` | MinerU | MinerU layout output |

    Use [`load()`][ragdoc.parsing.load] to automatically select the right parser,
    or pass `parser=` to force a specific one.

    ## Next steps

    - See the [Parsing notebook](parsing.py) for a deep dive into each parser
    - See the [Pipeline notebook](pipeline.py) for the high-level `DocumentPipeline` facade
    - See the [Rendering notebook](rendering.py) for output format options
    """)
    return


if __name__ == "__main__":
    app.run()
