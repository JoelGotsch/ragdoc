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

    The simplest pipeline: infer source type from file extension, parse, split at heading
    boundaries, and render each section into a `Chunk`.
    """)
    return


@app.cell
async def _(FILE_PATH):
    from ragdoc.parsing import load
    from ragdoc.rendering import Renderer, render_for_prompt, OutputFormat
    from ragdoc.splitting import split_by_headings
    from ragdoc.chunking import Chunk

    # 1. Parse — infers parser from file extension
    document = await load(FILE_PATH)

    # 2. Split into sections at heading boundaries
    sections = split_by_headings(document)

    # 3. Render and chunk
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)

    chunks = [
        Chunk(
            prompt_content=renderer.render(doc),
            embedding_content=renderer.render(doc),
            metadata=doc.metadata,
        )
        for doc in sections
    ]

    print(f"Produced {len(chunks)} chunks")
    print(chunks[0].prompt_content[:500])
    return (Chunk, OutputFormat, Renderer, chunks, document, render_for_prompt,
            sections, split_by_headings)


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
async def _(FILE_PATH):
    # Private imports (_-prefix) to avoid name clashes with the minimal example cell above
    from ragdoc.parsing import load as _load
    from ragdoc.rendering import Renderer as _Renderer, render_for_prompt as _rfp, OutputFormat as _OutputFormat
    from ragdoc.processing import (
        ProcessingPipeline as _ProcessingPipeline,
        HeadingLevelProcessor as _HeadingLevelProcessor,
        TitleDetectionProcessor as _TitleDetectionProcessor,
        ImageSummaryProcessor as _ImageSummaryProcessor,
    )
    from ragdoc.splitting import split_by_headings as _split_by_headings
    from ragdoc.chunking import Chunk as _Chunk
    from openai import AsyncOpenAI as _AsyncOpenAI

    async def build_chunks(file_path: str) -> list:
        client = _AsyncOpenAI()
        document = await _load(file_path)
        pipeline = _ProcessingPipeline([
            _HeadingLevelProcessor(),
            _TitleDetectionProcessor(),
            _ImageSummaryProcessor(client=client),
        ])
        document = await pipeline.process(document)
        sections = _split_by_headings(document)
        renderer = _Renderer(format=_OutputFormat.MARKDOWN, element_renderer=_rfp)
        # SimpleChunker sets embedding_content = prompt_content
        return [
            _Chunk(
                prompt_content=renderer.render(doc),
                embedding_content=renderer.render(doc),
                metadata=doc.metadata,
            )
            for doc in sections
        ]

    enriched_chunks = await build_chunks(FILE_PATH)
    print(f"Produced {len(enriched_chunks)} enriched chunks")
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
