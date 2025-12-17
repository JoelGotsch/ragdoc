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
    # Pipeline

    The `ragdoc.pipeline` module provides a high-level facade that wires together all four
    stages — **parse → process → split → chunk** — into a single call.

    Use `DocumentPipeline` when you want to go from a file on disk to `Chunk` objects in one
    step. Use the individual stage APIs when you need fine-grained control — custom parsers,
    partial re-processing, or stage-level debugging.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario A — Linear pipeline

    ### Minimal example

    `DocumentPipeline` defaults:

    | Stage | Default |
    |-------|---------|
    | Parser | `AutoParser` — picks parser from file extension |
    | Processors | none (identity pass-through) |
    | Splitter | none — whole document → one chunk |
    | Chunker | `SimpleChunker` — one chunk per document |
    """)
    return


@app.cell
async def _():
    from pathlib import Path

    from ragdoc.pipeline import DocumentPipeline

    pipeline = DocumentPipeline()
    # chunks = await pipeline.run(Path("report.docx"))
    # print(f"Produced {len(chunks)} chunks")
    print("DocumentPipeline() ready — call await pipeline.run(Path('report.docx')) to run")
    return (DocumentPipeline, Path, pipeline)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a splitter

    To split documents into retrieval-sized chunks, pass a
    [`TokenSplitter`][ragdoc.pipeline.TokenSplitter].
    Chunk size is independent of your LLM's context window — chunks are sized
    for retrieval quality, typically 1/50th to 1/10th of the context window.
    Test different `max_tokens` values for your use case:
    """)
    return


@app.cell
async def _(DocumentPipeline, Path):
    from ragdoc.pipeline import TokenSplitter

    pipeline_with_split = DocumentPipeline(
        splitter=TokenSplitter(max_tokens=4000),
    )
    # chunks = await pipeline_with_split.run(Path("annual_report.docx"))
    print("Pipeline with TokenSplitter(max_tokens=4000) ready")
    return (TokenSplitter, pipeline_with_split)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding processors
    """)
    return


@app.cell
async def _(DocumentPipeline, TokenSplitter):
    from ragdoc.processing import HeadingLevelProcessor, TitleDetectionProcessor

    pipeline_with_processors = DocumentPipeline(
        processors=[
            HeadingLevelProcessor(),
            TitleDetectionProcessor(),
        ],
        splitter=TokenSplitter(max_tokens=4000),
    )
    print("Pipeline with HeadingLevelProcessor + TitleDetectionProcessor ready")
    return (HeadingLevelProcessor, TitleDetectionProcessor, pipeline_with_processors)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Custom parser

    Any callable `(Path) -> Document` satisfies the `Parser` protocol:
    """)
    return


@app.cell
def _(DocumentPipeline, Path):
    from ragdoc.document import Document

    def my_parser(path: Path) -> Document:
        # e.g. call an internal API, parse a custom format, …
        return Document()

    pipeline_custom = DocumentPipeline(parser=my_parser)
    print("Pipeline with custom parser callable ready")
    return (Document, my_parser, pipeline_custom)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deterministic chunk IDs

    `SimpleChunker` uses `Document.content_hash()` as the default chunk ID. The same file
    content always produces the same chunk IDs, making vector-store upserts idempotent.
    """)
    return


@app.cell
async def _(DocumentPipeline, Path, TokenSplitter):
    pipeline_idem = DocumentPipeline(splitter=TokenSplitter())
    # chunks1 = await pipeline_idem.run(Path("report.docx"))
    # chunks2 = await pipeline_idem.run(Path("report.docx"))
    # assert chunks1[0].id == chunks2[0].id  # always True for same content
    print("Chunk IDs are deterministic: same content → same ID on every run")
    return (pipeline_idem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Scenario D — Concurrent processing and streaming

    ### Processing many files

    `run_many` fans out across a list of paths and returns a `PipelineResult` with all chunks
    aggregated:
    """)
    return


@app.cell
async def _(DocumentPipeline, Path, TokenSplitter):
    pipeline_many = DocumentPipeline(splitter=TokenSplitter())
    paths: list[Path] = []  # list(Path("docs/").glob("**/*.docx"))

    # result = await pipeline_many.run_many(paths, concurrency=8)
    # print(f"Produced {len(result.chunks)} chunks from {len(paths)} files")
    print("run_many fans out across multiple files with bounded concurrency")
    return (paths, pipeline_many)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Error handling

    By default errors propagate immediately (`on_error="raise"`).
    Use `on_error="skip"` to collect failures and continue:
    """)
    return


@app.cell
async def _(DocumentPipeline, TokenSplitter, paths):
    pipeline_skip = DocumentPipeline(
        splitter=TokenSplitter(),
        on_error="skip",
    )
    # result = await pipeline_skip.run_many(paths)
    # for path, exc in result.errors:
    #     print(f"  FAILED {path}: {exc}")
    print("on_error='skip' collects failures into result.errors without stopping processing")
    return (pipeline_skip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming (memory-efficient)

    `stream` yields a batch of chunks per file as it completes, so you can hand off each
    batch to a vector store before the next file starts:
    """)
    return


@app.cell
async def _(pipeline_skip, paths):
    # async for batch in pipeline_skip.stream(paths, concurrency=4):
    #     await vector_store.upsert(batch)
    print("stream() yields one batch per file — memory-efficient for large corpora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TokenSplitter reference

    ```python
    TokenSplitter(
        max_tokens=7000,    # token budget per split
        overlap_tokens=200, # overlap between consecutive splits
        renderer=None,      # Renderer for token measurement; defaults to MARKDOWN + render_for_prompt
        tokenizer=None,     # tiktoken cl100k_base by default
    )
    ```

    ---

    ## Scenario B — Incremental update

    `VectorStorePipeline` wraps `DocumentPipeline` and adds hash-based change detection
    so that a corpus of documents can be kept in sync with a vector store incrementally —
    only changed or new files are re-processed.
    """)
    return


@app.cell
async def _():
    from ragdoc.pipeline import VectorStorePipeline

    # vs_pipeline = VectorStorePipeline(
    #     pipeline=DocumentPipeline(splitter=TokenSplitter()),
    #     vector_store=my_vector_store,
    # )
    print("VectorStorePipeline adds hash-based change detection on top of DocumentPipeline")
    print("The vector store is the single source of truth — no local state file needed")
    return (VectorStorePipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two-hash design

    | Hash | What it covers | Purpose |
    |------|---------------|---------|
    | `source_hash` | Raw file bytes | Cheap change detection before parsing |
    | `chunk.id` | Rendered document content | Stable, idempotent chunk IDs for the vector store |

    These are orthogonal. `source_hash` is never exposed as a chunk ID.

    ### What the pipeline handles automatically

    | Case | What happens |
    |------|-------------|
    | **New file** | Parsed, chunked, upserted; state saved. |
    | **Modified file** (byte hash changed) | Old chunks deleted, file re-processed, new chunks upserted. |
    | **Removed file** (path no longer in source list) | Old chunks deleted; entry removed from state. |

    ## See Also

    - [API Reference: Pipeline][ragdoc.pipeline]
    - [Qdrant Pipeline](qdrant_pipeline.py) — full production setup: MinerU + processors + LLMChunker + QdrantVectorStore
    - [Pipeline Walkthrough](pipeline_walkthrough.py) — stage-by-stage walkthrough with intermediate output
    - [Chunking notebook](chunking.py) — `SimpleChunker`, `LLMChunker`
    - [Splitting notebook](splitting.py) — lower-level splitting API
    """)
    return


if __name__ == "__main__":
    app.run()
