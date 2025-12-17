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
    # Production Qdrant Pipeline

    End-to-end guide for running an **incremental update** pipeline with:

    | Component | Role |
    |---|---|
    | `MinerUParser` | Parse PDF via MinerU `_middle.json` → `Document` |
    | `LLMHeadingResolver` | Assign semantic heading levels via LLM |
    | `TitleDetectionProcessor` | Strip cover-page preamble |
    | `FootnoteProcessor` | Link inline citation numbers to their definitions |
    | `DocumentDumpProcessor` | Cache enriched documents to disk (pass-through) |
    | `TokenSplitter` | Split into retrieval-sized sub-documents |
    | `LLMChunker` | Generate per-topic `embedding_content` summaries via LLM |
    | `QdrantVectorStore` | Qdrant-backed vector store |
    | `VectorStorePipeline` | Hash-based incremental update — only changed files are re-processed |

    > **Related notebooks:**
    > - [Pipeline](pipeline.py) — `DocumentPipeline` and `VectorStorePipeline` API reference
    > - [Pipeline Walkthrough](pipeline_walkthrough.py) — step-by-step walkthrough of each stage with a real document
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites

    Install the `qdrant` extra and ensure a Qdrant instance is reachable:

    ```bash
    uv add "ragdoc[qdrant]"
    ```

    Required environment variables (loaded from `.env`):

    ```dotenv
    # OpenAI-compatible LLM endpoint — used by LLMHeadingResolver and LLMChunker
    LLM_HEADING_RESOLVER_BASE_URL=https://your-llm-endpoint/v1
    LLM_HEADING_RESOLVER_API_KEY=<your-api-key>
    LLM_HEADING_RESOLVER_MODEL_NAME=your-model-name

    # Qdrant
    QDRANT_URL=http://localhost:6333
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    from pathlib import Path

    from dotenv import load_dotenv
    from httpx import AsyncClient
    from openai import AsyncOpenAI

    from ragdoc.processing import LLMHeadingResolver, LLMHeadingResolverSettings

    load_dotenv()

    settings = LLMHeadingResolverSettings()
    http_client = AsyncClient(verify=False)
    openai_client = AsyncOpenAI(
        max_retries=3,
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        http_client=http_client,
    )
    llm_model = settings.model_name

    print(f"LLM endpoint: {settings.base_url}")
    print(f"LLM model:    {llm_model}")
    return (
        AsyncClient,
        AsyncOpenAI,
        LLMHeadingResolver,
        LLMHeadingResolverSettings,
        Path,
        http_client,
        load_dotenv,
        llm_model,
        openai_client,
        settings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 1 — DocumentPipeline

    `DocumentPipeline` wires together parse → process → split → chunk into a single `await pipeline.run(path)` call.

    ### Processor chain

    | Processor | Type | Purpose |
    |---|---|---|
    | `LLMHeadingResolver` | Async (LLM) | Assigns `h1`/`h2`/`h3` levels; sets `document.title` |
    | `TitleDetectionProcessor` | Sync | Strips preamble elements before the detected title |
    | `FootnoteProcessor` | Async | Inserts `<ref>` tags linking citation numbers to definitions |
    | `DocumentDumpProcessor` | Async | Writes each enriched document to disk as `.ragdoc.json` — useful for debugging and re-ingestion without re-parsing |

    ### Chunker: `LLMChunker`

    `LLMChunker` sends each split document to an LLM which produces 2–5 topic summaries.
    Each summary becomes a separate chunk with the **same `prompt_content`** but a **distinct `embedding_content`**,
    allowing a single section to be retrieved via multiple semantic angles.

    Use `SimpleChunker` instead when LLM cost or latency is a concern —
    it sets `embedding_content = prompt_content` with no LLM calls.
    """)
    return


@app.cell
def _(LLMHeadingResolver, Path, llm_model, openai_client):
    from ragdoc.chunking import LLMChunker
    from ragdoc.parsing.mineru import MinerUParser
    from ragdoc.pipeline import DocumentPipeline, TokenSplitter
    from ragdoc.processing import (
        DocumentDumpProcessor,
        FootnoteProcessor,
        TitleDetectionProcessor,
    )

    DOCUMENTS_DIR = Path("data/raw")
    DUMP_DIR = Path("data/documents_dump")

    doc_pipeline = DocumentPipeline(
        parser=MinerUParser(),
        processors=[
            LLMHeadingResolver(
                client=openai_client,
                model_name=llm_model,
                remove_elements_before_title=True,
            ),
            TitleDetectionProcessor(remove_elements_before_title=True),
            FootnoteProcessor(),
            DocumentDumpProcessor(output_dir=DUMP_DIR),
        ],
        splitter=TokenSplitter(max_tokens=7000, overlap_tokens=200),
        chunker=LLMChunker(client=openai_client, model=llm_model),
        on_error="skip",
    )
    print("DocumentPipeline ready")
    print("  Parser:    MinerUParser")
    print("  Splitter:  TokenSplitter(max_tokens=7000, overlap_tokens=200)")
    print("  Chunker:   LLMChunker (generates per-topic embedding_content)")
    print(f"  Dump dir:  {DUMP_DIR}")
    return (
        DOCUMENTS_DIR,
        DUMP_DIR,
        DocumentDumpProcessor,
        DocumentPipeline,
        FootnoteProcessor,
        LLMChunker,
        MinerUParser,
        TitleDetectionProcessor,
        TokenSplitter,
        doc_pipeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 2 — QdrantVectorStore

    `QdrantVectorStore.create()` creates the collection (if it does not exist) and adds a payload
    index on `source_id`.  It is safe to call on every startup — the 409 response is silently ignored.

    The `vector_size` must match your embedder's output dimension.

    ### Dense-only (single unnamed vector)

    ```python
    store = await QdrantVectorStore.create(qdrant_client, COLLECTION, vector_size=VECTOR_SIZE)
    ```

    ### Hybrid: dense + BM25 sparse (server-side inference)

    Pass `sparse_vectors` to add a BM25 sparse vector field.  Qdrant computes the sparse
    embedding **server-side** on every upsert — no extra embedding call needed from the client.

    ```python
    from ragdoc.integrations.vector_stores import QdrantVectorStore, ServerSideVector
    from ragdoc.pipeline import prompt_content_text

    store = await QdrantVectorStore.create(
        qdrant_client,
        COLLECTION,
        vector_size=VECTOR_SIZE,
        sparse_vectors={
            "sparse": ServerSideVector(model="Qdrant/bm25", text_fn=prompt_content_text),
        },
    )
    ```

    | Parameter | Effect |
    |---|---|
    | `sparse_vectors=None` (default) | Single unnamed dense vector (backward-compatible) |
    | `sparse_vectors={...}` | Named-vector layout: `"dense"` for the dense field + one sparse field per entry |
    | `dense_vector_name` | Override the dense field name (default `"dense"`) |

    `"Qdrant/bm25"` is a built-in model available on all Qdrant deployments (Cloud and self-hosted).
    `text_fn=prompt_content_text` sends the full document text to BM25 — ideal for keyword recall.
    """)
    return


@app.cell
async def _():
    import os

    from qdrant_client import AsyncQdrantClient

    from ragdoc.integrations.vector_stores import QdrantVectorStore, ServerSideVector
    from ragdoc.pipeline import prompt_content_text

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION = "my_docs"
    VECTOR_SIZE = 1536  # match your embedder (e.g. text-embedding-3-small → 1536)

    # qdrant_client = AsyncQdrantClient(QDRANT_URL)
    #
    # Dense-only:
    # store = await QdrantVectorStore.create(qdrant_client, COLLECTION, vector_size=VECTOR_SIZE)
    #
    # Hybrid — dense + BM25 (server-side):
    # store = await QdrantVectorStore.create(
    #     qdrant_client,
    #     COLLECTION,
    #     vector_size=VECTOR_SIZE,
    #     sparse_vectors={
    #         "sparse": ServerSideVector(model="Qdrant/bm25", text_fn=prompt_content_text),
    #     },
    # )

    print(f"QdrantVectorStore would connect to {QDRANT_URL}, collection={COLLECTION!r}")
    print(f"  vector_size={VECTOR_SIZE} — must match the embedder output dimension")
    print("  create() adds a source_id payload index → O(log n) filters in list_source_state / delete_by_source")
    print()
    print("  With sparse_vectors={'sparse': ServerSideVector('Qdrant/bm25')}:")
    print("    → named-vector layout: 'dense' + 'sparse' fields")
    print("    → Qdrant computes BM25 server-side from prompt_content on every upsert")
    return (
        AsyncQdrantClient,
        COLLECTION,
        QDRANT_URL,
        VECTOR_SIZE,
        QdrantVectorStore,
        ServerSideVector,
        os,
        prompt_content_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 3 — VectorStorePipeline with embedders

    `VectorStorePipeline` wraps `DocumentPipeline` with **hash-based change detection**.

    ### Embedder configuration

    Pass an `embedders` dict to compute **client-side** named vectors before upsert.
    Each entry pairs an `Embedder` with the text it should embed:

    | Key | Embedder | `text_fn` | Best for |
    |---|---|---|---|
    | `"dense"` | dense model (e.g. `text-embedding-3-small`) | `embedding_content_text` (default) | Semantic similarity — uses the focused LLM topic summary |
    | `"sparse"` | client-side sparse model (e.g. SPLADE) | `prompt_content_text` | Exact-match recall — uses the full rendered document text |

    > **BM25 via server-side inference (recommended):** Instead of a client-side sparse embedder,
    > configure `sparse_vectors` on `QdrantVectorStore` (Step 2).  Qdrant computes BM25 server-side
    > on every upsert — no `Embedder` needed, no extra network call from the client.

    `VectorStorePipeline` sets two first-class `Chunk` fields before upsert
    — they are **not** injected into `chunk.metadata`:

    | Field | Value |
    |---|---|
    | `chunk.source_id` | `source_id_fn(path)` — default: `path.name` |
    | `chunk.source_hash` | SHA-256 hex digest of raw file bytes |
    """)
    return


@app.cell
def _(doc_pipeline):
    # prompt_content_text is already imported in the QdrantVectorStore cell above
    from ragdoc.pipeline import (
        EmbedderConfig,
        VectorStorePipeline,
        embedding_content_text,
    )

    # class MyDenseEmbedder:
    #     async def embed(self, texts: list[str]) -> list[list[float]]:
    #         # call your embedding API here
    #         ...
    #
    # class MySparseEmbedder:
    #     async def embed(self, texts: list[str]) -> list[list[float]]:
    #         ...
    #
    # # source_id_fn lives on the DocumentPipeline (single source of truth):
    # doc_pipeline = DocumentPipeline(
    #     splitter=..., chunker=...,
    #     source_id_fn=lambda p: str(p.relative_to(DOCUMENTS_DIR)),  # portable multi-dir
    # )
    # vs_pipeline = VectorStorePipeline(
    #     pipeline=doc_pipeline,
    #     vector_store=store,
    #     embedders={
    #         "dense":  EmbedderConfig(MyDenseEmbedder()),
    #         "sparse": EmbedderConfig(MySparseEmbedder(), text_fn=prompt_content_text),
    #     },
    # )

    print("VectorStorePipeline (illustrative — requires live Qdrant + embedders):")
    print("  embedders={")
    print("    'dense':  EmbedderConfig(dense_embedder)                        # embeds embedding_content")
    print("    'sparse': EmbedderConfig(sparse_embedder, text_fn=prompt_content_text)  # embeds prompt_content")
    print("  }")
    return (
        EmbedderConfig,
        VectorStorePipeline,
        embedding_content_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 4 — Incremental update

    Call `run()` with the current list of source paths on every ingestion run.
    The pipeline handles all three cases automatically:

    | Case | What happens |
    |---|---|
    | **New file** | Parsed, chunked, embedded, upserted; `source_id` and `source_hash` set on each chunk |
    | **Modified file** | Old chunks deleted by `source_id`, file re-processed, new chunks upserted |
    | **Removed file** | All chunks for that `source_id` deleted from the store |

    ### First run — all files processed
    """)
    return


@app.cell
def _(DOCUMENTS_DIR, Path):
    paths: list[Path] = list(DOCUMENTS_DIR.glob("**/*_middle.json")) if DOCUMENTS_DIR.exists() else []

    # result = await vs_pipeline.run(paths)
    # print(f"processed: {len(result.processed)}, skipped: {len(result.skipped)}, deleted: {len(result.deleted)}")
    # for path, exc in result.errors:
    #     print(f"  FAILED {path}: {exc}")

    print(f"Source paths: {len(paths)} files in {DOCUMENTS_DIR}")
    print("First run → all files processed (vector store has no stored hashes yet)")
    return (paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subsequent runs — unchanged files skipped

    On re-runs, `VectorStorePipeline.plan()` reads all stored hashes in one call via
    `list_source_state()`. If a file's bytes are unchanged (its `source_hash` matches), it is
    skipped — it appears in `result.skipped` and in none of the ChangeSet lists:
    """)
    return


@app.cell
def _():
    # result2 = await vs_pipeline.run(paths)
    # assert len(result2.skipped) == len(paths)
    # assert len(result2.processed) == 0
    print("Second run (no changes) → all files skipped, nothing re-processed")
    print("  result.processed == [], result.skipped == paths")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Running against a directory

    `run_directory` discovers files automatically using a glob pattern:
    """)
    return


@app.cell
def _(DOCUMENTS_DIR):
    # result = await vs_pipeline.run_directory(DOCUMENTS_DIR, glob="**/*_middle.json")
    print(f"run_directory({DOCUMENTS_DIR!r}, glob='**/*_middle.json')")
    print("  → discovers files, then calls run() automatically")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Source ID strategies

    The default `source_id_fn` uses the bare filename (`path.name`).
    Collisions arise when the same filename appears in multiple directories —
    `run()` raises `ValueError` before any processing if two paths map to the same ID.

    | Strategy | Expression | Use case |
    |---|---|---|
    | Filename (default) | `lambda p: p.name` | Simple flat folders |
    | Full path | `lambda p: str(p)` | Multi-dir, single machine |
    | Relative path | `lambda p: str(p.relative_to(base_dir))` | Portable multi-dir |

    `source_id_fn` is set on the `DocumentPipeline` (the single source of truth); the
    `VectorStorePipeline` reads it from there for collision/orphan checks.

    ```python
    doc_pipeline = DocumentPipeline(
        splitter=..., chunker=...,
        source_id_fn=lambda p: str(p.relative_to(DOCUMENTS_DIR)),
    )
    vs_pipeline = VectorStorePipeline(pipeline=doc_pipeline, vector_store=store)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Payload schema

    Each Qdrant point stores the full `Chunk` as payload via `Chunk.model_dump(mode="json")`.
    The provenance fields live at the top level — they are **not** injected into `chunk.metadata`:

    | Payload key | Chunk field | Purpose |
    |---|---|---|
    | `source_id` | `chunk.source_id` | Groups chunks by source for cleanup |
    | `source_hash` | `chunk.source_hash` | Detects whether the source has changed |
    | `source_path` | `chunk.source_path` | Full path to the source file |
    | `prompt_content` | `chunk.prompt_content` | Full-fidelity text for LLM context |
    | `embedding_content` | `chunk.embedding_content` | Topic summary used to generate the embedding |
    | `metadata` | `chunk.metadata` | User-defined key/value data |
    | `created_at` | `chunk.created_at` | ISO 8601 creation timestamp |

    The point ID is `chunk.id` (not stored in the payload).
    Named vectors are stored in Qdrant's vector fields, not the payload.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## See Also

    - [Pipeline](pipeline.py) — `DocumentPipeline`, `VectorStorePipeline`, and `TokenSplitter` API
    - [Pipeline Walkthrough](pipeline_walkthrough.py) — stage-by-stage walkthrough with intermediate output
    - [Chunking](chunking.py) — `SimpleChunker` vs `LLMChunker` in depth
    - [API Reference: Pipeline](../api/pipeline.md)
    - [API Reference: Integrations](../api/integrations.md)
    """)
    return


if __name__ == "__main__":
    app.run()
