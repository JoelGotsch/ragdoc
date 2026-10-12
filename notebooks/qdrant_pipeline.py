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
    # Qdrant Pipeline

    End-to-end guide for keeping a **Qdrant** collection in sync with a folder of documents:

    | Component | Role |
    |---|---|
    | `DocumentPipeline` | parse → process → split → chunk one file |
    | `QdrantVectorStore` | Qdrant-backed `VectorStore` |
    | `EmbedderConfig` | which embedder embeds which text of a chunk |
    | `VectorStorePipeline` | hash-based incremental sync — only changed files are re-processed |
    | `QdrantDocumentStore` | optional: keep parsed `Document`s in Qdrant too and chunk from there |

    **This notebook runs offline.** It uses Qdrant's in-process mode
    (`AsyncQdrantClient(":memory:")`), a small HTML corpus in a temp directory, the default
    `SimpleChunker` and a deterministic fake embedder — so every sync call below really
    executes, and the `test_*` cells assert the results. Each step also shows what you would
    swap in for production (a Qdrant server, a real embedder, LLM-based processors and
    chunker).

    > **Related notebooks:**
    > - [Pipeline](pipeline.py) — `DocumentPipeline`, `VectorStorePipeline`, `plan()` / `apply()` in detail
    > - [Pipeline Walkthrough](pipeline_walkthrough.py) — step-by-step walkthrough of each stage with a real document
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites

    Install the `qdrant` extra:

    ```bash
    uv add "ragdoc[qdrant]"
    ```

    For a production setup you also need a reachable Qdrant instance, and — if you use the
    LLM-based components shown in Step 1 — an OpenAI-compatible endpoint (`ragdoc[llm]`).
    Configuration is read from the environment or a `.env` file:

    ```dotenv
    # Qdrant
    QDRANT_URL=http://localhost:6333

    # OpenAI-compatible LLM endpoint — used by LLMHeadingResolver and LLMChunker
    LLM_HEADING_RESOLVER_API_KEY=<your-api-key>
    LLM_HEADING_RESOLVER_MODEL_NAME=gpt-4o-mini
    # Optional: only for an endpoint other than api.openai.com
    # LLM_HEADING_RESOLVER_BASE_URL=https://your-llm-endpoint/v1
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample corpus
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    DOCUMENTS_DIR = Path(tempfile.mkdtemp(prefix="ragdoc-qdrant-"))

    (DOCUMENTS_DIR / "onboarding.html").write_text(
        "<html><body><h1>Onboarding</h1>"
        "<h2>First day</h2><p>Collect your laptop and badge from the front desk, then meet your buddy.</p>"
        "<h2>First week</h2><p>Set up your development environment and ship a small change.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (DOCUMENTS_DIR / "security.html").write_text(
        "<html><body><h1>Security Policy</h1>"
        "<p>Use a password manager and enable two-factor authentication on every account.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (DOCUMENTS_DIR / "travel.html").write_text(
        "<html><body><h1>Travel Policy</h1>"
        "<p>Book trains for journeys under four hours. Flights need approval from your lead.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    paths = sorted(DOCUMENTS_DIR.glob("*.html"))
    print(f"{len(paths)} files in {DOCUMENTS_DIR}")
    return DOCUMENTS_DIR, Path, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 1 — DocumentPipeline

    `DocumentPipeline` wires parse → process → split → chunk into a single
    `await pipeline.run(path)` call. The offline pipeline below uses the registry parser
    (`load`), no processors, a `TokenSplitter` and the default `SimpleChunker`
    (`embedding_content = prompt_content`, no LLM calls).

    `source_id_fn` lives on the `DocumentPipeline`; the sync pipeline reads it from there.
    Here the id is the path relative to the corpus directory.
    """)
    return


@app.cell
def _(DOCUMENTS_DIR):
    from ragdoc.pipeline import DocumentPipeline, TokenSplitter

    doc_pipeline = DocumentPipeline(
        splitter=TokenSplitter(max_tokens=7000, overlap_tokens=200),
        source_id_fn=lambda p: str(p.relative_to(DOCUMENTS_DIR)),
        on_error="skip",
    )
    print("DocumentPipeline ready: load → TokenSplitter(7000/200) → SimpleChunker")
    return DocumentPipeline, TokenSplitter, doc_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Production variant: MinerU + LLM processors + `LLMChunker`

    For PDFs parsed with MinerU, a typical production pipeline adds LLM-based enrichment:

    | Stage | Component | Purpose |
    |---|---|---|
    | Parser | `MinerUParser` | MinerU `_middle.json` → `Document` |
    | Processor | `LLMHeadingResolver` | Assigns `h1`/`h2`/`h3` levels; sets `document.title` |
    | Processor | `TitleDetectionProcessor` | Strips preamble elements before the detected title |
    | Processor | `FootnoteProcessor` | Inserts `<ref>` tags linking citation numbers to definitions |
    | Processor | `DocumentDumpProcessor` | Writes each enriched document to disk as `.ragdoc.json` |
    | Chunker | `LLMChunker` | Several topic summaries per split: same `prompt_content`, distinct `embedding_content` |

    The cell below builds that pipeline **only if** `LLM_HEADING_RESOLVER_API_KEY` is set
    (constructing it makes no LLM call). Without a key it is skipped and the rest of the
    notebook is unaffected.
    """)
    return


@app.cell
def _(DOCUMENTS_DIR, DocumentPipeline, TokenSplitter, mo):
    import os

    from dotenv import load_dotenv

    load_dotenv()
    mo.stop(
        not os.environ.get("LLM_HEADING_RESOLVER_API_KEY"),
        mo.md("*Skipped: set `LLM_HEADING_RESOLVER_API_KEY` to build the LLM-based pipeline.*"),
    )

    from openai import AsyncOpenAI

    from ragdoc.chunking import LLMChunker
    from ragdoc.parsing.mineru import MinerUParser
    from ragdoc.processing import (
        DocumentDumpProcessor,
        FootnoteProcessor,
        LLMHeadingResolver,
        LLMHeadingResolverSettings,
        TitleDetectionProcessor,
    )

    _settings = LLMHeadingResolverSettings()
    openai_client = AsyncOpenAI(
        api_key=_settings.api_key.get_secret_value(),
        base_url=_settings.base_url,  # None → api.openai.com
        max_retries=3,
    )

    llm_doc_pipeline = DocumentPipeline(
        parser=MinerUParser(),
        processors=[
            LLMHeadingResolver(
                client=openai_client,
                model_name=_settings.model_name,
                remove_elements_before_title=True,
            ),
            TitleDetectionProcessor(remove_elements_before_title=True),
            FootnoteProcessor(),
            DocumentDumpProcessor(output_dir=DOCUMENTS_DIR / "documents_dump"),
        ],
        splitter=TokenSplitter(max_tokens=7000, overlap_tokens=200),
        chunker=LLMChunker(client=openai_client, model=_settings.model_name),
        on_error="skip",
    )
    print(f"LLM pipeline ready (model: {_settings.model_name})")
    return (llm_doc_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 2 — QdrantVectorStore

    `QdrantVectorStore.create()` creates the collection and a payload index on `source_id`,
    and returns the store. `vector_size` must match your embedder's output dimension.

    `create()` is idempotent, against a server and in-process alike: it checks whether the
    collection exists and uses an existing one as is, so it can be called on every startup.
    The in-process mode used in this notebook has no payload indexes (Qdrant emits a warning
    saying so, silenced below).

    ```python
    # Production: connect to a server instead of ":memory:"
    qdrant_client = AsyncQdrantClient(os.environ["QDRANT_URL"])
    ```
    """)
    return


@app.cell
async def _():
    import warnings

    from qdrant_client import AsyncQdrantClient

    from ragdoc.integrations.vector_stores import QdrantVectorStore

    COLLECTION = "my_docs"
    VECTOR_SIZE = 8  # must match the embedder (e.g. text-embedding-3-small → 1536)

    qdrant_client = AsyncQdrantClient(":memory:")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")
        store = await QdrantVectorStore.create(qdrant_client, COLLECTION, vector_size=VECTOR_SIZE)

    print(f"collection {COLLECTION!r} created, vector_size={VECTOR_SIZE}")
    return COLLECTION, VECTOR_SIZE, qdrant_client, store, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hybrid: dense + BM25 sparse (server-side inference)

    Pass `sparse_vectors` to add a BM25 sparse vector field. Qdrant computes the sparse
    embedding **server-side** on every upsert — no extra embedding call from the client.
    This needs a Qdrant server, so it is shown here but not executed:

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
    | `sparse_vectors=None` (default) | Single unnamed dense vector |
    | `sparse_vectors={...}` | Named-vector layout: `"dense"` for the dense field + one sparse field per entry |
    | `dense_vector_name` | Override the dense field name (default `"dense"`) |

    `text_fn=prompt_content_text` sends the full chunk text to BM25 — a good fit for keyword recall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 3 — VectorStorePipeline with embedders

    `VectorStorePipeline` wraps the `DocumentPipeline` with **hash-based change detection**
    and embeds chunks before writing them.

    Pass an `embedders` dict to compute **client-side** named vectors. Each entry pairs an
    `Embedder` (anything with `async embed(texts) -> vectors`) with the text it should embed:

    | Key | Embedder | `text_fn` | Best for |
    |---|---|---|---|
    | `"dense"` | dense model (e.g. `text-embedding-3-small`) | `embedding_content_text` (default) | Semantic similarity |
    | `"sparse"` | client-side sparse model (e.g. SPLADE) | `prompt_content_text` | Exact-match recall on the full text |

    The fake embedder below is deterministic and offline. In production use `OpenAIEmbedder`
    (any OpenAI-compatible embeddings endpoint) or your own class:

    ```python
    from openai import AsyncOpenAI
    from ragdoc.pipeline import EmbedderConfig, OpenAIEmbedder

    embedders = {"dense": EmbedderConfig(OpenAIEmbedder(AsyncOpenAI(), model="text-embedding-3-small"))}
    ```

    The pipeline stamps `chunk.source_hash` (SHA-256 of the raw file bytes) before writing;
    `chunk.source_id` comes from the `DocumentPipeline`'s `source_id_fn`. Both are
    first-class `Chunk` fields — nothing is injected into `chunk.metadata`.
    """)
    return


@app.cell
def _(VECTOR_SIZE, doc_pipeline, store):
    import hashlib

    from ragdoc.pipeline import EmbedderConfig, VectorStorePipeline, embedding_content_text

    class FakeEmbedder:
        """Deterministic stand-in for a real embedding model."""

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[b / 255 for b in hashlib.sha256(t.encode("utf-8")).digest()[:VECTOR_SIZE]] for t in texts]

    fake_embedder = FakeEmbedder()
    vs_pipeline = VectorStorePipeline(
        pipeline=doc_pipeline,
        vector_store=store,
        embedders={"dense": EmbedderConfig(fake_embedder, text_fn=embedding_content_text)},
    )
    print("VectorStorePipeline ready")
    return EmbedderConfig, VectorStorePipeline, fake_embedder, vs_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 4 — Incremental update

    Call `run()` with the current list of source paths on every ingestion run:

    | Case | What happens |
    |---|---|
    | **New file** | Parsed, chunked, embedded, upserted |
    | **Unchanged file** | Skipped (its `source_hash` matches the store) |
    | **Modified file** | Re-processed; the new chunks are written first, then the source's stale chunks are removed (`replace_source`) |
    | **Removed file** | Left in the store by default; deleted only with `delete_orphans=True` |

    ### First run — all files processed
    """)
    return


@app.cell
async def _(paths, store, vs_pipeline):
    first_run = await vs_pipeline.run(paths)
    sources_after_first_run = await store.list_source_ids()

    print(f"processed={sorted(first_run.processed)}")
    print(f"skipped={first_run.skipped} deleted={first_run.deleted} errors={first_run.errors}")
    return first_run, sources_after_first_run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subsequent runs — unchanged files skipped

    On re-runs the pipeline reads all stored hashes in one call via `list_source_state()`.
    A file whose bytes are unchanged is skipped — it appears in `result.skipped`.
    `run_directory` is a shortcut that globs a directory and calls `run()`:
    """)
    return


@app.cell
async def _(DOCUMENTS_DIR, first_run, vs_pipeline):
    _ = first_run  # run after the first sync
    second_run = await vs_pipeline.run_directory(DOCUMENTS_DIR, glob="*.html")
    print(f"processed={second_run.processed} skipped={sorted(second_run.skipped)}")
    return (second_run,)


@app.cell
def test_first_run_adds_second_run_skips(first_run, second_run, sources_after_first_run):
    _all = ["onboarding.html", "security.html", "travel.html"]
    assert sorted(first_run.processed) == _all
    assert first_run.skipped == [] and first_run.errors == []
    assert sources_after_first_run == set(_all)
    assert second_run.processed == []
    assert sorted(second_run.skipped) == _all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A modified file — review with `plan()`, write with `apply()`

    `plan()` computes a `ChangeSet` without writing; `apply()` embeds and writes it. Use
    the pair when you want to look at (or edit, or save to disk) what is about to change.
    """)
    return


@app.cell
async def _(DOCUMENTS_DIR, paths, second_run, store, vs_pipeline):
    _ = second_run  # run after the cells above
    (DOCUMENTS_DIR / "security.html").write_text(
        "<html><body><h1>Security Policy</h1>"
        "<p>Hardware security keys are mandatory for all production access.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    update_plan = await vs_pipeline.plan(paths)
    print(f"to_add={[sc.source_id for sc in update_plan.to_add]}")
    print(f"to_update={[sc.source_id for sc in update_plan.to_update]}")

    update_result = await vs_pipeline.apply(update_plan)
    state_after_update = await store.list_source_state()
    print(f"apply(): processed={update_result.processed} errors={update_result.errors}")
    return state_after_update, update_plan, update_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A removed file — `delete_orphans`

    Chunks of a file that is no longer in the list are **kept** unless you pass
    `delete_orphans=True`. Only do that when the list is your complete corpus: with a partial
    list, everything else in the collection would count as an orphan.
    """)
    return


@app.cell
async def _(DOCUMENTS_DIR, store, update_result, vs_pipeline):
    _ = update_result  # run after the cells above
    (DOCUMENTS_DIR / "travel.html").unlink()
    _remaining = sorted(DOCUMENTS_DIR.glob("*.html"))

    kept_run = await vs_pipeline.run(_remaining)
    sources_after_kept = await store.list_source_ids()

    orphan_run = await vs_pipeline.run(_remaining, delete_orphans=True)
    sources_after_orphan = await store.list_source_ids()

    print(f"default:             deleted={kept_run.deleted} store={sorted(sources_after_kept)}")
    print(f"delete_orphans=True: deleted={orphan_run.deleted} store={sorted(sources_after_orphan)}")
    return kept_run, orphan_run, sources_after_kept, sources_after_orphan


@app.cell
def test_update_and_orphans(
    kept_run,
    orphan_run,
    sources_after_kept,
    sources_after_orphan,
    state_after_update,
    update_plan,
    update_result,
):
    # Only the edited file is planned and written.
    assert update_plan.to_add == []
    assert [sc.source_id for sc in update_plan.to_update] == ["security.html"]
    assert update_result.processed == ["security.html"]
    assert update_result.errors == []
    # The store now records the new file hash for it.
    assert state_after_update["security.html"].source_hash == update_plan.to_update[0].source_hash
    # Removed file: kept by default, deleted on opt-in.
    assert kept_run.deleted == []
    assert sources_after_kept == {"onboarding.html", "security.html", "travel.html"}
    assert orphan_run.deleted == ["travel.html"]
    assert sources_after_orphan == {"onboarding.html", "security.html"}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## What is stored

    Each Qdrant point stores the `Chunk` as payload via `Chunk.model_dump(mode="json")`
    (without `id` and `named_embeddings`). The provenance fields live at the top level —
    they are **not** inside `metadata`:

    | Payload key | Chunk field | Purpose |
    |---|---|---|
    | `source_id` | `chunk.source_id` | Groups chunks by source for cleanup |
    | `source_hash` | `chunk.source_hash` | File-byte hash: detects whether the source file changed |
    | `content_hash` | `chunk.content_hash` | Document content hash: change detection when chunking from a document store |
    | `source_path` | `chunk.source_path` | Full path to the source file |
    | `prompt_content` | `chunk.prompt_content` | Full-fidelity text for LLM context |
    | `embedding_content` | `chunk.embedding_content` | Text the dense embedding was computed from |
    | `metadata` | `chunk.metadata` | Document metadata (user-defined keys plus e.g. `filename`) |
    | `created_at` | `chunk.created_at` | ISO 8601 creation timestamp |

    The point ID is `chunk.id` (a UUID string, not stored in the payload). Vectors are stored
    in Qdrant's vector fields, not the payload.
    """)
    return


@app.cell
async def _(COLLECTION, fake_embedder, orphan_run, qdrant_client):
    _ = orphan_run  # run after the sync cells above
    _points, _next = await qdrant_client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)
    payload_keys = sorted(_points[0].payload)
    print(f"{len(_points)} points; payload keys: {payload_keys}")

    # A nearest-neighbour query with the embedding of one chunk's text returns that chunk.
    _security = next(p for p in _points if p.payload["source_id"] == "security.html")
    _query_vector = (await fake_embedder.embed([_security.payload["embedding_content"]]))[0]
    _hits = await qdrant_client.query_points(COLLECTION, query=_query_vector, limit=1)
    top_hit_payload = _hits.points[0].payload
    print(f"top hit: {top_hit_payload['source_id']}")
    return payload_keys, top_hit_payload


@app.cell
def test_payload_schema(payload_keys, top_hit_payload):
    assert payload_keys == [
        "content_hash",
        "created_at",
        "embedding_content",
        "metadata",
        "prompt_content",
        "source_hash",
        "source_id",
        "source_path",
    ]
    assert top_hit_payload["source_id"] == "security.html"
    assert "Hardware security keys" in top_hit_payload["prompt_content"]
    assert top_hit_payload["metadata"]["filename"] == "security.html"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Optional — two boundaries with `QdrantDocumentStore`

    Instead of going from files straight to chunks, you can store the parsed and processed
    `Document`s first and chunk from there. Re-chunking (new splitter settings, a different
    chunker) then needs no re-parsing and no repeated LLM processing:

    1. `DocumentStorePipeline(ingest=IngestPipeline(...), document_store=...)` — files → documents
    2. `VectorStorePipeline.from_document_store(chunk=ChunkPipeline(...), ...)` — documents → chunks

    Boundary 2 detects changes by the document's `content_hash`, and takes `source_id`
    strings (or nothing, meaning every stored document) instead of paths.
    """)
    return


@app.cell
async def _(
    DOCUMENTS_DIR,
    EmbedderConfig,
    TokenSplitter,
    VECTOR_SIZE,
    VectorStorePipeline,
    fake_embedder,
    orphan_run,
    qdrant_client,
    warnings,
):
    from ragdoc.integrations.document_stores import QdrantDocumentStore
    from ragdoc.integrations.vector_stores import QdrantVectorStore as _QdrantVectorStore
    from ragdoc.pipeline import ChunkPipeline, DocumentStorePipeline, IngestPipeline

    _ = orphan_run  # run after the direct-mode cells (they edit the corpus)
    _files = sorted(DOCUMENTS_DIR.glob("*.html"))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")
        document_store = await QdrantDocumentStore.create(qdrant_client, "my_documents")
        chunk_store = await _QdrantVectorStore.create(qdrant_client, "my_chunks", vector_size=VECTOR_SIZE)

    # Boundary 1: files → documents
    documents_pipeline = DocumentStorePipeline(ingest=IngestPipeline(), document_store=document_store)
    b1_first = await documents_pipeline.run(_files)
    b1_second = await documents_pipeline.run(_files)

    # Boundary 2: documents → chunks
    chunks_pipeline = VectorStorePipeline.from_document_store(
        chunk=ChunkPipeline(splitter=TokenSplitter(max_tokens=7000, overlap_tokens=200)),
        vector_store=chunk_store,
        document_store=document_store,
        embedders={"dense": EmbedderConfig(fake_embedder)},
    )
    b2_first = await chunks_pipeline.run()
    b2_second = await chunks_pipeline.run()

    stored_onboarding = await document_store.get_document("onboarding.html")
    print(f"boundary 1: first={sorted(b1_first.processed)} then skipped={sorted(b1_second.skipped)}")
    print(f"boundary 2: first={sorted(b2_first.processed)} then skipped={sorted(b2_second.skipped)}")
    return b1_first, b1_second, b2_first, b2_second, stored_onboarding


@app.cell
def test_two_boundaries(b1_first, b1_second, b2_first, b2_second, stored_onboarding):
    _remaining = ["onboarding.html", "security.html"]
    assert sorted(b1_first.processed) == _remaining
    assert b1_second.processed == [] and sorted(b1_second.skipped) == _remaining
    assert sorted(b2_first.processed) == _remaining
    assert b2_second.processed == [] and sorted(b2_second.skipped) == _remaining
    # The document store returns a full Document, not chunks.
    assert stored_onboarding is not None
    assert stored_onboarding.source_id == "onboarding.html"
    assert len(stored_onboarding.headings) == 3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Source ID strategies

    The default `source_id_fn` uses the bare filename (`path.name`). Collisions arise when
    the same filename appears in multiple directories — the sync raises `ValueError` before
    any processing if two paths map to the same id.

    | Strategy | Expression | Use case |
    |---|---|---|
    | Filename (default) | `lambda p: p.name` | Simple flat folders |
    | Full path | `lambda p: str(p)` | Multi-dir, single machine |
    | Relative path | `lambda p: str(p.relative_to(base_dir))` | Portable multi-dir (used in this notebook) |

    `source_id_fn` is set on the `DocumentPipeline` (or on the `IngestPipeline` when you
    build the halves yourself); the sync pipelines read it from there.

    ## See Also

    - [Pipeline](pipeline.py) — `DocumentPipeline`, `VectorStorePipeline`, `ChangeSet` and `TokenSplitter`
    - [Pipeline Walkthrough](pipeline_walkthrough.py) — stage-by-stage walkthrough with intermediate output
    - [Chunking](chunking.py) — `SimpleChunker` vs `LLMChunker` in depth
    - [Sync Engine guide](../guide/sync-engine.md)
    - [API Reference: Pipeline](../api/pipeline.md)
    - [API Reference: Integrations](../api/integrations.md)
    """)
    return


if __name__ == "__main__":
    app.run()
