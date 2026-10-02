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

    The `ragdoc.pipeline` module wires the stages — **parse → process → split → chunk** —
    into single calls, and adds incremental **sync** pipelines that keep a store up to date
    with a corpus.

    | Scenario | Class | What it does |
    |---|---|---|
    | A | `DocumentPipeline` | One file → `Chunk` objects |
    | D | `DocumentPipeline.run_many` / `.stream` | Many files, bounded concurrency |
    | B | `VectorStorePipeline` | Files → vector store, only changed files re-processed |
    | C | `DocumentStorePipeline` + `VectorStorePipeline.from_document_store` | Files → document store → vector store (two boundaries) |

    Everything in this notebook runs offline: a small HTML corpus in a temp directory, the
    default `SimpleChunker` (no LLM), a deterministic fake embedder, and an in-memory vector
    store. The `test_*` cells assert the behaviour shown, so the notebook doubles as a test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample corpus

    Three small HTML files written to a temp directory.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    corpus_dir = Path(tempfile.mkdtemp(prefix="ragdoc-pipeline-"))

    (corpus_dir / "handbook.html").write_text(
        "<html><body>"
        "<h1>Team Handbook</h1>"
        "<p>This handbook describes how the team works together day to day.</p>"
        "<h2>Communication</h2>"
        "<p>We prefer written, asynchronous communication. Decisions are recorded in the "
        "shared decision log so that people in other time zones can follow along.</p>"
        "<p>Meetings have an agenda and an owner. Notes are published within one day.</p>"
        "<h2>Releases</h2>"
        "<p>Releases are cut every second Tuesday. Each release has a named release manager "
        "who owns the checklist and the rollback plan.</p>"
        "<p>Hotfixes skip the train but still require a second reviewer.</p>"
        "<h2>On-call</h2>"
        "<p>The on-call rotation changes weekly. The on-call engineer triages incoming alerts "
        "and writes a short handover note at the end of the shift.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (corpus_dir / "faq.html").write_text(
        "<html><body>"
        "<h1>FAQ</h1>"
        "<h2>How do I request access?</h2>"
        "<p>Open a ticket in the access queue and name the system you need.</p>"
        "<h2>Who approves expenses?</h2>"
        "<p>Your team lead approves expenses below the limit; finance approves the rest.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (corpus_dir / "release_notes.html").write_text(
        "<html><body>"
        "<h1>Release Notes</h1>"
        "<p>Version 1.0 adds the export feature and fixes two crashes on startup.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    handbook_path = corpus_dir / "handbook.html"
    faq_path = corpus_dir / "faq.html"
    notes_path = corpus_dir / "release_notes.html"
    paths = [handbook_path, faq_path, notes_path]
    print(f"corpus: {sorted(p.name for p in paths)}")
    return Path, corpus_dir, faq_path, handbook_path, notes_path, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario A — Linear pipeline

    ### Minimal example

    `DocumentPipeline` defaults:

    | Stage | Default |
    |-------|---------|
    | Parser | `load` — picks the parser from the file name via the registry |
    | Processors | none (identity pass-through) |
    | Splitter | none — whole document → one chunk |
    | Chunker | `SimpleChunker` — one chunk per (split) document, `embedding_content = prompt_content` |
    """)
    return


@app.cell
async def _(handbook_path):
    from ragdoc.pipeline import DocumentPipeline

    pipeline = DocumentPipeline()
    chunks = await pipeline.run(handbook_path)

    print(f"Produced {len(chunks)} chunk(s)")
    print(f"source_id={chunks[0].source_id!r}  metadata={chunks[0].metadata}")
    print(chunks[0].prompt_content[:200])
    return DocumentPipeline, chunks, pipeline


@app.cell
def test_minimal_pipeline(chunks):
    # No splitter: the whole document becomes exactly one chunk.
    assert len(chunks) == 1
    assert "Team Handbook" in chunks[0].prompt_content
    # SimpleChunker: one rendering serves both purposes.
    assert chunks[0].embedding_content == chunks[0].prompt_content
    # Provenance: source_id defaults to the bare filename; the filename is also in metadata.
    assert chunks[0].source_id == "handbook.html"
    assert chunks[0].metadata["filename"] == "handbook.html"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a splitter

    To split documents into retrieval-sized chunks, pass a
    [`TokenSplitter`][ragdoc.pipeline.TokenSplitter].
    Chunk size is independent of your LLM's context window — chunks are sized
    for retrieval quality, typically 1/50th to 1/10th of the context window.
    Test different `max_tokens` values for your use case. The tiny budget below is only
    there to make the small sample document split:
    """)
    return


@app.cell
async def _(DocumentPipeline, handbook_path):
    from ragdoc.pipeline import TokenSplitter

    pipeline_with_split = DocumentPipeline(
        splitter=TokenSplitter(max_tokens=60, overlap_tokens=0),
    )
    split_chunks = await pipeline_with_split.run(handbook_path)

    print(f"Produced {len(split_chunks)} chunks")
    for _c in split_chunks:
        print(f"  split {_c.metadata['split_sequence']}/{_c.metadata['split_total']}: {_c.prompt_content[:50]!r}")
    return TokenSplitter, pipeline_with_split, split_chunks


@app.cell
def test_splitter(split_chunks):
    assert len(split_chunks) > 1
    # Every chunk records its position; all chunks of one source share source_id and content_hash.
    assert [c.metadata["split_sequence"] for c in split_chunks] == list(range(1, len(split_chunks) + 1))
    assert {c.source_id for c in split_chunks} == {"handbook.html"}
    assert len({c.content_hash for c in split_chunks}) == 1
    assert len({c.id for c in split_chunks}) == len(split_chunks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding processors

    Processors run between parsing and splitting and enrich the `Document` in place.
    """)
    return


@app.cell
async def _(DocumentPipeline, TokenSplitter, handbook_path):
    from ragdoc.processing import HeadingLevelProcessor, TitleDetectionProcessor

    pipeline_with_processors = DocumentPipeline(
        processors=[
            HeadingLevelProcessor(),
            TitleDetectionProcessor(),
        ],
        splitter=TokenSplitter(max_tokens=60, overlap_tokens=0),
    )
    processed_chunks = await pipeline_with_processors.run(handbook_path)
    print(f"Produced {len(processed_chunks)} chunks with HeadingLevelProcessor + TitleDetectionProcessor")
    return (processed_chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two halves: `IngestPipeline` and `ChunkPipeline`

    `DocumentPipeline` is a composition of two sub-pipelines, split where the sync
    boundaries are:

    - `IngestPipeline` — parser + processors + `source_id_fn` → a whole `Document`
    - `ChunkPipeline` — splitter + chunker + `chunk_id_fn` + `metadata_type` → `Chunk`s

    The flat keyword arguments used above (`DocumentPipeline(splitter=…)`) build these for
    you; they are available as `pipeline.ingest` and `pipeline.chunk`. You can also build
    them yourself and run the halves separately — Scenario C does exactly that.
    """)
    return


@app.cell
async def _(DocumentPipeline, TokenSplitter, handbook_path):
    from ragdoc.pipeline import ChunkPipeline, IngestPipeline

    ingest = IngestPipeline()
    chunker_half = ChunkPipeline(splitter=TokenSplitter(max_tokens=60, overlap_tokens=0))

    handbook_doc = await ingest.run(handbook_path)  # parse → process
    halves_chunks = await chunker_half.run(handbook_doc)  # split → chunk

    composed = DocumentPipeline(ingest=ingest, chunk=chunker_half)
    composed_chunks = await composed.run(handbook_path)
    print(f"halves: {len(halves_chunks)} chunks, composed: {len(composed_chunks)} chunks")
    return ChunkPipeline, IngestPipeline, composed_chunks, halves_chunks, handbook_doc


@app.cell
def test_halves_equal_composed(composed_chunks, halves_chunks, split_chunks):
    # Running the two halves by hand is the same as running the composed pipeline…
    assert [c.id for c in halves_chunks] == [c.id for c in composed_chunks]
    # …which is the same as the flat-kwargs form used earlier.
    assert [c.id for c in composed_chunks] == [c.id for c in split_chunks]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Custom parser

    Any **async** callable `(Path) -> Document` can be passed as `parser=`. The pipeline
    stamps provenance (`parser`, `source_path`, `metadata["filename"]`, `source_id`) on the
    returned document, so a custom parser only has to produce the elements:
    """)
    return


@app.cell
async def _(DocumentPipeline, Path, corpus_dir):
    from ragdoc.document import Document, Paragraph

    async def parse_lines(path: Path) -> Document:
        # e.g. call an API, parse a custom format, …
        lines = path.read_text(encoding="utf-8").splitlines()  # noqa: ASYNC240 — tiny file in a doc example
        return Document(elements=[Paragraph(html=f"<p>{line}</p>") for line in lines if line])

    _notes_txt = corpus_dir / "notes.txt"
    _notes_txt.write_text("first line\nsecond line\n", encoding="utf-8")

    custom_chunks = await DocumentPipeline(parser=parse_lines).run(_notes_txt)
    print(custom_chunks[0].prompt_content)
    return Document, custom_chunks


@app.cell
def test_custom_parser(custom_chunks):
    assert len(custom_chunks) == 1
    assert "first line" in custom_chunks[0].prompt_content
    assert "second line" in custom_chunks[0].prompt_content
    assert custom_chunks[0].source_id == "notes.txt"
    assert custom_chunks[0].metadata["filename"] == "notes.txt"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deterministic chunk IDs

    Chunk ids are minted by `ChunkPipeline.run` (not by the chunker) with
    `ragdoc.chunking.provenance.mint_chunk_id` over
    `(source_id, split_sequence, chunk_ordinal, content_hash)`, and emitted as a UUID
    string. The same source content therefore always produces the same ids — re-running
    is idempotent — while two splits with identical text still get distinct ids. Pass
    `chunk_id_fn=` to use your own scheme.
    """)
    return


@app.cell
async def _(handbook_path, pipeline_with_split, split_chunks):
    from ragdoc.chunking.provenance import mint_chunk_id

    rerun_chunks = await pipeline_with_split.run(handbook_path)
    _first = split_chunks[0]
    expected_first_id = mint_chunk_id(_first.source_id, 1, 0, _first.content_hash)
    print(f"first chunk id: {_first.id}")
    print(f"same on re-run: {[c.id for c in rerun_chunks] == [c.id for c in split_chunks]}")
    return expected_first_id, rerun_chunks


@app.cell
def test_chunk_ids_deterministic(expected_first_id, rerun_chunks, split_chunks):
    import uuid

    assert [c.id for c in rerun_chunks] == [c.id for c in split_chunks]
    assert split_chunks[0].id == expected_first_id
    assert str(uuid.UUID(split_chunks[0].id)) == split_chunks[0].id  # canonical UUID string
    return


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
async def _(DocumentPipeline, paths):
    pipeline_many = DocumentPipeline()
    many_result = await pipeline_many.run_many(paths, concurrency=2)
    print(f"Produced {len(many_result.chunks)} chunks from {len(paths)} files, {len(many_result.errors)} errors")
    return (many_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Error handling

    By default errors propagate immediately (`on_error="raise"`).
    Use `on_error="skip"` to collect failures in `result.errors` and continue:
    """)
    return


@app.cell
async def _(DocumentPipeline, corpus_dir, paths):
    pipeline_skip = DocumentPipeline(on_error="skip")

    _missing = corpus_dir / "does_not_exist.html"
    skip_result = await pipeline_skip.run_many([*paths, _missing])
    for _path, _exc in skip_result.errors:
        print(f"  FAILED {_path.name}: {type(_exc).__name__}")
    print(f"{len(skip_result.chunks)} chunks produced despite {len(skip_result.errors)} failure(s)")
    return pipeline_skip, skip_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming

    `stream` yields one batch of chunks per file as it completes, so you can hand each
    batch off (for example to a vector store) instead of collecting everything first.
    It is not back-pressured (finished batches wait in memory until you consume them) and it
    stops at the first failing file — for syncing into a store, use `VectorStorePipeline.run`
    (Scenario B), which isolates failures per source and keeps memory bounded:
    """)
    return


@app.cell
async def _(paths, pipeline_skip):
    stream_batches = []
    async for _batch in pipeline_skip.stream(paths, concurrency=2):
        stream_batches.append(_batch)  # e.g. await vector_store.upsert(_batch)
    print(f"stream() yielded {len(stream_batches)} batches")
    return (stream_batches,)


@app.cell
def test_fan_out(many_result, paths, skip_result, stream_batches):
    # One chunk per file (no splitter), no errors.
    assert len(many_result.chunks) == len(paths)
    assert many_result.errors == []
    assert sorted(c.source_id for c in many_result.chunks) == ["faq.html", "handbook.html", "release_notes.html"]
    # on_error="skip": the missing file is reported, the others still produce chunks.
    assert len(skip_result.chunks) == len(paths)
    assert [p.name for p, _ in skip_result.errors] == ["does_not_exist.html"]
    # stream(): one batch per file.
    assert len(stream_batches) == len(paths)
    assert sorted(b[0].source_id for b in stream_batches) == ["faq.html", "handbook.html", "release_notes.html"]
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

    ## Scenario B — Incremental sync into a vector store

    `VectorStorePipeline` wraps a `DocumentPipeline` and adds hash-based change detection,
    so a corpus can be kept in sync with a vector store incrementally — only new or changed
    files are re-processed. The vector store itself holds the state (the `source_hash` of
    every stored chunk); there is no local state file.

    It needs two collaborators, both defined by small protocols:

    - a `VectorStore` (`upsert` / `delete` / `delete_by_source` / `list_source_ids` /
      `list_source_state`). Below is a complete in-memory implementation; for Qdrant see the
      [Qdrant pipeline notebook](qdrant_pipeline.py).
    - optionally one or more `Embedder`s (`async embed(texts) -> vectors`). The fake below
      is deterministic and needs no network; production code would use `OpenAIEmbedder` or
      your own.
    """)
    return


@app.cell
def _():
    import hashlib

    from ragdoc.chunking import Chunk
    from ragdoc.pipeline import SourceState

    class InMemoryVectorStore:
        """Minimal VectorStore: a dict of chunks keyed by chunk id."""

        def __init__(self) -> None:
            self.chunks: dict[str, Chunk] = {}

        async def upsert(self, chunks: list[Chunk]) -> list[str]:
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
            return [chunk.id for chunk in chunks]

        async def delete(self, ids: list[str]) -> None:
            for chunk_id in ids:
                self.chunks.pop(chunk_id, None)

        async def delete_by_source(self, source_id: str) -> None:
            self.chunks = {cid: c for cid, c in self.chunks.items() if c.source_id != source_id}

        async def list_source_ids(self) -> set[str]:
            return {c.source_id for c in self.chunks.values()}

        async def list_source_state(self) -> dict[str, SourceState]:
            return {
                c.source_id: SourceState(source_hash=c.source_hash or "", content_hash=c.content_hash)
                for c in self.chunks.values()
            }

    class FakeEmbedder:
        """Deterministic 8-dimensional 'embedding' derived from a SHA-256 of the text."""

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[b / 255 for b in hashlib.sha256(t.encode("utf-8")).digest()[:8]] for t in texts]

    return Chunk, FakeEmbedder, InMemoryVectorStore


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `plan()` → review → `apply()`

    `plan(paths)` hashes the files, compares against the store, and chunks the new/changed
    ones. It returns a `ChangeSet` and **does not write to the store**. The `ChangeSet` is a
    Pydantic model: `to_add` / `to_update` (one `SourceChange` per source, carrying its
    chunks) and `to_delete` (source ids).
    """)
    return


@app.cell
async def _(DocumentPipeline, FakeEmbedder, InMemoryVectorStore, TokenSplitter, paths):
    from ragdoc.pipeline import EmbedderConfig, VectorStorePipeline

    vector_store = InMemoryVectorStore()
    vs_pipeline = VectorStorePipeline(
        pipeline=DocumentPipeline(splitter=TokenSplitter(max_tokens=60, overlap_tokens=0)),
        vector_store=vector_store,
        embedders={"dense": EmbedderConfig(FakeEmbedder())},
    )

    first_plan = await vs_pipeline.plan(paths)
    stored_after_plan = len(vector_store.chunks)

    print(f"to_add:    {sorted(sc.source_id for sc in first_plan.to_add)}")
    print(f"to_update: {[sc.source_id for sc in first_plan.to_update]}")
    print(f"to_delete: {first_plan.to_delete}")
    print(f"chunks in store after plan(): {stored_after_plan}")
    return EmbedderConfig, VectorStorePipeline, first_plan, stored_after_plan, vector_store, vs_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the plan is serializable, it can be written to disk, reviewed (or edited) by a
    human, loaded again and applied later. Call `load` on the concrete type —
    `ChangeSet[Chunk]` — so the items are validated as chunks:
    """)
    return


@app.cell
async def _(Chunk, corpus_dir, first_plan, vector_store, vs_pipeline):
    from ragdoc.pipeline import ChangeSet

    _plan_file = corpus_dir / "plan.json"
    first_plan.save(_plan_file)
    reloaded_plan = ChangeSet[Chunk].load(_plan_file)
    plan_round_trips = reloaded_plan.model_dump() == first_plan.model_dump()
    print(f"plan survives save()/load(): {plan_round_trips}")

    first_apply = await vs_pipeline.apply(reloaded_plan)  # embeds the plan's chunks, then writes
    stored_after_apply = dict(vector_store.chunks)
    print(f"apply(): processed={sorted(first_apply.processed)} errors={first_apply.errors}")
    print(f"chunks in store after apply(): {len(stored_after_apply)}")
    return ChangeSet, first_apply, plan_round_trips, stored_after_apply


@app.cell
def test_plan_then_apply_adds(first_apply, first_plan, plan_round_trips, stored_after_apply, stored_after_plan):
    # plan(): every file is new, nothing is written.
    assert sorted(sc.source_id for sc in first_plan.to_add) == ["faq.html", "handbook.html", "release_notes.html"]
    assert first_plan.to_update == []
    assert first_plan.to_delete == []
    assert stored_after_plan == 0
    # The ChangeSet survives a save()/load() round trip.
    assert plan_round_trips
    # apply(): all three sources written, every chunk embedded and stamped with the file hash.
    assert sorted(first_apply.processed) == ["faq.html", "handbook.html", "release_notes.html"]
    assert first_apply.errors == []
    assert len(stored_after_apply) == sum(len(sc.items) for sc in first_plan.to_add)
    assert all(len(c.named_embeddings["dense"]) == 8 for c in stored_after_apply.values())
    assert all(c.source_hash for c in stored_after_apply.values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unchanged files are skipped

    `run(paths)` is the one-call form: it detects changes and writes them. With nothing
    changed since the `apply()` above, every source is skipped — no parsing, no chunking, no
    embedding. (`run()` reaches the same end state as `apply(plan(...))`; it differs in
    scheduling, writing each source as soon as it is ready instead of preparing the whole
    change set first. It is back-pressured: at most `concurrency + 2` sources' chunks are in
    memory at a time, however large the corpus is.)
    """)
    return


@app.cell
async def _(first_apply, paths, vs_pipeline):
    _ = first_apply  # run after the apply() above
    unchanged_plan = await vs_pipeline.plan(paths)
    unchanged_run = await vs_pipeline.run(paths)
    print(f"plan(): to_add={len(unchanged_plan.to_add)} to_update={len(unchanged_plan.to_update)}")
    print(f"run():  processed={unchanged_run.processed} skipped={sorted(unchanged_run.skipped)}")
    return unchanged_plan, unchanged_run


@app.cell
def test_unchanged_is_skipped(unchanged_plan, unchanged_run):
    assert unchanged_plan.to_add == []
    assert unchanged_plan.to_update == []
    assert unchanged_run.processed == []
    assert sorted(unchanged_run.skipped) == ["faq.html", "handbook.html", "release_notes.html"]
    assert unchanged_run.errors == []
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A changed file is updated

    Editing a file changes its byte hash. The next plan lists it under `to_update`; applying
    it deletes that source's old chunks and upserts the new ones. The other files are skipped.
    """)
    return


@app.cell
async def _(faq_path, paths, unchanged_run, vector_store, vs_pipeline):
    _ = unchanged_run  # run after the cells above
    faq_path.write_text(
        "<html><body><h1>FAQ</h1><h2>How do I request access?</h2>"
        "<p>Use the self-service portal; access is granted automatically.</p></body></html>",
        encoding="utf-8",
    )

    changed_plan = await vs_pipeline.plan(paths)
    changed_run = await vs_pipeline.run(paths)
    faq_chunks_after_update = [c for c in vector_store.chunks.values() if c.source_id == "faq.html"]
    print(f"plan(): to_update={[sc.source_id for sc in changed_plan.to_update]}")
    print(f"run():  processed={changed_run.processed} skipped={sorted(changed_run.skipped)}")
    return changed_plan, changed_run, faq_chunks_after_update


@app.cell
def test_changed_is_updated(changed_plan, changed_run, faq_chunks_after_update):
    assert changed_plan.to_add == []
    assert [sc.source_id for sc in changed_plan.to_update] == ["faq.html"]
    assert changed_run.processed == ["faq.html"]
    assert sorted(changed_run.skipped) == ["handbook.html", "release_notes.html"]
    # The store now holds only the new content for faq.html.
    assert faq_chunks_after_update
    assert any("self-service portal" in c.prompt_content for c in faq_chunks_after_update)
    assert not any("Open a ticket" in c.prompt_content for c in faq_chunks_after_update)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Removed files: `delete_orphans` is opt-in

    A source that is in the store but **not** in the list you pass is an *orphan*. By default
    (`delete_orphans=False`) orphans are left alone — passing a partial list must not wipe
    the rest of your index. Pass `delete_orphans=True` only when the list is your **complete**
    corpus.
    """)
    return


@app.cell
async def _(changed_run, faq_path, handbook_path, vector_store, vs_pipeline):
    _ = changed_run  # run after the cells above
    _remaining = [handbook_path, faq_path]  # release_notes.html is no longer part of the corpus

    orphan_kept_run = await vs_pipeline.run(_remaining)
    sources_after_default = await vector_store.list_source_ids()

    orphan_plan = await vs_pipeline.plan(_remaining, delete_orphans=True)
    orphan_deleted_run = await vs_pipeline.run(_remaining, delete_orphans=True)
    sources_after_opt_in = await vector_store.list_source_ids()

    print(f"default:              deleted={orphan_kept_run.deleted}  store={sorted(sources_after_default)}")
    print(f"delete_orphans=True:  deleted={orphan_deleted_run.deleted}  store={sorted(sources_after_opt_in)}")
    return orphan_deleted_run, orphan_kept_run, orphan_plan, sources_after_default, sources_after_opt_in


@app.cell
def test_orphans_are_opt_in(
    orphan_deleted_run,
    orphan_kept_run,
    orphan_plan,
    sources_after_default,
    sources_after_opt_in,
):
    # Default: the missing file's chunks stay in the store.
    assert orphan_kept_run.deleted == []
    assert sources_after_default == {"faq.html", "handbook.html", "release_notes.html"}
    # Opt-in: the plan lists the orphan, and running removes it.
    assert orphan_plan.to_delete == ["release_notes.html"]
    assert orphan_deleted_run.deleted == ["release_notes.html"]
    assert sources_after_opt_in == {"faq.html", "handbook.html"}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two hashes, one id

    | Value | What it covers | Purpose |
    |------|---------------|---------|
    | `chunk.source_hash` | Raw file bytes (SHA-256) | Change detection on the direct path, before parsing |
    | `chunk.content_hash` | `Document.content_hash()` — canonical JSON of title + elements | Change detection when chunking from a document store (Scenario C) |
    | `chunk.id` | `mint_chunk_id(source_id, split_sequence, chunk_ordinal, content_hash)` | Stable point id in the vector store |

    All three are first-class `Chunk` fields; none of them is written into `chunk.metadata`.

    ### What a sync does per source

    | Case | What happens |
    |------|-------------|
    | **New file** | Parsed, chunked, embedded, upserted. |
    | **Unchanged file** (byte hash matches the store) | Skipped — listed in `UpdateResult.skipped`. |
    | **Modified file** (byte hash differs) | Re-processed and replaced. A store with `replace_source` (e.g. `QdrantVectorStore`) gets the new chunks before the stale ones are removed; any other store — like the in-memory one here — gets delete-then-upsert. |
    | **Removed file** (in the store, not in the list) | Left in place by default. Deleted only with `delete_orphans=True`. |
    | **Failing file** | Reported in `UpdateResult.errors`; the other sources are still processed. A failed update keeps the old chunks in a `replace_source` store; with delete-then-upsert a failed upsert leaves the source empty until the next run re-processes it. |

    ---

    ## Scenario C — Two boundaries: files → documents → chunks

    Parsing and LLM processing are usually the expensive part; chunking is cheap and is what
    you tune. Splitting the sync in two lets you re-chunk without re-parsing:

    1. **Boundary 1** — `DocumentStorePipeline(ingest=IngestPipeline(...), document_store=...)`
       parses and processes files into a `DocumentStore` (change detection: file-byte
       `source_hash`).
    2. **Boundary 2** — `VectorStorePipeline.from_document_store(chunk=ChunkPipeline(...), ...)`
       reads those documents back and chunks them into the vector store (change detection:
       `content_hash`).

    The types enforce the split: an `IngestPipeline` has no splitter/chunker parameter and a
    `ChunkPipeline` has no parser/processors parameter. `LocalDocumentStore` keeps one JSON
    file per source in a directory.
    """)
    return


@app.cell
async def _(
    ChunkPipeline,
    EmbedderConfig,
    FakeEmbedder,
    InMemoryVectorStore,
    IngestPipeline,
    TokenSplitter,
    VectorStorePipeline,
    corpus_dir,
    faq_path,
    handbook_path,
    orphan_deleted_run,
):
    from ragdoc.pipeline import DocumentStorePipeline, LocalDocumentStore

    _ = orphan_deleted_run  # run after Scenario B (which edits the corpus files)
    two_step_paths = [handbook_path, faq_path]

    # Boundary 1: files → DocumentStore
    document_store = LocalDocumentStore(corpus_dir / "documents")
    doc_store_pipeline = DocumentStorePipeline(ingest=IngestPipeline(), document_store=document_store)
    b1_first = await doc_store_pipeline.run(two_step_paths)
    b1_second = await doc_store_pipeline.run(two_step_paths)

    # Boundary 2: DocumentStore → VectorStore
    b2_store = InMemoryVectorStore()
    b2_pipeline = VectorStorePipeline.from_document_store(
        chunk=ChunkPipeline(splitter=TokenSplitter(max_tokens=60, overlap_tokens=0)),
        vector_store=b2_store,
        document_store=document_store,
        embedders={"dense": EmbedderConfig(FakeEmbedder())},
    )
    b2_plan = await b2_pipeline.plan()  # None → every document in the store
    b2_first = await b2_pipeline.apply(b2_plan)
    b2_second = await b2_pipeline.run()

    print(f"boundary 1: first={sorted(b1_first.processed)} second skipped={sorted(b1_second.skipped)}")
    print(f"boundary 2: first={sorted(b2_first.processed)} second skipped={sorted(b2_second.skipped)}")
    print(f"vector store: {len(b2_store.chunks)} chunks")
    return (
        b1_first,
        b1_second,
        b2_first,
        b2_pipeline,
        b2_plan,
        b2_second,
        b2_store,
        doc_store_pipeline,
        document_store,
        two_step_paths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now edit one source file and run both boundaries again. Boundary 1 re-parses only the
    edited file; Boundary 2 re-chunks only the document whose `content_hash` changed.
    """)
    return


@app.cell
async def _(b2_pipeline, b2_second, b2_store, doc_store_pipeline, document_store, faq_path, two_step_paths):
    _ = b2_second  # run after the cell above
    faq_path.write_text(
        "<html><body><h1>FAQ</h1><h2>Who approves expenses?</h2>"
        "<p>All expenses are approved by finance.</p></body></html>",
        encoding="utf-8",
    )

    b1_changed = await doc_store_pipeline.run(two_step_paths)
    b2_changed_plan = await b2_pipeline.plan()
    b2_changed = await b2_pipeline.apply(b2_changed_plan)

    stored_faq = await document_store.get_document("faq.html")
    b2_faq_chunks = [c for c in b2_store.chunks.values() if c.source_id == "faq.html"]
    print(f"boundary 1: processed={b1_changed.processed} skipped={b1_changed.skipped}")
    print(f"boundary 2: to_update={[sc.source_id for sc in b2_changed_plan.to_update]} processed={b2_changed.processed}")
    return b1_changed, b2_changed, b2_changed_plan, b2_faq_chunks, stored_faq


@app.cell
def test_two_boundary_sync(
    b1_changed,
    b1_first,
    b1_second,
    b2_changed,
    b2_changed_plan,
    b2_faq_chunks,
    b2_first,
    b2_plan,
    b2_second,
    stored_faq,
):
    # Boundary 1: add, then unchanged → skipped.
    assert sorted(b1_first.processed) == ["faq.html", "handbook.html"]
    assert b1_second.processed == []
    assert sorted(b1_second.skipped) == ["faq.html", "handbook.html"]
    # Boundary 2: add, then unchanged → skipped.
    assert sorted(sc.source_id for sc in b2_plan.to_add) == ["faq.html", "handbook.html"]
    assert sorted(b2_first.processed) == ["faq.html", "handbook.html"]
    assert b2_second.processed == []
    assert sorted(b2_second.skipped) == ["faq.html", "handbook.html"]
    # After editing faq.html: only that source moves through each boundary.
    assert b1_changed.processed == ["faq.html"]
    assert b1_changed.skipped == ["handbook.html"]
    assert b2_changed_plan.to_add == []
    assert [sc.source_id for sc in b2_changed_plan.to_update] == ["faq.html"]
    assert b2_changed.processed == ["faq.html"]
    # The chunks carry the stored document's content hash and the new text.
    assert {c.content_hash for c in b2_faq_chunks} == {stored_faq.content_hash()}
    assert any("approved by finance" in c.prompt_content for c in b2_faq_chunks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [API Reference: Pipeline][ragdoc.pipeline]
    - [Sync Engine guide](../guide/sync-engine.md) — the shared engine behind every sync pipeline
    - [Qdrant Pipeline](qdrant_pipeline.py) — the same sync against a Qdrant collection
    - [Pipeline Walkthrough](pipeline_walkthrough.py) — stage-by-stage walkthrough with intermediate output
    - [Chunking notebook](chunking.py) — `SimpleChunker`, `LLMChunker`
    - [Splitting notebook](splitting.py) — lower-level splitting API
    """)
    return


if __name__ == "__main__":
    app.run()
