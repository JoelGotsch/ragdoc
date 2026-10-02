# ragdoc Change Log

Pre-1.0: breaking changes land at will and are documented here.

## Unreleased

> The entries below are grouped by the phase in which they landed, newest first. Where a
> later phase renamed or removed something an earlier phase introduced, the earlier entry
> names the **current** API and says so.

### ⚠ Breaking changes (review fixes)

Every entry is *what breaks → what to do instead*. There are no deprecation shims: the old
form raises or is gone.

**Document model**

- **Unknown constructor keywords and JSON keys raise.** Elements and `Document` use
  `extra="forbid"`; `Paragraph(html_content=…)`, `Heading(html_content=…)` and
  `RawText(innerhtml=…)` raise a `ValidationError` naming the `html` field. (These were the
  real constructor keywords before the element-model refactor on this branch; between that
  refactor and this fix they silently built an *empty* element.) → Pass `html=` with the full HTML including the outer
  tag; custom element subclasses must declare every field they are constructed with.
- **JSON written by the pre-refactor element model no longer loads** for `Heading`,
  `Paragraph`, `Table`, `DocumentList` and `RawText` (dumps carrying an `html_content` /
  `innerhtml` key next to the computed `html`). Old `Image` / `Footnote` dumps still load.
  This affects every place such JSON lives: a `DocumentStore`, `.ragdoc.json` snapshots,
  `DocumentDumpProcessor` dumps and saved `ChangeSet[Document]` files. → Regenerate them from
  the sources (for a `DocumentStore`: run `DocumentStorePipeline` again); there is no loader
  for the old format.
- **`RawText` no longer unwraps or rewrites its content.** A single non-`div` root tag is now
  wrapped verbatim: `RawText(html="<pre>x</pre>")` stores `<div><pre>x</pre></div>` and
  `.innerhtml` returns `<pre>x</pre>` (it used to store `<div>x</div>`). → Code that relied on
  the unwrapping must strip the tag itself. `content_hash` changes for documents containing
  such elements, so they are re-chunked once at Boundary 2.
- **MinerU code blocks have a new HTML shape**: `<div><pre><code>escaped text</code></pre></div>`
  with one line per source line (was the raw source text inside the `<div>`). → Nothing to
  do, but `content_hash` changes once for documents containing code blocks.
- **Oversized tables and lists split into `Table` / `DocumentList` fragments**, not `RawText`.
  → Code that looked for `RawText` fragments in splits must look for the original element
  type. Splits of sources containing an oversized table or list change (count and content), so
  their chunks are rewritten on the next sync of a changed source.

**Sync and stores**

- **`apply()` no longer writes embeddings back onto the `ChangeSet` you pass in.** → Read
  vectors from the store (or embed the chunks yourself); the changeset stays exactly the plan.
- **A store that has a `replace_source(source_id, items)` method is now updated through it**
  (structural `SourceReplaceStore` detection) instead of `delete_by_source` + `upsert`. → If a
  custom store already has a method of that name with other semantics, rename it; if it should
  be used, make it honour the contract (a failure must not leave the source looking unchanged).
- **Qdrant: a failed multi-batch upsert no longer deletes the affected sources' points.** It
  deletes only the point ids that call created. Stored sources are no longer removed as a side
  effect; a source whose write was interrupted reads as "changed" (blank hashes) instead.
  → Nothing to do; do not rely on a failed upsert emptying the source.
- **A blank (`""`) or mixed stored hash now means "always changed".** `content_hash_token`
  returns `None` for `""`; Qdrant `list_source_state()` reports a source whose points disagree
  on their hashes with blank hashes. → A custom store that stores `""` for "unknown" gets
  that source re-processed on every run; store the real hash.
- **`QdrantDocumentStore.get_document` returns `source_hash=None`** for a document stored
  without one (it used to come back as `""`). → Compare against `None`.
- **`DocumentPipeline.run_many(on_error="skip")` no longer swallows `KeyboardInterrupt`,
  `SystemExit` or other non-`Exception` `BaseException`s** — they propagate. → Catch them
  around `run_many` if you really want to continue.

**LLM layer**

- **The client's own `max_retries` is ignored for ragdoc calls.** For a real `AsyncOpenAI` /
  `AsyncAzureOpenAI` the library calls through `client.with_options(max_retries=0)`; a call
  makes `max_retries + 1` HTTP attempts (was up to `(max_retries + 1) × 3` with the SDK
  default). → Set the retry budget on the ragdoc side where it is exposed: `max_retries`
  on the settings objects, `LLMFootnoteResolver`, `make_llm_reviewer`, `OpenAIEmbedder`, and
  `openai_image_summarizer(client, max_retries=…)` for `ImageSummaryProcessor`. `LLMChunker`
  has no such parameter (fixed at 2 retries).
- **`Retry-After` is capped at `backoff_max`** (30 s by default). → Pass a larger
  `backoff_max` to `retry_llm` / `call_structured` if you call them yourself and want longer
  waits.
- **`RagdocConfig.openai_client` is typed `ChatClient | EmbeddingsClient | None`** (was
  `LLMClient | None`). → Code that reads the field and uses both capabilities must narrow it
  (or use `resolve_openai_client()` for chat).
- **`EmbeddingsEndpoint.create` declares `dimensions: int = ...`** (was `int | None = None`),
  so that a real `AsyncOpenAI` type-checks. → A custom embeddings client typed against the old
  protocol must accept an omitted `dimensions` rather than `None`.
- **`OpenAIEmbedder.embed` sends several requests** (`batch_size=256` inputs each) and raises
  `ValueError` when the endpoint returns a different number of vectors than inputs. → Lower
  or raise `batch_size` to match your provider's per-request limits.

**Extraction**

- **`KnowledgeGraphExtractor` halves only on size-related failures** (context length,
  truncated output, malformed structured output, timeout after retries). A 401, 429, 5xx or
  connection error now fails the source after the shared retry policy instead of triggering
  halving. → Handle those in `UpdateResult.errors` and re-run.
- **`KG_GLEANING_MESSAGE` takes only `{next}`** (the `{last}` placeholder is gone) and the
  gleaning call is a continuation of the first call. → Use `build_kg_gleaning_messages` rather
  than formatting the constant.
- **`build_graph_batch_model` returns `tuple[type[GraphBatchBase], type[LocalNode]]`** and
  `build_node_union` / `build_edge_union` / `KnowledgeGraphExtractor.payload_model` are typed
  `TypeForm[BaseModel]` (were `BaseModel` types / `Any`). Runtime values are unchanged; type
  checkers may need an updated annotation.
- **`EntityQuery.where` values are typed `str | int | bool`.** → Other value types were never
  matched reliably by the Qdrant filter; convert them.
- **`MentionStorePipeline` extracts at most `split_concurrency=4` splits of a source at
  once** (was unbounded), and a failing split cancels the source's other extractions. → Raise
  `split_concurrency` (or pass a shared `asyncio.Semaphore`) for more parallelism.

**Parsing, processing, chunking, utilities**

- **`ragdoc.utils.fan_out` is removed.** → `ragdoc.utils.run_ordered(factories, concurrency)`
  with zero-argument coroutine factories: `run_ordered([partial(work, x) for x in xs], 4)`. It
  returns the results in input order and cancels the siblings when one fails.
- **Parsers run in a worker thread.** `HTMLParser`, `AzureJSONParser`, `AzureDIParser` and the
  MinerU extraction now run under `asyncio.to_thread`. → Custom MinerU handlers must not touch
  event-loop-bound objects; the active `configure()` context is available in the thread.
- **`LLMChunker` returns one chunk instead of none** when the LLM returns no usable summary
  (`embedding_content = prompt_content`, with a warning). → Nothing to do unless you relied on
  the empty result.
- **`ImageSummaryProcessor` no longer aborts the document on an LLM-call failure for one
  image** (400, exhausted retries, …): the image is skipped with a warning. → Watch the log
  for skipped images; programming errors and cancellation still propagate.
- **Element equality is field-based** (same concrete type and equal fields). It previously
  also compared private cache state, so reading `.text` could make two identical elements
  unequal. → Nothing to do.

**Earlier on this branch — breaking relative to `origin/main`, not spelled out in the phase
sections below** (some appear only in the 0.1.0 notes, which were also written on this
branch)

- **`FootnoteProcessor(only_orphaned=…)` defaults to `True`** (was `False`): footnotes that
  already have a reference are not re-resolved. → Pass `only_orphaned=False` for the old
  behaviour.
- **Boundary 2 no longer runs processors.** Re-chunking a stored Document only splits and
  chunks. → Run processors at Boundary 1 (`DocumentStorePipeline(IngestPipeline(processors=…))`).
- **`VectorStorePipeline.run()` is no longer `apply(plan(...))`**: it embeds and writes per
  source, so an embedding failure no longer leaves the whole store untouched. → Use
  `apply(await plan(...))` when you need whole-corpus embed-first.
- **`DocumentPipeline`'s stage arguments are keyword-only** and its first two positional
  parameters are now `ingest` and `chunk`: `DocumentPipeline(my_parser)` binds the parser to
  `ingest`. → Pass `parser=`, `processors=`, `splitter=`, … by keyword.
- **The sync pipelines do not check the class of the pipeline object.**
  `DocumentStorePipeline(document_pipeline, store)` (the old call shape) still constructs and
  then fails every source at run time; `VectorStorePipeline`'s fourth positional parameter is
  now `hash_fn`, not `document_store`. → Pass an `IngestPipeline` (`pipeline.ingest`) to
  `DocumentStorePipeline`, use `VectorStorePipeline.from_document_store(...)` for Boundary 2,
  and pass arguments by keyword.
- **`ChunkPipeline(metadata_type=…)` can newly raise**: `filename` is now among the required
  keys, and `Required[...]` keys are really checked. → Make sure `metadata["filename"]` and
  your required keys are set before chunking (they are when documents come from `load()` or
  `IngestPipeline`).
- **An embedder returning a different number of vectors than chunks raises `ValueError`** (it
  used to truncate silently). → Fix the embedder.
- **`LLMHeadingResolver` leaves an existing title alone.** On a document whose title is
  already set (HTML with `<title>`, MinerU), the model's document-title judgment no longer
  overwrites it and `remove_title_from_elements` / `remove_elements_before_title` do nothing.
  → Clear `document.title` before the resolver if you want it to pick the title.
- **Parsing entry points removed or changed**: `ragdoc.parsing.from_path`,
  `ragdoc.parsing.pandoc.load_wordfile`, the modules `ragdoc.parsing.base` and
  `ragdoc.pipeline.parser` are gone → `ragdoc.parsing.load(path)` and
  `from ragdoc.parsing.parser import Parser`. `load_html` / `load_pandoc` / `load_excel` take a
  path instead of a source object. `AzureJSONFile` / `AzureAnalyzeRun` are no longer exported
  from `ragdoc` or `ragdoc.parsing` (internal; `ragdoc.parsing.azure_di`). MinerU's
  `PreProcessingMiddleware` / `PostProcessingMiddleware` are removed → subclass
  `ragdoc.parsing.mineru.base.BaseExtractionStage`.
- **`load("x.pdf")` without Azure credentials returns a `pdf_basic` document** when pymupdf
  is installed (it used to raise "Azure DI credentials not configured"). → Pass
  `parser="azure_di"` to insist on Azure DI.
- **Root exports**: `import ragdoc` no longer provides `MetadataDict`, `MetadataValue`,
  `TMetadata`, `metadata_json_schema` (→ `ragdoc.metadata`); `DocumentProcessor`,
  `ProcessingPipeline`, `HeadingLevelProcessor`, `TitleDetectionProcessor`
  (→ `ragdoc.processing`); `OutputFormat`, `Renderer`, `render_for_prompt`, `render_raw`
  (→ `ragdoc.rendering`); `split_by_headings`, `split_hierarchical` (→ `ragdoc.splitting`).
  `DocumentSource`, `ExcelSource`, `HTMLSource`, `PandocFile`, `WordFile`, `from_path` and
  `load_document` are removed outright (→ `ragdoc.parsing.load`).
- **`ragdoc.integrations.llama_index` is removed** with no replacement. → Convert `Chunk`s to
  your framework's node type yourself.
- **CLI**: the click entry function `ragdoc.cli.document_processing` is now `ragdoc.cli.cli`
  (installed as the `ragdoc` console script); it exits with status 1 when any file fails. →
  Call the `ragdoc` command.

### Added (review fixes)

- **`ragdoc.pipeline.SourceReplaceStore`** — optional store protocol
  (`replace_source(source_id, items)`). Implemented by `QdrantVectorStore`,
  `QdrantMentionStore`, `QdrantDocumentStore` and `LocalDocumentStore`.
- **`ragdoc.utils.run_ordered`** — the one bounded fan-out helper (generic over the result
  type, factory-based, cancels siblings on failure).
- **`ragdoc.llm`**: `without_sdk_retries`, `is_llm_failure`, `is_size_related_failure`.
- **New parameters**: `OpenAIEmbedder(batch_size=256)`;
  `openai_image_summarizer(request_timeout=None, max_retries=2)`;
  `LLMFootnoteResolver(request_timeout=None, max_retries=2)`;
  `make_llm_reviewer(request_timeout=None, max_retries=2)`;
  `MentionStorePipeline(split_concurrency=4)` and `MentionStorePipeline.from_document_store(split_concurrency=4)`;
  `split_at_sentences(overlap_tokens=…)`.
- **`ragdoc.splitting.token`**: `plan_html_split`, `build_fragment_html`,
  `build_fragment_element`, `collect_ref_ids`, `HtmlSplitPlan`, `HtmlSplitUnit`,
  `MIN_SLICE_TOKENS`.
- **`ragdoc.parsing.mineru`**: `build_code_html`, `extract_code_from_lines`.
- **`ragdoc.extraction.kg`**: `build_kg_gleaning_messages`, `merge_graph_batches`,
  `drop_dangling_edges`, `edge_refs`, `payload_kind`, `next_local_index`, and the static base
  models `GraphBatchBase` / `LocalNode`. `ragdoc.extraction.graph_store` gains
  `entity_type_for` / `node_entity_type` / `edge_entity_type`.
- `MinerUParser` and `parse_mineru_file` accept a `str` path, like every other parser.

### Fixed (review fixes)

- **`SyncEngine.run()` is back-pressured.** It used to keep every finished source's items
  until the writer got to them; producers now hand their result to the writer through a
  one-slot queue while holding their concurrency slot, so at most `concurrency + 2` sources'
  items are in memory, independent of corpus size. `run()` and `plan()` close the stream
  deterministically, so an abort cancels and awaits the running producers before returning.
- **Updates no longer pass through an empty window where the store allows it.** With a
  `SourceReplaceStore` the new items are written before the stale ones are removed, so an
  update that fails while writing keeps the previously stored items. Stores without it keep
  delete-then-upsert: a failed upsert leaves the source empty and the next run re-processes
  it as new (the docstrings previously claimed the source was left "unapplied").
- **`apply()` writes source by source** through the same path as `run()` (orphans in
  `to_delete` first), instead of all stale-deletes followed by all upserts, and **no longer
  mutates the `ChangeSet`**: it embeds copies, so a saved-then-loaded plan applies exactly
  like the in-memory one and vectors from one `apply()` cannot leak into a later one.
- **Qdrant multi-request writes are interruption-safe.** Points are written with blank
  `source_hash` / `content_hash` and the real hashes are stamped only after every batch has
  landed, so a cancelled or crashed write never reads as "unchanged". When a batch of a
  multi-batch write fails or is cancelled, the points that call created are deleted by id
  (shielded on cancellation); points that existed before are never deleted, and one that was
  already overwritten is not restored. There is no cleanup for a single-request write or for
  a failure in the stale-delete / stamp step of a replace: those leave the new points (and,
  before the stale delete, the old ones too) in place with blank hashes. After a hard crash the partial points remain, with blank
  hashes, until the next sync replaces them. A write that fits one request is one upsert.
- **Qdrant `create()` is idempotent on the in-process client.** A second
  `QdrantVectorStore.create()` (and the document / mention / entity / graph equivalents)
  against `AsyncQdrantClient(":memory:")` or `path=` raised `ValueError: Collection … already
  exists`; existence is now checked first. An existing collection is used as is.
- **`QdrantDocumentStore.get_document` works again**: it strips the index keys the store
  writes next to the Document dump (strict `Document` validation rejected them).
- **`LocalDocumentStore` writes atomically** (temp file, fsync, `os.replace`), so a crash
  never leaves a truncated document, and **one unreadable stored file no longer aborts
  `list_source_state()`**: it is logged and reported as "always changed" — never treated as
  an orphan, overwritten by a Boundary-1 re-ingest, and reported in `UpdateResult.errors` for
  that one source at Boundary 2.
- **Soup cache**: it is now keyed on the html string's value, which makes it effective for
  `Footnote` (whose html is rebuilt per access); `Image` no longer caches a parsed copy of its
  base64 payload and answers `text` / `inline_refs` without building its html.
- **Oversized tables and lists** are split on row / item boundaries into well-formed
  fragments (caption, `<colgroup>` and header repeated; `<ol>` numbering continued; rows in
  `<tbody>` handled; bare text between child tags no longer dropped). The sentence and
  token-slice tiers now verify the rendered chunk against `max_tokens`.
- **No more "coroutine … was never awaited"** when a fan-out is interrupted:
  `DocumentSummarizerProcessor`, `ImageSummaryProcessor`, `LLMHeadingResolver`,
  `FootnoteProcessor` and `MentionStorePipeline` create each coroutine only when it is about
  to run, and a failure cancels and awaits the in-flight siblings.
- **KG extraction**: an edge referencing an unknown node id is dropped with a warning instead
  of failing the whole source; gleaning sends the first answer back to the model and
  de-duplicates what it repeats.
- **`LLMHeadingResolver`** on a document that already has a title: heading levels are still
  resolved; the document-title judgment is ignored and the two title-removal options are
  no-ops (documented and pinned by a test).
- **Linting on current ruff**: tracked non-script files lost their executable bit, and the
  code passes `ruff check` with ruff 0.16 (the dev dependency is unpinned).

### Not changed, now documented

- A source that yields **no items** stores no change token and is therefore re-produced on
  every run.
- A processor-filtered document deletes its stored chunks in direct-mode
  `VectorStorePipeline`, while `DocumentStorePipeline` keeps the stored Document.
- `DocumentPipeline.stream()` is not back-pressured and stops at the first failing file.

### Fixed (review remediation — correctness)

- **Default chunk ids are valid vector-store point ids.** `mint_chunk_id` now emits a
  deterministic, canonical **UUID string** derived from
  `(source_id, split_sequence, chunk_ordinal, content_hash)` (previously a raw hex digest,
  which Qdrant rejects as a point id). The default pipeline can upsert into
  `QdrantVectorStore` without a custom `chunk_id_fn`.
- **`MentionStorePipeline` on multi-split sources**: mention ordinals now run per source
  across splits, and knowledge-graph edge references are rewritten to the finalized mention
  ids (new `rewrite_edge_refs`). Previously a source that was split could silently overwrite
  mentions and lose edges.
- **Qdrant batched upsert**: a batch failing part-way can no longer leave a partially
  written source looking unchanged. (First implemented as a source-wide rollback that deleted
  the affected sources' points; replaced by the blank-then-stamp write and id-scoped rollback
  described under *Fixed (review fixes)* above, which never removes previously stored points.)
- **Unreadable files no longer abort a sync.** Hashing errors on source files are captured
  per path (`SyncPlanInput.pre_failed`) and reported in `UpdateResult.errors`; a file that
  could not be read is never treated as an orphan. (The same now holds for an unreadable
  *stored* document in `LocalDocumentStore` — see above.)
- **`LocalDocumentStore`**: the filename is the authoritative `source_id`. A hand-edited
  `source_id` inside the stored JSON is re-stamped on read (with a warning) and can no
  longer orphan live chunks.
- **`DocumentStorePipeline` keeps the stored document when processing filters a source
  out** (skipped with a warning). A transient filter, or a misclassifying LLM processor, no
  longer deletes the stored copy.
- **`IngestPipeline` stamps provenance on the explicit-parser path**: a custom `parser=`
  callable now gets the same `parser` / `source_path` / `metadata["filename"]` as `load()`.
- **`Required` metadata keys are actually enforced** by `ChunkPipeline(metadata_type=…)`
  (the check was silently empty for TypedDicts declared in modules using
  `from __future__ import annotations`).
- **Element model**: `Heading.level` rejects values outside 1–6; assigning `Image.html`
  re-derives all fields and raises on a missing `<img>` tag or a non-`data:` src.
- **`LLMHeadingResolver`** resolves heading levels on documents that already have a title
  (only the title assignment respects the existing title).
- **Token splitter** no longer emits heading-only trailing splits after delegating an
  oversized section.
- **Extraction**: a malformed compact date (`YYYY-MMDD`) fails loudly instead of parsing as
  a reversed interval; gleaning merge survives non-contiguous primary ids and fails loudly
  on duplicate secondary ids; the Qdrant entity date filter matches half-bounded intervals
  (parity with `LocalEntityStore`).
- **`retry_llm`** no longer masks a non-openai transport error with an `ImportError` on a
  base install.

### Changed (review remediation — breaking)

- **`ragdoc.document.merge_documents` is renamed `concat_documents`** (it collided with the
  alignment-based `ragdoc.merging.merge_documents`). `join_documents` is unchanged.
- **`Mention.source_hash` is `str | None`** (was `str`) — the file-byte hash in direct-mode
  sync, `None` otherwise; never faked from the content hash. Mirrors `Chunk.source_hash`.
- **LLM-using extractors and summarizers resolve their client in `__init__`** and raise
  `LLMNotConfiguredError` at construction when none is available.
- **Chunk id format** (see Fixed above): existing chunks get new ids the next time their
  source is updated.

### Changed (review remediation — internals)

- One shared default-renderer / default-tokenizer resolution
  (`ragdoc.rendering.resolve_renderer`, `ragdoc.utils.resolve_tokenizer`), one
  `mint_entity_id`, shared `aiofiles` helpers for the local extraction stores.
- `MentionStorePipeline` defaults its splitter to `TokenSplitter()`; per-split extraction
  (bounded by `split_concurrency` since the review fixes above) and the two halves of the
  knowledge-graph halving fallback run concurrently; blocking pandoc render/split
  call sites are wrapped in `asyncio.to_thread`.

### Changed (documentation and notebooks)

- The sync notebooks (`pipeline.py`, `qdrant_pipeline.py`) now execute for real, offline
  (in-memory stores, a deterministic fake embedder), and assert the plan → apply behaviour
  in `test_*` cells; `quickstart.py` and `parsing.py` run against a sample file instead of
  placeholders; `document_model.py` covers `concat_documents`.
- `uv run pytest docs/notebooks/` now collects and runs the notebooks' `test_*` cells
  (it previously collected nothing), and CI runs it.
- `pipeline_walkthrough.py` uses a standard OpenAI-compatible endpoint configured from the
  environment and skips its LLM-backed cells when no API key is set.

### Added (Phase 10 — documentation)

- **Extraction/knowledge-graph docs** (the feature previously had zero docs presence):
  `docs/guide/extraction.md` (the mention → entity → graph three-layer workflow on the
  `Extractor` protocol, `MentionStorePipeline`, and the resolution pipelines),
  `docs/api/extraction.md` (full API reference for the public `ragdoc.extraction` surface),
  and `docs/notebooks/extraction.py` — a marimo notebook running the whole workflow with a
  stub extractor and fake resolution collaborators (no API key), wrapped in
  `tests/test_notebooks.py` and listed in `docs/tutorials.md`.
- **`docs/guide/sync-engine.md`** — the shared `SyncEngine[T]` architecture (plan/apply/run,
  two-hash change detection, `SourceSyncStore`, producers, error isolation, how the three
  sync pipelines compose it), replacing per-pipeline duplicated prose.
- **`docs/guide/processing.md`** — processor catalog, ordering guidance, `None`-to-drop
  semantics, pre- vs post-split placement, and LLM-processor configuration.
- **`docs/api/llm.md`** — the `ragdoc.llm` reliability layer got its own API page
  (`api/config.md` now links to it instead of embedding the reference).
- **API-reference completeness pass**: `api/pipeline.md` gains `DocumentStorePipeline`,
  `SyncEngine` + `SourceSyncStore`/`SyncSource`/`SyncPlanInput`/`SourceOutcome`/`file_hash`,
  `ChangeSet`/`SourceChange`, `DocumentStore`/`SourceState`/`LocalDocumentStore`, and the
  embedders (`Embedder`/`EmbedderConfig`/`OpenAIEmbedder` + content selectors);
  `api/integrations.md` gains `ServerSideVector`, `QdrantDocumentStore` +
  `DocumentTooLargeError`, and the mention/entity/graph Qdrant stores; `api/processing.md`
  gains the document/image summarizer surface; `api/splitting.md` gains `split_document`;
  `api/rendering.md` gains `format_metadata_value`. New nav entries for all new pages.
- CLAUDE.md final truth pass: `pdf_basic` in the parser list + `is_available()` resolution,
  the extraction stage in the pipeline diagram, the `SyncEngine` architecture in the sync
  section, the honest async statement (`to_thread` at orchestration points), and removal of
  the stale `get_source_hash` claim.
- Housekeeping: missing `Field(description=…)` added on `ChangeSet` and `ExcelConfig`
  fields; completed plans moved to `plans/done/`, stray `DESIGN_*.md` files moved to
  `designs/`, dormant plans marked `disregarded`.

> **⚠ BREAKING — the pipeline is split at the sync boundaries (Phase 9).**
> `DocumentPipeline` is now the composition of two new classes: `IngestPipeline`
> (parser + processors + `source_id_fn`; Boundary 1) and `ChunkPipeline` (splitter +
> chunker + `chunk_id_fn` + `metadata_type`; Boundary 2), exposed as `pipeline.ingest` /
> `pipeline.chunk`. A stage on the wrong side of a boundary is now **inexpressible**
> (there is no parameter for it — a misplaced keyword is a `TypeError`) instead of a
> runtime-rejected prose `ValueError`. The class of the pipeline object passed to a sync
> pipeline is not checked at runtime.

### Added (Phase 9 — pipeline split by boundary)

- **`IngestPipeline(parser=None, processors=None, source_id_fn=None)`** — parse → process →
  stamp `source_id`; `run(path) -> Document | None`. Exported from `ragdoc` and
  `ragdoc.pipeline`.
- **`ChunkPipeline(splitter=None, chunker=None, chunk_id_fn=None, metadata_type=None)`** —
  split → chunk → mint chunk ids; `run(document) -> list[Chunk]`. Exported from `ragdoc`
  and `ragdoc.pipeline`.
- **`VectorStorePipeline.from_document_store(chunk, vector_store, document_store, embedders=None, concurrency=10)`**
  — the Boundary-2 constructor. Takes a `ChunkPipeline`, so processors/parsers on the
  Boundary-2 path are unexpressible.
- **`MentionStorePipeline.from_document_store(extractor, mention_store, document_store, splitter=None, concurrency=10)`**
  — the Boundary-2 constructor for mention extraction (no ingest/parser/processors
  parameter exists).

### Changed (Phase 9)

- **`DocumentPipeline`** now composes the two sub-pipelines: construct from
  `ingest=`/`chunk=` or from the flat per-stage kwargs (which delegate into freshly built
  sub-pipelines). Mixing a sub-pipeline with its own stages' flat kwargs is a `TypeError`.
  `parse_and_process` → `pipeline.ingest.run`, `chunk_document` → `pipeline.chunk.run`.
- **`DocumentStorePipeline(ingest: IngestPipeline, document_store, …)`** — takes the
  Boundary-1 type directly (was `pipeline: DocumentPipeline`); a splitter/chunker has no
  parameter to arrive through.
- **`MentionStorePipeline(ingest: IngestPipeline, extractor, mention_store, …)`** — direct
  mode takes the Boundary-1 type (was `pipeline: DocumentPipeline`); the `document_store`
  constructor parameter moved to `from_document_store`.
- **`VectorStorePipeline.__init__` no longer accepts `document_store`** — Boundary-2 mode
  is `from_document_store` (see Added).

### Removed (Phase 9)

- The boundary-validation predicates `DocumentPipeline.has_splitter` /
  `has_custom_chunker` / `has_processors` / `has_custom_parser` and the three
  prose-rejection constructor blocks in the sync pipelines — the type split makes those
  states unexpressible.
- `DocumentPipeline.parse_and_process` and `DocumentPipeline.chunk_document` — use
  `pipeline.ingest.run(path)` and `pipeline.chunk.run(document)`.

> **⚠ MIGRATION — every stored `content_hash` and every chunk id changes once.**
> `Document.content_hash()` is now a versioned canonical-JSON hash (`ragdoc.content_hash.v1`)
> over `(title, elements)` — pure Python, pandoc-free. Its values differ from the old
> renderer-based hashes, so the **first Boundary-2 sync after upgrading re-chunks and re-embeds
> the full corpus once**. Direct-path syncs are unaffected for unchanged files (`source_hash`,
> the file-byte hash, short-circuits before any content hashing). Chunk ids move to a new
> deterministic scheme minted by the pipeline (`mint_chunk_id` over
> `(source_id, split_sequence, chunk_ordinal, content_hash)`), so all stored chunk ids change
> once too. Phases 5 and 6 (canonical hash + element-model normalization) ship in **this one
> release** so the corpus migrates a single time.

> **⚠ BREAKING INSTALL CHANGE — the dependency set is now split into extras.**
> The base `pip install ragdoc` no longer ships pandas, transformers, pymupdf, openai,
> pillow, or numpy. Install the extras you use: `llm` (openai + pillow — `LLMChunker`, LLM
> processors, `OpenAIEmbedder`), `tokenizers` (transformers — `RerankerTokenizer`), `xlsx`
> (pandas + openpyxl), `pdf` (pymupdf — the new `pdf_basic` parser), plus the existing
> `azure-di` (now also pulls pymupdf), `pdf-mineru`, `qdrant`, and `extraction` (now also
> pulls numpy). Every optional import fails loudly with the exact `pip install
> 'ragdoc[<extra>]'` command. `funcy`, `orjson`, `lxml`, and `tabulate` were unused and are
> gone entirely; `pydantic-settings` moved into the base dependencies (fixing a base-install
> crash in `ragdoc.processing`).

### Added (Phase 8 — dependency diet, PDF path, async remainder)

- **Extras split** (see the breaking-change note above): base deps are now just pydantic,
  pydantic-settings, beautifulsoup4, pypandoc-binary, tiktoken, aiofiles, markdownify, click,
  and httpx. `import ragdoc` imports none of torch/transformers/pandas/openai/PIL/pymupdf/
  numpy (enforced by a test and a new CI base-install smoke job).
- **Zero-config local PDF parsing** — `parsing/pdf_basic/` (`ragdoc[pdf]`): pymupdf text
  extraction with font-size/bold heading detection (reusing
  `compute_size_to_level_mapping`), registered for `.pdf` at priority 10 (below `azure_di`
  40; `mineru`, priority 50, is registered for `_middle.json`, not `.pdf`). Font sizes are emitted as inline CSS so downstream visual processors
  keep working.
- **Availability-aware parser resolution** — `Parser.is_available()` /
  `Parser.unavailable_reason()`: resolution skips parsers whose dependencies or credentials
  are missing and, when nothing usable matches, raises an actionable `ValueError` **at
  resolve time** (e.g. a `.pdf` on a base install explains `ragdoc[pdf]` /
  `ragdoc[azure-di]` / `ragdoc[pdf-mineru]` before any network/credential use).
- Python 3.13 support: `requires-python` cap lifted to `<3.14`; 3.13 legs added to the CI
  matrix and tox.

### Changed (Phase 8)

- **Async honesty**: blocking work is now wrapped off the event loop — splitter invocation in
  `ChunkPipeline.run` (then `DocumentPipeline.chunk_document`), chunker rendering (`SimpleChunker`/`LLMChunker`), the
  pandoc and xlsx and pdf_basic parsers, and PIL transforms in `ImageSummaryProcessor` all
  run under `asyncio.to_thread`; `LocalDocumentStore`, `LocalMentionStore`, and
  `LocalGraphStore` file I/O moved to `aiofiles`; `VectorStorePipeline.run_directory`'s glob
  is threaded. `DocumentPipeline.stream()` no longer abandons in-flight tasks when the
  consumer stops early or a task raises (cancel + gather in a `finally`).
- `GPTTokenizer` is no longer instantiated at import time in `ragdoc.splitting.token`
  (tiktoken's BPE load — network on a cold cache — now happens on first use).
- MinerU "middleware" terminology renamed: `CoreExtractionMiddleware` → `CoreExtractor`,
  `ParserMiddleware` → `ExtractionStage`, `BaseMiddleware` → `BaseExtractionStage`,
  `MinerUExtractor(middlewares=…, use_default_middlewares=…)` →
  `MinerUExtractor(stages=…, use_default_stages=…)`. The mineru package now imports without
  the `pdf-mineru` extra (LaTeX conversion is guarded lazily); the parser reports itself
  unavailable until the extra is installed.
- `ragdoc.rendering.serialize_metadata_value` renamed to `format_metadata_value` (it formats
  for display; the canonical serializer stays `ragdoc.metadata.serialize_metadata_value`).
- Cross-package underscore helpers got public homes: `ragdoc.utils.concurrency.resolve_semaphore`
  and `fan_out` (from `processing._concurrency`; `fan_out` has since been replaced by
  `run_ordered`, see *Breaking changes* above) and `ragdoc.utils.normalize_text`
  (from `_normalize_text`).

### Removed (Phase 8)

- Dead code: the unused `DocumentPrimaryElement`/`DocumentSecondaryElement` unions, the
  string-typed `Renderable` alias, the dead `fallback_rendered` renderer parameter,
  `MinerUExtractor.use_before`/`use_after` (zero callers), and the mineru-internal
  `_latex_to_text` (now a test helper).
- The evaluation/debug scripts `evaluate_footnotes.py`, `evaluate_middle_json.py`, and
  `debug_footnotes.py` moved out of the wheel (`ragdoc.utils` → repo `scripts/`).
- Renderer docs no longer claim "no I/O in rendering" — non-HTML formats fork a pandoc
  subprocess per render; call it via `asyncio.to_thread` from async code.

### Added (Phase 7 — shared LLM reliability layer)

- **`ragdoc.llm` — one module for every LLM call**: structural client protocols
  (`ChatClient` for `chat.completions.parse`, `EmbeddingsClient` for `embeddings.create`, and
  their intersection `LLMClient` — `AsyncOpenAI` / `AsyncAzureOpenAI` satisfy all three
  structurally); `resolve_openai_client()` (the single
  fallback policy: explicit → `get_config().openai_client` → `LLMNotConfiguredError`, fail-loud
  at construction); `call_structured()` (one structured-output call on the **non-beta**
  `client.chat.completions.parse`); and the generic retry engine `retry_llm()` (shared with
  `OpenAIEmbedder`). `ChatClient`/`EmbeddingsClient`/`LLMClient` are exported from the `ragdoc`
  root. The module never imports `openai` at module level (exception classes load lazily),
  ready for the Phase 8 extras split.
- **One retry policy everywhere** (transport errors only): 429 (a parseable `Retry-After`
  header replaces the computed delay, capped at `backoff_max` since the review fixes above),
  connection/timeout errors, and 5xx are retried with ±50 %-jitter
  exponential backoff (`min(backoff_max, backoff_base * 2**attempt) * uniform(0.5, 1.5)`);
  other 4xx, `pydantic.ValidationError`, and non-openai errors raise immediately. A model
  refusal (`parsed is None` or empty `choices`) raises **`LLMRefusalError`** and is never
  retried — a refusal is deterministic for a given input.
- **`OpenAIEmbedder`** gains `timeout` and `max_retries` and runs `embeddings.create` under
  `retry_llm` (previously no retry at all).
- **`LLMChunker`** gains `request_timeout`, `max_prompt_tokens`, and `tokenizer` parameters:
  when the rendered document exceeds `max_prompt_tokens`, the **LLM input** is truncated (one
  WARNING per document) while the emitted chunks keep the full text as `prompt_content`.
- **`request_timeout` settings** on `LLMHeadingResolverSettings` and
  `DocumentSummarizerSettings` (both forwarded per request); `ExtractionSettings.request_timeout`
  is now honored by *both* extractors.

### Changed (Phase 7)

- **All eight LLM call sites migrated off the deprecated `beta.chat.completions.parse`
  namespace** onto `ragdoc.llm.call_structured` (footnote resolver, heading resolver, document
  summarizer, image summarizer, `LLMChunker`, both extractors via `parse_with_retry`, and the
  entity-resolution reviewer). Per-site degrade semantics are deliberately preserved: heading →
  `[]`, image summary → skip-with-warning, footnote → `None`, reviewer → `ReviewResult()`;
  summarizer/chunker/extraction raise.
- **`RagdocConfig.openai_client` is typed** (was `Any | None`): `LLMClient | None` in this
  phase, widened to `ChatClient | EmbeddingsClient | None` by the review fixes above. Invalid
  objects fail validation when the `RagdocConfig` is built. There is no `client: Any` left in
  the library.
- **Default models bumped**: `default_llm_model` / `default_image_llm_model` are now `gpt-4.1`
  (were the two-year-old `gpt-4o-2024-08-06` snapshot). Kept within the gpt-4.x family because
  every call site pins `temperature=0.0`, which reasoning-family models reject.
- **`LLMFootnoteResolver`** moved from string-parsing (`"NONE"` / int-parse of free text) to a
  structured `_FootnoteSelection` response model; its client is resolved fail-loud at
  construction (`LLMNotConfiguredError` instead of a silent `None` that crashed at resolve
  time), and `model=None` now falls back to `get_config().default_llm_model` (was a hard-coded
  `gpt-4o-mini` default).
- **`LLMChunker` fails loudly at `__init__`** when no client is resolvable (was: silent `None`,
  `AttributeError` at chunk time); a refusal at chunk time raises `LLMRefusalError` (was
  `ValueError`).
- **`LLMHeadingResolver`** no longer retries deterministic 4xx errors (previously its manual
  loop retried everything with no backoff); it still degrades to `[]` on final failure. The
  settings-based client factory (base_url + api_key) survives as a documented layer above
  `resolve_openai_client`; `from openai import AsyncOpenAI` is now a lazy import inside it.
- **`ImageSummaryProcessor`** resolves a missing client fail-loud (`LLMNotConfiguredError`
  instead of an unchecked `None`); per-image refusals skip with a WARNING (image keeps
  `text_representation=None`).
- **`make_llm_reviewer`** (entity resolution) gains retry it never had; refusal still degrades
  to `ReviewResult()`.
- **openai dependency floor raised to `>=1.92.0`** — the first release shipping the non-beta
  `chat.completions.parse` namespace.

### Removed (Phase 7)

- The seven-protocol `_SummaryClient` stack in `processing/summary_document.py` and the
  four-protocol `_EmbeddingsClient` stack in `pipeline/embedders.py` (replaced by the shared
  `ragdoc.llm` protocols; `EmbeddingsClient` is re-exported from `ragdoc.pipeline`).
- `ragdoc.extraction._llm.resolve_client` (superseded by `ragdoc.llm.resolve_openai_client`).
- All four hand-rolled retry loops (heading resolver, document summarizer, and the extraction
  `parse_with_retry` internals — its body is now a `call_structured` call) and every
  `beta.chat.completions` reference in the library.
- `LLMFootnoteResolver`'s `"NONE"`-string protocol and int-parsing block.

### Added (Phase 4 — first-class `Extractor` stage)

- **`ragdoc.extraction.extractor` — the `Extractor[PayloadT]` protocol** (Decision D3-A):
  `async extract(document) -> list[Mention[PayloadT]]` is the typed extraction channel.
  Extraction results are **returned**, never written to `document.metadata`;
  `MentionStorePipeline` consumes them directly (the serialize→`model_validate` round trip
  through metadata is gone) and rejects non-conforming extractors with `TypeError` at
  construction. `payload_model` remains a documented attribute on the concrete classes (store
  queries, `ChangeSet[Mention[...]]` parametrization), deliberately outside the protocol.
- **`as_processor(extractor, metadata_key="mentions", overwrite=False)`** — explicit, opt-in
  adapter wrapping an `Extractor` into a `DocumentProcessor` that dumps serialized mentions into
  `document.metadata` (the one legitimate document-dump use case, e.g. a DocumentStore payload).
  Carries the old idempotency guard; with `overwrite=True` it strips the stale key *before*
  extracting so a mention's metadata locator never embeds a previous mention list.
- **`GraphSchema.patterns` are now enforced** (previously validated but never consumed), in two
  layers: `render_patterns_prompt(schema)` appends the legal `(source)-[edge]->(target)` triples
  to every KG system prompt, and each extracted edge's `(source_kind, edge_kind, target_kind)`
  is validated against `allowed_pattern_kinds(schema)`. Violations follow
  `ExtractionSettings.on_pattern_violation`: `"drop"` (default — edge withheld, counted, one
  WARNING per document) or `"error"` (`ValueError` naming the illegal triple). New schema rule:
  every edge type must appear in at least one pattern (dead configuration fails at declaration
  time). New helpers exported from `ragdoc.extraction`: `kind_of`, `allowed_pattern_kinds`,
  `render_patterns_prompt`.
- **`ExtractionSettings` new fields** (all KG-facing knobs now reachable from settings):
  `gleaning` (was an ad-hoc `EXTRACTION_GLEANING` env read), `max_union_size` (was env
  `KG_MAX_UNION_SIZE`; now `EXTRACTION_MAX_UNION_SIZE` via the prefix), `halving_max_depth` /
  `halving_min_chars` (were hard-coded method defaults), `on_pattern_violation`.

### Changed (Phase 4)

- **`StructuredExtractionProcessor` → `StructuredExtractor`** (module
  `ragdoc.extraction.processor` → `ragdoc.extraction.structured`) and
  **`KnowledgeGraphProcessor` → `KnowledgeGraphExtractor`** (`kg_processor` → `kg`). Both
  classes no longer subclass `DocumentProcessor`: `process()`, `metadata_key`, `overwrite`, and
  the `document.metadata` writes are gone — `extract()` is the only entry point. Both are now
  exported from `ragdoc.extraction` (alongside `Extractor`, `as_processor`,
  `ExtractionSettings`).
- **`ExtractionSettings.system_prompt` and `request_timeout` are honored by both extractors**
  (previously each honored a different one). The settings validator no longer injects the
  generic built-in prompt when unset — `system_prompt` stays `None` and each extractor resolves
  its own default (`EXTRACTION_SYSTEM_PROMPT` / `KG_EXTRACTION_SYSTEM_PROMPT`), so a user-set
  prompt finally reaches the KG extractor.
- `GraphSchema` validation is environment-independent: the union-size check moved from the model
  validator (which read `KG_MAX_UNION_SIZE` from the environment) to
  `KnowledgeGraphExtractor.__init__` against `settings.max_union_size`.
- The duplicated client/model/renderer/tokenizer fallbacks and the structured-output retry loop
  now live once in `ragdoc.extraction._llm` (`resolve_*`, `parse_with_retry`, `build_messages`);
  `build_extraction_messages` / `build_kg_messages` remain as thin wrappers documenting the
  per-extractor defaults.

### Removed (Phase 4)

- `KG_MAX_UNION_SIZE` environment variable (use `EXTRACTION_MAX_UNION_SIZE` /
  `ExtractionSettings.max_union_size`), the internal gleaning env read, and the extractors'
  `metadata_key` / `overwrite` constructor parameters (relocated to `as_processor`).

### Changed (Phase 5 — canonical `content_hash`)

- **`Document.content_hash()` is canonical-JSON and pandoc-free** (Decision D4-A):
  `sha256("ragdoc.content_hash.v1" + canonical JSON of (title, elements))` with per-element-type
  payload builders that **fail loudly** (`TypeError`) on unknown element types. Inline
  `<ref id=...>` uuids are normalized to document-order ordinals (dangling refs →
  `"unresolved"`), so a re-parse of an unchanged file hashes identically. Excluded, as before
  but now documented: document/element `metadata` (metadata-only edits do **not** re-chunk),
  provenance fields (`id`, `source_path`, `source_id`, `source_hash`, `parser`,
  `parser_version`, `external_refs`), element `page`/`bounding_box`. Newly participating
  (more-sensitive deltas): element type (a `RawText` vs `Paragraph` with identical text now
  differ) and Image `image`/`image_type`/`width`/`height`. No caching — the hash is pure
  Python and cheap; new standalone helpers `document_content_payload`,
  `element_content_payload`, `normalize_ref_ids` in `ragdoc.document`.
- **`content_hash()`'s `renderer` parameter is removed** (breaking). Customization is
  subclass-and-override, composing over `super().content_hash()`. `ragdoc.document` no longer
  imports from `ragdoc.rendering`.
- `VectorStorePipeline.plan()` over an unchanged Boundary-2 corpus now makes **zero** pandoc
  invocations (previously one pandoc subprocess per stored document per plan).

### Changed (Phase 6 — element model normalization; ships with Phase 5, one corpus migration)

- **`BaseElement.html` is a plain stored field on `Heading`, `Paragraph`, `Table`,
  `DocumentList`, and `RawText`** (Decision D5-A), each with a normalizing `field_validator`
  (Heading: first `<h1>`–`<h6>` tag re-serialized by bs4, else text escaped + `<h1>`-wrapped;
  Paragraph/Table/DocumentList: `.strip()`; RawText: guaranteed a single outer `<div>` — a
  single root `<div>` passes through, anything else is wrapped verbatim; since the review
  fixes above the content is never unwrapped or re-serialized).
  The `html_content`/`innerhtml` dual-storage split, the `@computed_field html`
  property/abstract-setter machinery, `_fields_from_html`, the `_parse_html_input`
  before-validator, and the seven per-subclass `fset` blocks are **gone**. Construct with
  `Heading(html="<h2>…</h2>")` etc.; `Heading(innerhtml=…, level=…)` and `html_content=` no
  longer exist. `Heading.innerhtml` / `RawText.innerhtml` survive as read-only properties;
  `Heading.level` is derived (setter rebuilds html).
- **`Image` and `Footnote` keep *derived* `html`** (property + real setter): Image's html is a
  projection of six structured fields (a stored copy would duplicate base64 and desync);
  Footnote's html embeds the live `id="footnote-{id}"`. `Image(html='<img …/>')` /
  `Footnote(html='<aside …>')` still work via before-validators; the parsing helpers are public
  standalone functions `image_fields_from_html` / `footnote_fields_from_html`.
- **Soup cache**: derived accessors (`text`, `inline_refs`, `footnote_ids`, `image_ids`,
  `level`, …) share one cached BeautifulSoup per element — at most one parse per element per
  pipeline pass (previously a fresh parse per access). The cache is keyed on the html string's
  value (introduced identity-keyed; changed by the review fixes above, which also made `Image`
  bypass it). `html_tag` deliberately stays fresh-parse (it returns
  a mutable `Tag`). `BaseElement` now has `validate_assignment=True`: every field assignment
  re-validates and re-normalizes.
- **`from_html` removed; `from_markdown` is typed** — `cls(html=…)` replaces `from_html(…)`;
  `from_markdown(markdown_text, *, page=0)` lost its `**kwargs`.
- `BaseElement.inline_refs` skips refs with unknown `rel` values with a WARNING (once per
  value) instead of a blind `except Exception: pass`.
- **Serialization shape**: `model_dump()` now emits `html` as a real field for the five
  stored-html classes (no `html_content`/`innerhtml` keys) and **no** `html` key for
  Image/Footnote. **Dumps written by the old model do not load** for the five stored-html
  classes (see *Breaking changes* above — unknown keys are rejected); new dumps are the format
  and round-trip for every element type.
- **Splits own their metadata**: every splitter output `Document` copies the parent's metadata
  dict at construction (previously shared by reference and compensated downstream), and the
  `split_document` no-split path **no longer mutates the input document** — it returns a
  shallow copy carrying `split_sequence`/`split_total`. Elements remain shared across splits
  by design.
- **Chunk ids are minted by the pipeline** — `ChunkPipeline.run` (introduced here as
  `DocumentPipeline.chunk_document`, renamed in Phase 9) — over
  `(source_id, split_sequence, chunk_ordinal, content_hash)` via the new
  `ragdoc.chunking.provenance.mint_chunk_id` (override with `ChunkPipeline(chunk_id_fn=…)` or
  the delegating `DocumentPipeline(chunk_id_fn=…)`, type `ChunkIdFn`). This fixes the collision where two identical-content splits of one source
  produced one chunk id, and gives `LLMChunker` stable ids. The chunkers' `id_fn` constructor
  parameters are **removed**; standalone chunker use yields uuid4 ids.
- **`Chunk.source_hash` is `str | None` and honest** — it is the file-byte hash from the sync
  pipelines' `hash_fn` or `None`; the silent `content_hash` fallback at every stamping site is
  gone (use `content_hash` for content change detection). The shared fallback chain lives in
  `ragdoc.chunking.provenance.resolve_chunk_provenance`.
- **`concat_documents(first, second, *, metadata_policy="first"|"second"|"strict")` replaces
  `Document.__or__`** (introduced here as `ragdoc.document.merge_documents`; renamed in the
  review remediation above) with an explicit, documented field policy (title bridge heading is now
  HTML-escaped; `parser`/`parser_version` kept iff identical; metadata always a fresh dict;
  `"strict"` raises on conflicting keys). `join_documents` gained the same `metadata_policy`
  keyword. (Not to be confused with `ragdoc.merging.merge_documents`, which aligns two parses
  of the *same* source.)
- **Parser provenance is stamped centrally in `ragdoc.parsing.load()`** via the new
  `stamp_provenance(document, source, parser_name)` (fills `parser`, `source_path`,
  `metadata["filename"]` only when unset; `IngestPipeline.run` calls it too, so custom
  `parser=` callables are covered). The standalone loaders (`load_html`, `load_pandoc`,
  `load_excel`, `load_azure_json`, mineru) no longer stamp these fields themselves — call
  them through `load()` (or stamp manually) when you need file provenance.
- **Escaping fixes** (four injection sites): Heading attribute values (via bs4
  re-serialization), `Image.html` alt text, the renderer's metadata `<head>`
  (title + `<meta>` tags), and xlsx sheet-name headings.
- HTML parser: `<h7>`+ tags (a pandoc docx artifact) are clamped to level 6.

### Added

- **`ragdoc.pipeline.sync` — generic `SyncEngine[T]`** (Decision D2-A). One streaming
  plan/apply/run core now drives all three sync pipelines (`VectorStorePipeline`,
  `DocumentStorePipeline`, `MentionStorePipeline`), which became thin compositions owning only
  constructor validation, source resolution, and a per-source producer. New public names in
  `ragdoc.pipeline`: `SyncEngine`, `SourceSyncStore`, `SyncSource`, `SyncPlanInput`,
  `SourceOutcome`, `file_hash` (the file-byte SHA-256 hash, now public; `_file_hash` removed).
  Pointing the engine at a new sink is configuration: any store with
  `upsert`/`delete_by_source`/`list_source_state` plus a producer coroutine.
- Concurrent, threaded file hashing: the shared current-map builder
  (`ragdoc.pipeline.sync.build_current_map`) fans hashing out via `asyncio.to_thread` under the
  concurrency semaphore, and detects `source_id` collisions **before** any hashing.

### Changed

- **Unified sync semantics across all three pipelines** (formerly drifted per copy):
  - A Boundary-2 target `source_id` missing from the DocumentStore, or a document that vanishes
    between `list_source_state` and `get_document`, is **logged and skip-counted** everywhere
    (previously uncounted; `MentionStorePipeline.plan()` was fully silent).
  - A **filtered source** (processor returns `None`) yields an empty change in the vector
    and mention pipelines: counted `processed` and its stale stored artifacts are **deleted**.
    (`DocumentStorePipeline` is the exception since the review remediation above: it keeps
    the stored Document and counts the source as skipped.)
  - `apply()` isolates errors per source everywhere, and on the delete-then-upsert path a
    source whose stale-delete failed is recorded in `errors` and **not** upserted (previously
    the mention path could mix old and new mentions; the docstore path did no stale-delete).
    Stores implementing `SourceReplaceStore` are updated with `replace_source` instead — see
    the review fixes above.
  - Error handling is `except Exception` at every per-source site (was `except BaseException`):
    `asyncio.CancelledError`, `KeyboardInterrupt`, `SystemExit`, and custom `BaseException`
    subclasses now always propagate. `UpdateResult.errors` is
    `list[tuple[str, Exception]]` (was `BaseException`).
- **`DocumentStorePipeline.run()` is now streaming** (was literally `apply(plan(...))`):
  each source is written as soon as it is produced, so sources written before an abort stay
  written. Same end state when nothing fails. (How an update is written — `replace_source`
  or delete-then-upsert — is described under the review fixes above.)
- `UpdateResult` moved to `ragdoc.pipeline.sync` (still re-exported from `ragdoc.pipeline`).
- `MentionStorePipeline.plan()/apply()` payload type is now `ChangeSet[Mention[P]]`
  (load with `ChangeSet[Mention[P]].load(path)`).
- `ChangeSet`'s type parameter loosened from `TypeVar("T", Document, Chunk)` to
  `TypeVar("T", bound=BaseModel)` — any Pydantic model payload.

### Removed (Phase 2 — public API surface reset)

- **`DocumentPipeline.run_sync()`** — it was advertised in the 0.1.0 notes below but never
  existed in the code. The library is async-only; call `asyncio.run(pipeline.run(path))`.
- **`AutoParser`** — `DocumentPipeline` defaults to `ragdoc.parsing.load` and accepts any
  async `(Path) -> Document` callable as `parser=`.
- The legacy parsing surface: `load_document`, `DocumentSource`, the `load_file`
  single-dispatch and its source classes (`HTMLSource`/`HTMLFile`, `PandocFile`/`WordFile`,
  `ExcelSource`/`ExcelPackage`). Use `ragdoc.parsing.load(path)`.
- `ragdoc.integrations.llama_index`.
- The `ragdoc` root now exports a deliberately small surface (see `ragdoc.__all__`); helpers
  such as `metadata_json_schema` are imported from their module
  (`from ragdoc.metadata import metadata_json_schema`).

### Removed

- **`VectorStore.get_source_hash`** (protocol + `QdrantVectorStore` implementation) — superseded
  by the bulk `list_source_state`. `list_source_ids` stays (consumer-facing; the sync engine
  does not use it).
- **`ragdoc.extraction.changeset`** (`MentionChangeSet`, `MentionSourceChange`) — use
  `ChangeSet[Mention[P]]` / `SourceChange[Mention[P]]` from `ragdoc.pipeline.changeset`.

### Fixed (Phase 0 — bug fixes)

- **xlsx**: `ExcelConfig.sheet_params` was silently discarded whenever `default_params` was
  set (operator precedence). Per-sheet parameters are now merged over the defaults.
- **Merging**: markup injection into a `Heading` in `ragdoc.merging` was a silent no-op (the
  update targeted a read-only property); it now takes effect.
- **Qdrant**: all five stores skipped payload-index creation when the collection already
  existed. Indexes are now ensured on every `create()`, each tolerating "already exists".
- **HTML parser**: remote images are fetched with `httpx` (10 s timeout, redirects followed)
  instead of `requests.get` without a timeout. `httpx` is a declared dependency.
- **LLM refusals** (`parsed` / `content` is `None`) raise a clear error in `LLMChunker`, skip
  the image with a warning in `ImageSummaryProcessor`, and leave the footnote unresolved in
  `LLMFootnoteResolver`, instead of failing with `AttributeError`.
- **`LLMFootnoteResolver`**: a failed LLM call is no longer reported as a confident
  "candidate 1"; the footnote stays unresolved and a warning is logged.
- **Token splitter**: the token slice measured a combined render but cut by a separately
  rendered overhead, dropping or duplicating text at the seam. It now renders and tokenizes
  the content only.

### Removed (Phase 0)

- **`SyncFootnoteProcessor`** — a sync wrapper (forbidden in an async-only library) whose only
  reachable path crashed. Use `FootnoteProcessor()`.

### Changed (Phase 1 — licence, packaging, hygiene)

- **Licence**: relicensed from Proprietary to **Apache-2.0** (`LICENSE`, `NOTICE`, `pyproject.toml`
  metadata, README badge). The release workflow now publishes to public PyPI via trusted
  publishing (`pypi` environment) when a GitHub release is published.
- **Docs match the code**: `CLAUDE.md` and the contributor docs were rewritten from the
  current tooling (ruff, basedpyright in *standard* mode, uv); a root `CONTRIBUTING.md` was
  added; the quickstart examples use `DocumentPipeline` instead of a hand-built `Chunk(...)`.
- **Notebooks**: all notebooks pass `marimo check` and are wrapped in
  `tests/test_notebooks.py`; CI gained a `marimo check` job.
- **Linting**: ruff's `B`, `ASYNC` and `BLE` rule families are enabled (blind `except`s
  narrowed, `zip(strict=…)` added); tests use `xfail_strict` and a 300 s timeout.

## 0.1.0 — 2026-07-04

> Historical entries. Several names in this section were later renamed or removed — see
> *Unreleased* above: `run_sync()` and `AutoParser` (removed), `chunk_document` /
> `parse_and_process` (now `ChunkPipeline.run` / `IngestPipeline.run`), the
> `has_*` boundary predicates (removed), `StructuredExtractionProcessor` /
> `KnowledgeGraphProcessor` (now `StructuredExtractor` / `KnowledgeGraphExtractor`),
> `SimpleChunker(id_fn=…)` (removed; ids are minted by the pipeline), and the
> renderer-based `content_hash()` (now canonical JSON).

### Added

- **`ragdoc.extraction` — typed knowledge-graph / structured extraction** _(new, `extraction`
  extra)_. Multi-node-type + typed-relationship extraction from Documents:
  - `FuzzyDate` (EDTF-backed arbitrary-precision dates), `Entity`, `Mention`, `GraphSchema`
    (closed ontology of node/edge types + legal patterns).
  - `StructuredExtractionProcessor` and `KnowledgeGraphProcessor` (single `.parse()` over a
    built discriminated-union batch model, gleaning, and a recursive `_extract_with_halving`
    fallback on LLM failures).
  - `MentionStore`/`LocalMentionStore`, `EntityStore`/`LocalEntityStore`,
    `GraphStore`/`LocalGraphStore`, `MentionStorePipeline` (plan/apply/run, Boundary-1 and
    Boundary-2), and `EntityResolutionPipeline` / `KnowledgeGraphResolutionPipeline`.
  - Qdrant adapters (`qdrant` extra): `QdrantMentionStore` (with a `payload_type` filter),
    `QdrantEntityStore` (server-side date-range filtering), `QdrantGraphStore`.
  - Env prefixes: `EXTRACTION_*` and `KG_MAX_UNION_SIZE`.

- **`DocumentSummarizerProcessor`** — recursive map-reduce whole-document summarization into
  `document.metadata["summary"]`, with `DocumentSummarizerSettings` (env prefix
  `DOCUMENT_SUMMARIZER_`) and `DocumentSummary` result model.
- **`OpenAIEmbedder`** — concrete `Embedder` wrapping an OpenAI-compatible embeddings endpoint
  (typed via a structural client Protocol; no `Any`). Exported from `ragdoc.pipeline`.
- **`QdrantDocumentStore`** (`ragdoc.integrations.document_stores`) — Qdrant-backed
  `DocumentStore` for Boundary-1 sync, with `DocumentTooLargeError` for oversized payloads.
- **Streaming per-source `VectorStorePipeline.run()`** — the standard sync path now processes
  each source independently (parse → chunk → embed → delete-then-upsert) with flat peak memory
  and per-source atomicity, instead of `apply(plan(...))`. `plan()`/`apply()` remain the
  reviewable whole-corpus path.
- **Boundary predicates on `DocumentPipeline`** — `has_splitter`, `has_custom_chunker`,
  `has_processors`, `has_custom_parser` — used by the sync pipelines to fail loudly on
  misconfiguration.

### Changed

- **`DocumentPipeline.chunk_document()` no longer processes** — processing now runs exactly once,
  at parse time (`_process_one`) or at Boundary 1 (`DocumentStorePipeline`). Re-chunking a stored
  Document (Boundary 2 / Mode 2) splits and chunks only, fixing destructive double-processing by
  non-idempotent processors (`LLMHeadingResolver`, `FootnoteProcessor`).
- **Boundary construction guards** — `DocumentStorePipeline` rejects a splitter/non-default
  chunker; `VectorStorePipeline` (with a `document_store`) rejects processors/a custom parser.
- **`FootnoteProcessor` `only_orphaned` defaults to `True`** — re-running is now idempotent
  (never appends duplicate refs). Pass `only_orphaned=False` for the legacy behaviour.
- **`LLMHeadingResolver` short-circuits when `document.title` is already set** — prevents a second
  pass from re-detecting a different title and re-stripping elements.

- **`ragdoc.pipeline` module** — high-level facade over parse → process → split → chunk.
  - `DocumentPipeline` — linear pipeline with async `run()`, `run_many()`, `stream()`,
    and `run_sync()` entry points.  Supports configurable concurrency via
    `asyncio.Semaphore` and per-file error handling (`on_error="raise"|"skip"`).
  - `TokenSplitter` — `Splitter`-Protocol-compatible wrapper around `split_document()`.
  - `AutoParser` / `Parser` Protocol — selects the right parser from a file extension.
  - `PipelineResult` dataclass — aggregates chunks and errors from `run_many()`.

- **`Document.content_hash()`** — content-stable SHA-256 hash of a document.
  Uses `MARKDOWN + render_for_prompt` by default.  Subclass and override to
  incorporate provenance (e.g. file path) or switch renderer.

### Changed

- **`SimpleChunker` default `id_fn`** _(breaking)_ — changed from `uuid.uuid4()` to
  `document.content_hash()`.  The same document content now always produces the same
  chunk ID, making vector-store upserts idempotent.  Pass `id_fn=lambda _: str(uuid4())`
  to restore the previous behaviour.