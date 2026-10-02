# Sync Engine

Every sync pipeline in ragdoc — [`VectorStorePipeline`](../api/pipeline.md#vectorstorepipeline),
[`DocumentStorePipeline`](../api/pipeline.md#documentstorepipeline), and the extraction layer's
[`MentionStorePipeline`](../api/extraction.md#mentionstorepipeline) — is a thin composition over
one generic streaming core: [`SyncEngine[T]`](../api/pipeline.md#syncengine). This page explains
that shared machinery once; the per-pipeline pages only describe what differs (how sources are
resolved and what each source produces).

## The shape: plan / apply / run

Every sync pipeline exposes the same three methods:

| Method | What it does | Store writes |
|---|---|---|
| `plan(...)` | Compute a reviewable [`ChangeSet`](../api/pipeline.md#changeset) | none |
| `apply(changeset)` | Write a (possibly edited) ChangeSet | the whole change set, prepared first, then source by source |
| `run(...)` | Detect changes and write them in one call | per source, as produced |

`run()` is a separate streaming path, not a literal `apply(plan(...))`: when every source
succeeds it reaches the **same end state** — same change detection, same final store
contents — but it is scheduled differently.

- **`run`** writes each changed source as soon as it has been produced, so sources written
  before a later failure stay written. It is **back-pressured**: a producer keeps its
  concurrency slot until the writer has taken its result, so at most `concurrency + 2`
  sources' items are held in memory at any time, however large the corpus is. (One
  lightweight task per source is still created up front.)
- **`apply`** runs the `pre_write` hook once over the whole change set before it writes
  anything; for `VectorStorePipeline` that hook is embedding, so `apply` embeds all chunks of
  the change set first. If that fails, the store is untouched and every changed source is
  reported in `errors`. Then it deletes the orphans in `to_delete` and writes the sources one
  by one. The whole change set is in memory by construction.

`apply` never mutates the `ChangeSet` it is given — embedding runs on copies — so a plan still
equals its saved copy afterwards, and a saved-then-loaded plan applies exactly like the
in-memory one.

Neither path is a transaction across the corpus. After a sync, look at
`UpdateResult.errors`: a source listed there was not (fully) written, and what the store holds
for it is described under [Write discipline](#write-discipline) below. One difference between
the paths: `run()` reports every failure there, whereas on the reviewable path a source that
fails during hashing or production is left out of the `ChangeSet` by `plan()` and only logged
(`plan(): N source(s) failed …`) — `apply()`'s `errors` covers embedding and write failures.
If embedding fails in `apply()`, nothing is written and the orphans in `to_delete` are not
deleted either.

### Human-in-the-loop review

`plan()` never touches the target store, and its `ChangeSet` is serializable:

```python
changeset = await pipeline.plan(paths)
changeset.save(Path("review/changes.json"))
# ... inspect / edit the JSON ...
approved = ChangeSet[Chunk].load(Path("review/changes.json"))   # load on the concrete type
result = await pipeline.apply(approved)
```

A `ChangeSet[T]` carries `to_add` / `to_update` (lists of
[`SourceChange[T]`](../api/pipeline.md#sourcechange) — one entry per changed source with its
new items) and `to_delete` (orphaned source_ids).

## Change detection: two hashes

Change detection is per **source** (not per item), keyed on `source_id` and compared via one of
two first-class change tokens:

- **`source_hash`** — SHA-256 of the raw source-file bytes
  ([`file_hash`](../api/pipeline.md#file_hash)). Used on the **direct path** (Boundary 1):
  unchanged files are skipped before any parsing or LLM work.
- **`content_hash`** — `Document.content_hash()`, a canonical-JSON hash over
  `(title, elements)` — pure Python, dependency-stable. Used on the **Boundary-2 path**
  (reading from a [`DocumentStore`](../api/pipeline.md#documentstore)): editing a stored
  Document's title or elements changes its hash, which triggers re-chunking/re-extraction.
  A metadata-only edit does **not** change the hash (metadata, `page`, `bounding_box` and
  provenance fields are excluded), so the stored chunks keep their old metadata until the
  content changes.

A **missing or blank stored hash means "always changed"** on either path (`None` or `""`).
Stores use that deliberately: a source whose stored items disagree on their hashes, or whose
write was interrupted, is reported with blank hashes so it is re-processed.

Both are first-class fields on `Chunk` and `Mention`. A `Document` carries `source_hash` as a
field and computes `content_hash()` on demand. Neither is ever stored in `metadata`.

## The engine, the stores, the producers

[`SyncEngine[T]`](../api/pipeline.md#syncengine) owns the invariant machinery:

- **Change detection** — compare each source's current token against the stored state
  (`list_source_state()`), skip matches.
- **Concurrency** — sources are produced under a semaphore (`concurrency`, an `int` or a
  shared `asyncio.Semaphore`) and streamed as completed. If you share a semaphore, nothing on
  the write path (the embedder, the store) may acquire that same semaphore: in `run` the
  producers hold their slots while they wait for the writer.
- **Error isolation** — resolution, production and write failures (`Exception` subclasses)
  are caught per source and collected into `UpdateResult.errors`, so one bad file does not
  stop the others. Cancellation (`asyncio.CancelledError`) and other `BaseException`s are not
  collected; they propagate to the caller, and the running producers are cancelled and awaited
  before control returns.
- **Write discipline** — per source: optional `pre_write` (embedding), then the write
  described in the next section.
- **Orphan deletion** — sources present in the store but absent upstream are deleted only
  when `delete_orphans=True` (opt-in footgun guard: on the direct path, only pass your
  complete corpus with this flag).

The engine writes to anything satisfying
[`SourceSyncStore[T]`](../api/pipeline.md#sourcesyncstore) — three methods
(`upsert`, `delete_by_source`, `list_source_state`) that `VectorStore`, `DocumentStore`, and
`MentionStore` all satisfy structurally. A store may additionally implement
[`SourceReplaceStore[T]`](../api/pipeline.md#sourcereplacestore) (`replace_source`), which
changes how updates are written.

## Write discipline

How a changed source is written depends on what it is and on what the store can do:

| Case | What the engine calls | If the write fails |
|---|---|---|
| New source | `upsert(items)` | Nothing was stored before; the source is re-processed as new next run (assuming the store's `upsert` does not leave partial items that already carry the final hash) |
| Updated source, store implements `SourceReplaceStore` | `replace_source(source_id, items)` — new items land **before** stale ones are removed | Store-specific; the contract is that the source must not look "unchanged". The built-in stores are described below: a failure before the stale items are removed leaves the old items in place |
| Updated source, any other store | `delete_by_source(source_id)` then `upsert(items)` | A failed delete skips the upsert (no old/new mixing). A failed upsert leaves the source **empty**; the next run finds no stored state and re-processes it as new |
| Updated source that now yields no items | `delete_by_source(source_id)` | Old items stay; retried next run |

`pre_write` (embedding) runs before any of this, so an embedding failure leaves the store
untouched for that source.

The built-in stores that implement `SourceReplaceStore` are `QdrantVectorStore`,
`QdrantMentionStore`, `QdrantDocumentStore` and `LocalDocumentStore` (`LocalMentionStore` does not, and its writes are not atomic). A custom
store only
needs `replace_source` if it can honour the contract: a failure must not leave the source
looking "unchanged", and should leave the previous items in place.

### What the Qdrant stores guarantee

Qdrant has no multi-request transactions, so the stores make an interrupted write
*detectable* rather than atomic:

- **Blank-then-stamp.** A write that spans several requests (more than 100 points, or any
  `replace_source` of chunks or mentions) first writes the points with `source_hash` / `content_hash` blank and
  stamps the real hashes with a final `set_payload` once every batch has landed. Until that
  stamp, the source reads as "changed". A write that fits one request is a single upsert.
- **A batch of a multi-batch write fails or is cancelled.** The points this call created are
  deleted again, by point id (on cancellation the cleanup runs shielded, then the cancellation
  is re-raised). Points that existed before the call are never deleted; if the call had
  already overwritten one, it is **not** restored to its old content — it keeps the new
  content with blank hashes. This cleanup covers only the batch-writing phase of a write
  larger than one batch: a single-request write has no cleanup, and neither has a failure in
  the later steps below.
- **Replace** writes the new points (blank hashes), then deletes only the source's stale
  points (ids not in the new set), then stamps. What a failure leaves depends on the step:
  while writing, the old chunks or mentions stay in place (plus the cleanup above for a
  multi-batch write); at the stale delete, old and new points **both** remain retrievable,
  the new ones with blank hashes; at the final stamp, the old points are already gone and the
  new ones are in place with blank hashes. In every case the source reads as changed and the
  next sync re-processes it.
- **Hard crash** (process killed, power loss). No cleanup runs: the partially written points
  remain and are retrievable, with blank hashes, so the next sync re-processes the source and
  replaces them.
- **Mixed state.** `list_source_state()` reports a source whose points disagree on their
  hashes with blank hashes — "always changed".

`QdrantDocumentStore` and `LocalDocumentStore` keep one record per source, so an update is a
single atomic replacement (one point; one file written to a temp name and renamed over the
old one). A stored document file that cannot be read or parsed is logged and reported as
"always changed" instead of aborting the listing: a Boundary-1 re-ingest overwrites it, and at
Boundary 2 that one source lands in `UpdateResult.errors` while the others proceed.

### Two things that are not guarantees

- **A source that yields no items is re-produced on every run.** The change token lives on
  the stored items; with nothing stored there is nothing to compare against, so the source is
  parsed, processed and (where applicable) sent to the LLM again each time until it yields
  items.
- **A processor-filtered document is treated differently by the direct paths.**
  `DocumentStorePipeline` keeps the previously stored Document (the producer returns `None`);
  `VectorStorePipeline` and `MentionStorePipeline` in direct mode delete the source's stored
  chunks / mentions (the producer returns an empty change) and count the source as
  `processed`. This is current behaviour, not a contract of the engine.

What each pipeline contributes is just:

1. **Resolution** — turning its request (file paths, or `source_id`s in a DocumentStore) into
   a [`SyncPlanInput`](../api/pipeline.md#syncplaninput): the sources to evaluate, each with
   its current change token, plus the reference set for orphan comparison.
2. **A producer** — one coroutine `SyncSource -> SourceChange[T] | None` that materializes a
   changed source's new items: parse+process+chunk (`VectorStorePipeline`), parse+process
   (`DocumentStorePipeline`), or split+extract (`MentionStorePipeline`). Returning an empty
   item list means "this source now yields nothing" (its stale entries are deleted);
   returning `None` means "unavailable" (warned + counted as skipped).
   `DocumentStorePipeline` returns `None` when processing filters a document out, so a
   transient filter (or a misclassifying LLM processor) never deletes the previously
   stored copy. Resolution failures (an unreadable or vanished file during hashing) are
   captured per path instead of aborting the run: they surface in `UpdateResult.errors`
   and the affected source is never treated as an orphan.

## How the three pipelines compose it

| Pipeline | Boundary | Change token | Producer | Sink |
|---|---|---|---|---|
| `DocumentStorePipeline(ingest=…)` | 1 | `source_hash` | `IngestPipeline.run` (parse → process) | `DocumentStore` |
| `VectorStorePipeline(pipeline=…)` | direct | `source_hash` | full `DocumentPipeline.run` (parse → … → chunk) | `VectorStore` |
| `VectorStorePipeline.from_document_store(chunk=…)` | 2 | `content_hash` | `ChunkPipeline.run` on stored Documents | `VectorStore` |
| `MentionStorePipeline(ingest=…)` | direct | `source_hash` | ingest → split → `Extractor.extract` | `MentionStore` |
| `MentionStorePipeline.from_document_store(…)` | 2 | `content_hash` | split → extract on stored Documents | `MentionStore` |

Each pipeline takes the boundary-appropriate pipeline type
([`IngestPipeline`](../api/pipeline.md#ingestpipeline) or
[`ChunkPipeline`](../api/pipeline.md#chunkpipeline)). Those types have no parameter for a
stage from the other side (`IngestPipeline(splitter=…)` is a `TypeError`), so a misplaced
stage cannot be expressed. The sync pipelines do not check the class of the pipeline object
at runtime, though — passing, say, a `DocumentPipeline` where an `IngestPipeline` is expected
is caught by a type checker, not at construction. See the
[Pipeline Guide](pipeline.md#scenario-e-two-stage-with-a-documentstore) for the two-stage
workflow.

## Result accounting

All three return the same [`UpdateResult`](../api/pipeline.md#updateresult), keyed on
`source_id`:

```python
result = await pipeline.run(paths)
result.processed   # newly added or updated (also: a source that now yields nothing, its items deleted)
result.skipped     # change token matched (or source unavailable)
result.deleted     # orphans removed (only with delete_orphans=True)
result.errors      # [(source_id, exception)] — isolated per source
```

## Extending: your own synced sink

Point the engine at a new sink by implementing the three `SourceSyncStore` methods and writing
a producer — no engine changes. Add `replace_source` (see
[Write discipline](#write-discipline)) if your backend can write a source's new items before
removing the old ones. Index your backing store on the item's `source_id` field (a
first-class field, not metadata) so `delete_by_source` and `list_source_state` stay cheap; see
[Implementing a custom VectorStore](pipeline.md#implementing-a-custom-vectorstore) for a
concrete example.

## See Also

- [Pipeline API — Sync engine](../api/pipeline.md#sync-engine)
- [Pipeline Guide](pipeline.md) — Scenarios B/C/E (vector store, Qdrant, two-stage)
- [Extraction Guide](extraction.md) — the mention sync workflow
