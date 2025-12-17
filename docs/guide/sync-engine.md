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
| `apply(changeset)` | Write a (possibly edited) ChangeSet | buffered, embed-first |
| `run(...)` | `apply(plan(...))` in streaming form | per source, as produced |

`run()` produces the **same end state** as `apply(plan(...))` — same change detection, same
final store contents — differing only in scheduling: `run` writes each changed source as it
completes (a finished source stays durable even if a later one fails), while `apply` calls the
`pre_write` hook once over the whole corpus first. For `VectorStorePipeline` that hook is
embedding: `apply` embeds **all** chunks before any deletion, so a mid-run embedding failure
leaves the store untouched.

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
  Document changes its hash, which triggers re-chunking/re-extraction. A stored
  `content_hash` of `None` means "always changed".

Both live as first-class fields on the synced items (`Chunk`, `Document`, `Mention`) — never
in `metadata`.

## The engine, the stores, the producers

[`SyncEngine[T]`](../api/pipeline.md#syncengine) owns the invariant machinery:

- **Change detection** — compare each source's current token against the stored state
  (`list_source_state()`), skip matches.
- **Concurrency** — sources are produced under a semaphore and streamed as completed.
- **Error isolation** — production and write failures are caught per source
  (`except Exception`, never `BaseException` — cancellation always propagates) and collected
  into `UpdateResult.errors`; one bad file never aborts the corpus.
- **Write discipline** — per source: optional `pre_write` (embedding) → delete stale entries
  by `source_id` → upsert. A source whose stale-delete failed is *not* upserted (no old/new
  mixing).
- **Orphan deletion** — sources present in the store but absent upstream are deleted only
  when `delete_orphans=True` (opt-in footgun guard: on the direct path, only pass your
  complete corpus with this flag).

The engine writes to anything satisfying
[`SourceSyncStore[T]`](../api/pipeline.md#sourcesyncstore) — three methods
(`upsert`, `delete_by_source`, `list_source_state`) that `VectorStore`, `DocumentStore`, and
`MentionStore` all satisfy structurally.

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
[`ChunkPipeline`](../api/pipeline.md#chunkpipeline)), so a stage on the wrong side of a
boundary is a `TypeError` at construction — see the
[Pipeline Guide](pipeline.md#scenario-e-two-stage-with-a-documentstore) for the two-stage
workflow.

## Result accounting

All three return the same [`UpdateResult`](../api/pipeline.md#updateresult), keyed on
`source_id`:

```python
result = await pipeline.run(paths)
result.processed   # newly added or updated
result.skipped     # change token matched (or source unavailable)
result.deleted     # orphans removed (only with delete_orphans=True)
result.errors      # [(source_id, exception)] — isolated per source
```

## Extending: your own synced sink

Point the engine at a new sink by implementing the three `SourceSyncStore` methods and writing
a producer — no engine changes. Index your backing store on the item's `source_id` field (a
first-class field, not metadata) so `delete_by_source` and `list_source_state` stay cheap; see
[Implementing a custom VectorStore](pipeline.md#implementing-a-custom-vectorstore) for a
concrete example.

## See Also

- [Pipeline API — Sync engine](../api/pipeline.md#sync-engine)
- [Pipeline Guide](pipeline.md) — Scenarios B/C/E (vector store, Qdrant, two-stage)
- [Extraction Guide](extraction.md) — the mention sync workflow
