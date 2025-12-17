"""VectorStorePipeline: incremental vector-store sync via plan() / apply().

:class:`VectorStorePipeline` composes a
:class:`~ragdoc.pipeline.linear.DocumentPipeline` with a
:class:`~ragdoc.pipeline.stores.VectorStore` to provide incremental updates.

Two entry points:

* :meth:`run` — the **standard** path. Syncs **per source** (parse → chunk → embed →
  delete-then-upsert), writing each source as soon as it is ready. Each source is atomic
  (embed-first per source); the corpus is **not** all-or-nothing — a completed source stays
  durable even if a later one fails. Flat peak memory (one source's chunks at a time).
* :meth:`plan` / :meth:`apply` — the **reviewable** path. :meth:`plan` determines what changed
  and computes chunks for new/modified sources, returning a serializable
  :class:`~ragdoc.pipeline.changeset.ChangeSet` without touching the store (safe to call
  repeatedly). :meth:`apply` embeds the **whole corpus first** (store untouched if embedding
  fails), then writes per source.

:meth:`run` streams rather than buffering the corpus, so it is **not** ``apply(plan(...))``;
use :meth:`plan` / :meth:`apply` when you need to review changes or want whole-corpus
embed-first semantics.

Provenance: each chunk carries ``source_id`` (from the ``DocumentPipeline``'s ``source_id_fn``)
and ``source_hash`` (file-byte SHA-256, set here via ``hash_fn``).  ``chunk.metadata`` is never
modified.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic

from ragdoc.metadata import TMetadata
from ragdoc.pipeline.changeset import ChangeSet, SourceChange
from ragdoc.processing._concurrency import _resolve_semaphore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.chunking.chunk import Chunk
    from ragdoc.pipeline.embedders import EmbedderConfig
    from ragdoc.pipeline.linear import DocumentPipeline
    from ragdoc.pipeline.stores import DocumentStore, VectorStore


@dataclass
class UpdateResult:
    """Aggregated result of a sync, keyed on ``source_id`` (not Path).

    Attributes:
        processed: source_ids newly added or updated (chunks upserted).
        skipped: source_ids whose stored ``source_hash`` matched the file (unchanged).
            Populated by :meth:`run`; empty for a bare :meth:`apply` (apply does not know
            what was skipped during planning).
        deleted: source_ids removed as orphans (only when ``delete_orphans=True``).
        errors: ``(source_id, exception)`` pairs collected during planning or applying.
            ``asyncio.CancelledError`` is always re-raised, never collected.
    """

    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: list[tuple[str, BaseException]] = field(default_factory=list)


class VectorStorePipeline(Generic[TMetadata]):
    """Incremental vector-store synchronisation via plan/apply.

    In **Boundary-2 mode** (a ``document_store`` is given), the pipeline loads already
    parsed-and-processed Documents from the store and only splits + chunks them. Processors and a
    custom parser are therefore meaningless here and are **rejected at construction** (not
    silently ignored): processing already happened at Boundary 1 (re-running it is destructive),
    and the source is the store, not a file. In **direct mode** (no ``document_store``) every
    stage runs and no such restriction applies.

    Args:
        pipeline: :class:`~ragdoc.pipeline.linear.DocumentPipeline` for parsing,
            processing, splitting, and chunking.  Its ``source_id_fn`` is the single source
            of truth for source identity (read here for collision/orphan checks).
        vector_store: :class:`~ragdoc.pipeline.stores.VectorStore` implementation.
        embedders: Named embedder configs.  Each key becomes an entry in
            ``chunk.named_embeddings``.  Embedders run concurrently in :meth:`apply`.
        document_store: Optional :class:`~ragdoc.pipeline.stores.DocumentStore`. When set,
            the pipeline runs in **Mode 2** (chunk Documents read from the store) instead of
            the direct file path.
        hash_fn: ``Path -> str`` producing the file-byte change-detection hash.  Defaults to
            SHA-256 of the raw bytes.  Override for custom filesystems (e.g. S3) where
            ``path.read_bytes()`` is unavailable.
        concurrency: Max sources processed concurrently in :meth:`plan` (``int`` or a shared
            ``asyncio.Semaphore``).  Defaults to 10.
    """

    def __init__(
        self,
        pipeline: DocumentPipeline[TMetadata],
        vector_store: VectorStore,
        embedders: dict[str, EmbedderConfig] | None = None,
        document_store: DocumentStore | None = None,
        hash_fn: Callable[[Path], str] = lambda p: _file_hash(p),
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        if document_store is not None:
            misplaced: list[str] = []
            if pipeline.has_processors:
                misplaced.append("processors (they ran once at Boundary 1; re-running is destructive)")
            if pipeline.has_custom_parser:
                misplaced.append("a custom parser (the source is the document_store, not a file)")
            if misplaced:
                raise ValueError(
                    "A Boundary-2 VectorStorePipeline (document_store given) splits and chunks "
                    "Documents loaded from the store; its DocumentPipeline must not carry "
                    + " or ".join(misplaced)
                    + ". Configure those on the DocumentStorePipeline (Boundary 1) instead."
                )

        self._pipeline = pipeline
        self._vector_store = vector_store
        self._embedders = embedders or {}
        self._document_store = document_store
        self._hash_fn = hash_fn
        self._concurrency = concurrency

    @property
    def _source_id_fn(self) -> Callable[[Path], str]:
        # Single source of truth — the DocumentPipeline owns it.
        return self._pipeline.source_id_fn

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    async def plan(
        self, sources: Iterable[Path] | Iterable[str] | None = None, delete_orphans: bool = False
    ) -> ChangeSet[Chunk]:
        """Compute the change set without touching the vector store.

        Two modes, selected by whether a ``document_store`` was configured:

        * **Direct (no document_store):** *sources* is an iterable of file ``Path``\\ s
          (required).  Files are hashed and compared via ``vector_store.list_source_state()``;
          changed files are parsed/chunked.  ``source_hash`` (file bytes) is the change token.
        * **From DocumentStore:** *sources* is an iterable of ``source_id`` strings, or
          ``None`` for every document in the store.  Documents whose ``content_hash`` differs
          from the vector store's are re-chunked.  ``content_hash`` is the change token.

        Args:
            sources: Paths (direct) or source_ids / ``None`` (from DocumentStore).
            delete_orphans: Opt-in removal.  **Footgun (direct):** only safe when *sources* is
                your complete corpus.  From a DocumentStore, orphans are source_ids in the
                vector store but absent from the document store.

        Returns:
            A :class:`~ragdoc.pipeline.changeset.ChangeSet` of chunks.
        """
        changeset, _skipped, errors = await self._plan_internal(sources, delete_orphans)
        if errors:
            logger.warning(f"plan(): {len(errors)} source(s) failed during processing: {[sid for sid, _ in errors]}")
        return changeset

    async def _plan_internal(
        self, sources: Iterable[Path] | Iterable[str] | None, delete_orphans: bool
    ) -> tuple[ChangeSet[Chunk], list[str], list[tuple[str, BaseException]]]:
        if self._document_store is not None:
            return await self._plan_from_doc_store(sources, delete_orphans)  # type: ignore[arg-type]
        if sources is None:
            raise ValueError("Direct mode (no document_store) requires `sources` paths.")
        return await self._plan_from_paths(sources, delete_orphans)  # type: ignore[arg-type]

    def _build_current_map(self, sources: Iterable[Path]) -> dict[str, tuple[Path, str]]:
        """Map ``source_id -> (path, file_hash)`` for *sources*, raising on id collisions.

        Shared by :meth:`_plan_from_paths` and :meth:`_run_from_paths` so both use one identity
        and collision implementation (single source of truth, no drift).
        """
        current: dict[str, tuple[Path, str]] = {}
        collisions: dict[str, list[Path]] = {}
        for path in sources:
            sid = self._source_id_fn(path)
            if sid in current:
                collisions.setdefault(sid, [current[sid][0]]).append(path)
            else:
                current[sid] = (path, self._hash_fn(path))

        if collisions:
            lines = [f"  {sid!r}: {[str(p) for p in paths]}" for sid, paths in collisions.items()]
            raise ValueError(
                "source_id_fn produced duplicate source_ids for different paths.\n"
                + "\n".join(lines)
                + "\nConsider a relative-path strategy: lambda p: str(p.relative_to(base_dir))"
            )
        return current

    async def _plan_from_paths(
        self, sources: Iterable[Path], delete_orphans: bool
    ) -> tuple[ChangeSet[Chunk], list[str], list[tuple[str, BaseException]]]:
        current = self._build_current_map(sources)

        store_state = await self._vector_store.list_source_state()
        logger.info(f"plan(): {len(current)} sources, {len(store_state)} known in store")

        to_add: list[SourceChange[Chunk]] = []
        to_update: list[SourceChange[Chunk]] = []
        skipped: list[str] = []
        errors: list[tuple[str, BaseException]] = []
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str, path: Path, file_hash: str) -> None:
            async with sem:
                try:
                    existing = store_state.get(sid)
                    if existing is not None and existing.source_hash == file_hash:
                        skipped.append(sid)
                        return
                    chunks = await self._pipeline.run(path)
                    # source_id is already stamped by chunk_document (via doc.source_id, same
                    # source_id_fn). source_hash is a sync-only concern unknown at chunk time.
                    for chunk in chunks:
                        chunk.source_hash = file_hash
                    content_hash = chunks[0].content_hash if chunks else None
                    change: SourceChange[Chunk] = SourceChange(
                        source_id=sid,
                        source_hash=file_hash,
                        content_hash=content_hash,
                        items=chunks,
                    )
                    (to_add if existing is None else to_update).append(change)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error(f"plan() failed for {path.name}: {exc!r}", exc_info=True)
                    errors.append((sid, exc))

        await asyncio.gather(*[_process(sid, p, h) for sid, (p, h) in current.items()])

        to_delete = [sid for sid in store_state if sid not in current] if delete_orphans else []

        return ChangeSet(to_add=to_add, to_update=to_update, to_delete=to_delete), skipped, errors

    async def _plan_from_doc_store(
        self, source_ids: Iterable[str] | None, delete_orphans: bool
    ) -> tuple[ChangeSet[Chunk], list[str], list[tuple[str, BaseException]]]:
        if self._document_store is None:
            raise AssertionError("_plan_from_doc_store called without a document_store")
        doc_state = await self._document_store.list_source_state()
        vec_state = await self._vector_store.list_source_state()
        targets = list(source_ids) if source_ids is not None else list(doc_state.keys())
        logger.info(f"plan(): {len(targets)} doc-store sources, {len(vec_state)} in vec store")

        to_add: list[SourceChange[Chunk]] = []
        to_update: list[SourceChange[Chunk]] = []
        skipped: list[str] = []
        errors: list[tuple[str, BaseException]] = []
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str) -> None:
            async with sem:
                try:
                    doc_entry = doc_state.get(sid)
                    if doc_entry is None:
                        logger.warning(f"plan(): source_id {sid!r} not in document store; skipping")
                        return
                    existing = vec_state.get(sid)
                    # content_hash drives Boundary-2 detection; None stored ⇒ always changed.
                    if (
                        existing is not None
                        and existing.content_hash is not None
                        and existing.content_hash == doc_entry.content_hash
                    ):
                        skipped.append(sid)
                        return
                    doc = await self._document_store.get_document(sid)  # type: ignore[union-attr]
                    if doc is None:
                        return
                    chunks = await self._pipeline.chunk_document(doc)
                    change: SourceChange[Chunk] = SourceChange(
                        source_id=sid,
                        source_hash=doc.source_hash or "",
                        content_hash=doc.content_hash(),
                        items=chunks,
                    )
                    (to_add if existing is None else to_update).append(change)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error(f"plan() failed for {sid}: {exc!r}", exc_info=True)
                    errors.append((sid, exc))

        await asyncio.gather(*[_process(sid) for sid in targets])

        # Orphans are compared against the FULL document store (not the target subset),
        # so a subset run never deletes documents that still exist upstream.
        to_delete = [sid for sid in vec_state if sid not in doc_state] if delete_orphans else []

        return ChangeSet(to_add=to_add, to_update=to_update, to_delete=to_delete), skipped, errors

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    async def apply(self, changeset: ChangeSet[Chunk]) -> UpdateResult:
        """Embed and write *changeset* to the store.

        Embeds **all** chunks first (non-destructive): if embedding fails, the store is left
        untouched and every changed source is reported in ``errors``.  Then, per source:
        delete orphans, delete stale chunks for updates, and upsert.

        Args:
            changeset: The (possibly hand-edited) ChangeSet to apply.

        Returns:
            :class:`UpdateResult` (``skipped`` is empty — apply does not know plan-time skips).
        """
        result = UpdateResult()
        changes = list(changeset.to_add) + list(changeset.to_update)
        all_chunks = [c for sc in changes for c in sc.items]

        try:
            await self._embed_chunks(all_chunks)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error("Embedding failed; store left untouched.", exc_info=True)
            for sc in changes:
                result.errors.append((sc.source_id, exc))
            return result

        errored: set[str] = set()

        for sid in changeset.to_delete:
            try:
                await self._vector_store.delete_by_source(sid)
                result.deleted.append(sid)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error(f"apply(): delete orphan {sid} failed: {exc!r}", exc_info=True)
                result.errors.append((sid, exc))

        for sc in changeset.to_update:
            try:
                await self._vector_store.delete_by_source(sc.source_id)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error(f"apply(): delete stale {sc.source_id} failed: {exc!r}", exc_info=True)
                result.errors.append((sc.source_id, exc))
                errored.add(sc.source_id)

        for sc in changes:
            if sc.source_id in errored:
                continue
            try:
                if sc.items:
                    await self._vector_store.upsert(sc.items)
                result.processed.append(sc.source_id)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error(f"apply(): upsert {sc.source_id} failed: {exc!r}", exc_info=True)
                result.errors.append((sc.source_id, exc))

        logger.info(
            f"apply(): {len(result.processed)} processed, {len(result.deleted)} deleted, {len(result.errors)} errors"
        )
        return result

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        sources: Iterable[Path] | Iterable[str] | None = None,
        delete_orphans: bool = False,
    ) -> UpdateResult:
        """Streaming, **per-source** sync (the standard path).

        Produces the **same end result** as ``apply(plan(sources, delete_orphans))`` — same change
        detection (skip/add/update/orphan) and the same final store contents. The difference is
        *scheduling*, not outcome: ``apply(plan(...))`` chunks the whole corpus and embeds *all*
        chunks before any write, whereas ``run()`` processes each source independently and writes
        it as soon as it is ready (parse → chunk → embed → delete-then-upsert). ``run()`` is **not**
        literally implemented as ``apply(plan(...))``, but it is equivalent up to the per-source vs
        whole-corpus failure boundary described below.

        **Per-source atomicity.** Each source is written delete-then-upsert, so an individual
        source is never left half-replaced, and its embed runs *before* its write (an embed
        failure leaves that one source untouched and records it in ``errors``). The **corpus** is
        *not* all-or-nothing: a source that succeeds is durable in the store even if a later
        source fails. Use ``apply(plan(...))`` when you need whole-corpus embed-first semantics
        or a reviewable :class:`~ragdoc.pipeline.changeset.ChangeSet`.

        Args:
            sources: Paths (direct mode) or source_ids / ``None`` for all (from a
                DocumentStore). See :meth:`plan`.
            delete_orphans: opt-in orphan removal (footgun in direct mode: pass the full
                corpus).

        Returns:
            :class:`UpdateResult` (``skipped`` = unchanged sources, ``errors`` = per-source
            parse/chunk/embed/write failures).
        """
        try:
            if self._document_store is not None:
                result = await self._run_from_doc_store(sources, delete_orphans)  # type: ignore[arg-type]
            elif sources is None:
                raise ValueError("Direct mode (no document_store) requires `sources` paths.")
            else:
                result = await self._run_from_paths(sources, delete_orphans)  # type: ignore[arg-type]
        except BaseException:
            # A fatal exception (e.g. asyncio.CancelledError) escaped the run loop — collected
            # per-source errors do not reach here; this is the abort path.
            logger.error("Run failed before completion", exc_info=True)
            raise
        logger.info(
            f"run complete: {len(result.processed)} processed, {len(result.skipped)} skipped, "
            f"{len(result.deleted)} deleted, {len(result.errors)} errors"
        )
        return result

    async def _sync_one(self, source_id: str, chunks: list[Chunk], *, is_update: bool) -> None:
        """Embed *chunks* for one source, then write them (atomic per source).

        Embed-first **per source**: embedding raises before any store mutation, so a failed embed
        leaves this source untouched. ``is_update`` ⇒ delete the source's stale chunks before
        upserting (atomic replace); a fresh source skips the delete.
        """
        await self._embed_chunks(chunks)
        if is_update:
            await self._vector_store.delete_by_source(source_id)
        if chunks:
            await self._vector_store.upsert(chunks)

    async def _run_from_paths(self, sources: Iterable[Path], delete_orphans: bool) -> UpdateResult:
        current = self._build_current_map(sources)
        store_state = await self._vector_store.list_source_state()
        logger.info(f"run(): {len(current)} sources, {len(store_state)} known in store")

        result = UpdateResult()
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str, path: Path, file_hash: str) -> None:
            async with sem:
                try:
                    existing = store_state.get(sid)
                    if existing is not None and existing.source_hash == file_hash:
                        result.skipped.append(sid)
                        return
                    chunks = await self._pipeline.run(path)
                    # source_id is already stamped by chunk_document (via doc.source_id, same
                    # source_id_fn). source_hash is a sync-only concern unknown at chunk time.
                    for chunk in chunks:
                        chunk.source_hash = file_hash
                    await self._sync_one(sid, chunks, is_update=existing is not None)
                    result.processed.append(sid)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error(f"run() failed for {path.name}: {exc!r}", exc_info=True)
                    result.errors.append((sid, exc))

        await asyncio.gather(*[_process(sid, p, h) for sid, (p, h) in current.items()])

        if delete_orphans:
            await self._delete_orphans([sid for sid in store_state if sid not in current], result)
        return result

    async def _run_from_doc_store(self, source_ids: Iterable[str] | None, delete_orphans: bool) -> UpdateResult:
        if self._document_store is None:
            raise AssertionError("_run_from_doc_store called without a document_store")
        doc_state = await self._document_store.list_source_state()
        vec_state = await self._vector_store.list_source_state()
        targets = list(source_ids) if source_ids is not None else list(doc_state.keys())
        logger.info(f"run(): {len(targets)} doc-store sources, {len(vec_state)} in vec store")

        result = UpdateResult()
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str) -> None:
            async with sem:
                try:
                    doc_entry = doc_state.get(sid)
                    if doc_entry is None:
                        logger.warning(f"run(): source_id {sid!r} not in document store; skipping")
                        return
                    existing = vec_state.get(sid)
                    # content_hash drives Boundary-2 detection; None stored ⇒ always changed.
                    if (
                        existing is not None
                        and existing.content_hash is not None
                        and existing.content_hash == doc_entry.content_hash
                    ):
                        result.skipped.append(sid)
                        return
                    doc = await self._document_store.get_document(sid)  # type: ignore[union-attr]
                    if doc is None:
                        return
                    # Already processed at Boundary 1 — split/chunk only (see _plan_from_doc_store).
                    chunks = await self._pipeline.chunk_document(doc)
                    await self._sync_one(sid, chunks, is_update=existing is not None)
                    result.processed.append(sid)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error(f"run() failed for {sid}: {exc!r}", exc_info=True)
                    result.errors.append((sid, exc))

        await asyncio.gather(*[_process(sid) for sid in targets])

        if delete_orphans:
            # Orphans compared against the FULL document store (not the target subset).
            await self._delete_orphans([sid for sid in vec_state if sid not in doc_state], result)
        return result

    async def _delete_orphans(self, orphan_ids: list[str], result: UpdateResult) -> None:
        """Delete *orphan_ids* from the vector store, recording outcomes on *result*."""
        for sid in orphan_ids:
            try:
                await self._vector_store.delete_by_source(sid)
                result.deleted.append(sid)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error(f"run(): delete orphan {sid} failed: {exc!r}", exc_info=True)
                result.errors.append((sid, exc))

    async def _embed_chunks(self, chunks: list[Chunk]) -> None:
        """Populate ``chunk.named_embeddings`` for all configured embedders (in place)."""
        if not self._embedders or not chunks:
            return

        async def _run_one(name: str, config: EmbedderConfig) -> tuple[str, list[list[float]]]:
            texts = [config.text_fn(chunk) for chunk in chunks]
            vectors = await config.embedder.embed(texts)
            return name, vectors

        results = await asyncio.gather(*(_run_one(name, cfg) for name, cfg in self._embedders.items()))
        for name, vectors in results:
            for chunk, vec in zip(chunks, vectors, strict=True):
                chunk.named_embeddings[name] = vec

    async def run_directory(self, directory: Path, glob: str = "**/*") -> UpdateResult:
        """Discover files matching *glob* under *directory*, then :meth:`run` them."""
        sources = [p for p in directory.glob(glob) if p.is_file()]  # noqa: ASYNC240 -- threaded in Phase 8 (D9)
        return await self.run(sources)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the raw bytes of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
