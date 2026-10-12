"""Generic plan/apply/run sync engine shared by all sync pipelines (Decision D2-A).

One streaming core drives every sync pipeline (``VectorStorePipeline``,
``DocumentStorePipeline``, ``MentionStorePipeline``):

* :meth:`SyncEngine.plan` buffers the per-source event stream into a serializable
  :class:`~ragdoc.pipeline.changeset.ChangeSet` (no store writes).
* :meth:`SyncEngine.run` consumes the **same** event stream, writing each changed source as
  its event is yielded (per-source ``pre_write`` → write, see below).
* :meth:`SyncEngine.apply` writes an already-buffered ChangeSet, calling ``pre_write`` once
  over the whole corpus first (embed-first semantics: the store is untouched if it raises).

**How an updated source is written** depends on the store:

* A store implementing :class:`SourceReplaceStore` gets one ``replace_source`` call. The
  store writes the new items before removing the stale ones, so a failure while writing
  leaves the previously stored items in place; a failure in a later step (removing stale
  items, committing) is store-specific but must never leave the source looking "unchanged".
* Any other store gets ``delete_by_source`` followed by ``upsert``. If the upsert fails
  after the delete, the source is left **empty** in the store (the error is recorded); the
  next run finds no stored state for it and re-processes it as new.

**Sources that yield no items** (an empty ``SourceChange.items``) leave nothing in the store
that could carry their change token, so they are re-produced on every run — see
:data:`Producer`.

``run()`` therefore produces the same end state as ``apply(plan(...))`` — same change
detection, same final store contents — differing only in *scheduling*: per-source
``pre_write``-and-write versus whole-corpus ``pre_write``-first.

Unified error discipline: production and write failures are caught per source with
``except Exception`` (never ``BaseException``), so ``asyncio.CancelledError``,
``KeyboardInterrupt``, and ``SystemExit`` always propagate.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

from ragdoc.pipeline.changeset import ChangeSet, SourceChange
from ragdoc.pipeline.stores import SourceState
from ragdoc.utils.concurrency import resolve_semaphore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
"""Sync payload type: ``Chunk``, ``Document``, ``Mention[P]``, ..."""


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the raw bytes of *path*.

    The default file-byte change-detection hash for the direct sync path (Boundary 1).
    Synchronous by design — the engine wraps it in ``asyncio.to_thread``; for remote
    filesystems (e.g. S3) pass a custom sync ``hash_fn`` that reads your backend.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@runtime_checkable
class SourceSyncStore(Protocol[T]):
    """The minimal source-aware sink contract the engine writes to.

    Structurally satisfied by ``VectorStore`` (payload ``Chunk``), ``DocumentStore``
    (payload ``Document``), and ``MentionStore`` (payload ``Mention``) — any store offering
    these three methods can be driven by a :class:`SyncEngine`. Invariant in ``T``:
    the payload appears inside mutable ``list`` parameters.
    """

    async def upsert(self, items: list[T], /) -> list[str]:
        """Add or replace *items*; return their stored ids."""
        ...

    async def delete_by_source(self, source_id: str, /) -> None:
        """Delete every item whose ``source_id`` equals *source_id* (no-op if absent)."""
        ...

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`~ragdoc.pipeline.stores.SourceState` for every source."""
        ...


@runtime_checkable
class SourceReplaceStore(Protocol[T]):
    """Optional sink capability: replace one source's items without a delete-first window.

    A store that can write a source's new items **before** removing its stale ones implements
    this next to :class:`SourceSyncStore`; the engine detects it structurally and uses it for
    every *update* that carries items. Stores without it get ``delete_by_source`` + ``upsert``.
    """

    async def replace_source(self, source_id: str, items: list[T], /) -> list[str]:
        """Make *items* the complete stored content of *source_id*; return their stored ids.

        Contract: a failure must not leave the source looking "unchanged" to
        ``list_source_state``, and should leave the previously stored items in place.
        """
        ...


@dataclass(frozen=True)
class SyncSource:
    """One source as the engine sees it before production.

    Attributes:
        source_id: Identity key (from ``source_id_fn`` or the DocumentStore).
        change_token: The *current* token — file-byte hash (direct mode) or the document's
            ``content_hash`` (Boundary 2). Compared against the stored token selected by
            the engine's ``token_of``.
        path: The source file (direct mode only; ``None`` in Boundary-2 mode).
    """

    source_id: str
    change_token: str
    path: Path | None = None


@dataclass(frozen=True)
class SyncPlanInput:
    """A resolved sync request: the targets to evaluate plus the orphan reference set.

    Attributes:
        sources: The sources to evaluate this run.
        live_source_ids: The set of ids considered alive upstream. Direct mode: the ids of
            *sources*. Boundary-2 mode: **every** id in the DocumentStore, so a subset run
            never deletes stored items whose documents still exist upstream. Sources that
            failed resolution (``pre_failed``) belong here too — unreadable is not orphaned.
        pre_skipped: source_ids resolved as unavailable before production (e.g. a requested
            Boundary-2 target absent from the DocumentStore). :meth:`SyncEngine.run` counts
            them in ``skipped``; :meth:`SyncEngine.plan` drops them (as it drops all skips).
        pre_failed: ``(source_id, exception)`` pairs that failed during resolution (e.g. an
            unreadable file whose ``hash_fn`` raised). :meth:`SyncEngine.run` surfaces them
            in ``UpdateResult.errors``; :meth:`SyncEngine.plan` logs a summary warning and
            omits them (the same treatment as production errors).
    """

    sources: Sequence[SyncSource]
    live_source_ids: frozenset[str]
    pre_skipped: tuple[str, ...] = ()
    pre_failed: tuple[tuple[str, Exception], ...] = ()


Producer = Callable[[SyncSource], Awaitable["SourceChange[T] | None"]]
"""Per-source item production. The two "nothing to write" results mean different things:

* A :class:`~ragdoc.pipeline.changeset.SourceChange` with **empty** ``items`` — "the source
  now yields nothing": the engine **deletes** whatever is stored for it and counts it
  processed.
* ``None`` — "no payload this run": the engine logs a warning, counts the source skipped
  and leaves any stored state **untouched**.

Which one a filtered source produces is the pipeline's choice, and the built-in pipelines
differ (current behaviour, not a guarantee of the engine):

* ``DocumentStorePipeline`` returns ``None`` for a processor-filtered document, so the stored
  Document is preserved.
* ``VectorStorePipeline`` in direct mode returns an empty change for a processor-filtered
  document (and for any document that chunks to nothing), so its stored chunks are deleted.
* ``None`` is also returned when a Boundary-2 document vanished between listing and loading.

Either way the store keeps **no change token** for a source without stored items (tokens
live on the items), so such a source is produced again on every run — parsed, processed and,
where applicable, LLM-extracted — until it yields items.
"""

TokenSelector = Callable[[SourceState], "str | None"]
"""Selects the stored comparand from :class:`SourceState`.

``None`` ⇒ "always changed" (pins the stored-``content_hash``-is-``None`` rule).
"""

PreWriteHook = Callable[[list[T]], Awaitable[None]]
"""Pre-write hook, called with the items about to be written (mutates in place — embedding).

:meth:`SyncEngine.run` calls it once per source; :meth:`SyncEngine.apply` calls it once with
the whole changeset's items (embed-first) — there it receives **copies**, so the caller's
changeset is left untouched.
"""


def source_hash_token(state: SourceState) -> str | None:
    """Direct-path selector: the stored file-byte hash (``''`` ⇒ unknown ⇒ always changed)."""
    return state.source_hash or None


def content_hash_token(state: SourceState) -> str | None:
    """Boundary-2 selector: the stored ``content_hash`` (``None`` or ``''`` ⇒ always changed)."""
    return state.content_hash or None


@dataclass(frozen=True)
class SourceOutcome(Generic[T]):
    """One event from the streaming core — exactly one per evaluated source, plus one per orphan.

    Attributes:
        kind: ``"skipped"`` — stored token matched (or the source was unavailable);
            ``"changed"`` — new or updated (``change`` carries the items; ``is_new``
            disambiguates); ``"error"`` — production raised (``error`` carries the
            exception); ``"orphan"`` — in the store but not in ``live_source_ids``
            (emitted only when ``delete_orphans``).
        source_id: The source this event is about.
        change: The produced change (``"changed"`` only).
        is_new: True when the source has no stored state yet (``"changed"`` only).
        error: The collected exception (``"error"`` only).
    """

    kind: Literal["skipped", "changed", "error", "orphan"]
    source_id: str
    change: SourceChange[T] | None = None
    is_new: bool = False
    error: Exception | None = None


@dataclass
class UpdateResult:
    """Aggregated result of a sync, keyed on ``source_id`` (not Path).

    Attributes:
        processed: source_ids newly added or updated (items written — or, for a source
            that now yields no items, its stored items deleted).
        skipped: source_ids whose stored change token matched (unchanged) or that were
            unavailable at production time. Populated by ``run``; empty for a bare
            ``apply`` (apply does not know what was skipped during planning).
        deleted: source_ids removed as orphans (only when ``delete_orphans=True``).
        errors: ``(source_id, exception)`` pairs collected during resolution (hashing —
            see ``SyncPlanInput.pre_failed``), production, or writing. Only ``Exception``
            subclasses are collected; ``asyncio.CancelledError`` and other
            ``BaseException`` subclasses always propagate.
    """

    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: list[tuple[str, Exception]] = field(default_factory=list)


class CurrentMap(dict[str, tuple[Path, str]]):
    """``source_id -> (path, file_hash)`` plus the per-source hashing failures.

    A plain ``dict`` (so every existing consumer keeps working) carrying an extra
    :attr:`errors` channel: sources whose ``hash_fn`` raised are **excluded** from the
    mapping and recorded here instead, so one unreadable file never aborts a whole sync.
    """

    def __init__(self, entries: dict[str, tuple[Path, str]], errors: dict[str, Exception]) -> None:
        super().__init__(entries)
        self.errors: dict[str, Exception] = errors
        """``source_id -> exception`` for every source whose hashing failed."""


async def build_current_map(
    sources: Iterable[Path],
    source_id_fn: Callable[[Path], str],
    hash_fn: Callable[[Path], str],
    concurrency: int | asyncio.Semaphore,
) -> CurrentMap:
    """Map ``source_id -> (path, file_hash)`` for *sources*, raising on id collisions.

    Collision detection runs **before** any hashing, so a duplicate ``source_id`` costs no
    I/O. Hashing then fans out with ``asyncio.to_thread`` under the semaphore (*hash_fn*
    stays synchronous; see :func:`file_hash`).

    Per-source error isolation: a *hash_fn* failure (unreadable file, permission error)
    excludes that source from the mapping and records it in :attr:`CurrentMap.errors`
    instead of aborting — callers thread those errors into ``SyncPlanInput.pre_failed``
    (and keep the failed ids in ``live_source_ids`` so they are never orphan-deleted).
    Only ``Exception`` subclasses are captured; cancellation always propagates.

    Args:
        sources: The source files to identify and hash.
        source_id_fn: ``Path -> source_id`` identity function (single source of truth).
        hash_fn: ``Path -> str`` sync change-detection hash.
        concurrency: Max concurrent hash threads (``int`` or a shared semaphore).

    Returns:
        A :class:`CurrentMap`: ``source_id -> (path, file_hash)`` for every hashable
        source, with per-source failures in ``.errors``.

    Raises:
        ValueError: If two paths map to the same ``source_id``.
    """
    paths: dict[str, Path] = {}
    collisions: dict[str, list[Path]] = {}
    for path in sources:
        sid = source_id_fn(path)
        if sid in paths:
            collisions.setdefault(sid, [paths[sid]]).append(path)
        else:
            paths[sid] = path

    if collisions:
        lines = [f"  {sid!r}: {[str(p) for p in collided]}" for sid, collided in collisions.items()]
        raise ValueError(
            "source_id_fn produced duplicate source_ids for different paths.\n"
            + "\n".join(lines)
            + "\nConsider a relative-path strategy: lambda p: str(p.relative_to(base_dir))"
        )

    sem = resolve_semaphore(concurrency)

    async def _hash_one(path: Path) -> str:
        async with sem:
            return await asyncio.to_thread(hash_fn, path)

    results = await asyncio.gather(*(_hash_one(path) for path in paths.values()), return_exceptions=True)
    entries: dict[str, tuple[Path, str]] = {}
    errors: dict[str, Exception] = {}
    for (sid, path), result in zip(paths.items(), results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result  # cancellation and other fatal BaseExceptions always propagate
            logger.error(f"hashing failed for {path} (source_id={sid!r}): {result!r}")
            errors[sid] = result
        else:
            entries[sid] = (path, result)
    return CurrentMap(entries, errors)


class SyncEngine(Generic[T]):
    """Streaming plan/apply/run core shared by all sync pipelines.

    The pipelines own source **resolution** (paths or store enumeration →
    :class:`SyncPlanInput`) and the per-source **producer**; the engine owns change
    detection, concurrency, error isolation, orphan deletion, and the write discipline.
    Pointing the engine at a new sink is configuration: any
    :class:`SourceSyncStore` plus a producer coroutine.

    Args:
        store: The sink to synchronize.
        producer: Per-source production (see :data:`Producer`). Called only for sources
            whose stored change token does not match.
        token_of: Selects the stored comparand from :class:`SourceState`
            (:func:`source_hash_token` for the direct path, :func:`content_hash_token`
            for Boundary 2).
        pre_write: Optional hook mutating items before they are written (embedding).
        concurrency: Max sources produced concurrently (``int`` or a shared semaphore).
            In :meth:`run` a producer keeps its slot until the writer has taken its result
            (back-pressure), so this also bounds how many produced-but-unwritten sources
            are held in memory. Consequence for a **shared** semaphore: nothing on the write
            path (``pre_write``, the store) may acquire that same semaphore, or the writer
            and the waiting producers block each other.
    """

    def __init__(
        self,
        store: SourceSyncStore[T],
        producer: Producer[T],
        token_of: TokenSelector,
        pre_write: PreWriteHook[T] | None = None,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        self._store = store
        self._producer = producer
        self._token_of = token_of
        self._pre_write = pre_write
        self._concurrency = concurrency

    # ------------------------------------------------------------------
    # streaming core
    # ------------------------------------------------------------------

    async def _evaluate(self, src: SyncSource, state: dict[str, SourceState]) -> SourceOutcome[T]:
        """Evaluate one source against the stored *state*: skip, produce, or collect the error."""
        try:
            existing = state.get(src.source_id)
            if existing is not None:
                stored = self._token_of(existing)
                if stored is not None and stored == src.change_token:
                    return SourceOutcome(kind="skipped", source_id=src.source_id)
            change = await self._producer(src)
            if change is None:
                logger.warning(f"sync: source {src.source_id!r} unavailable; skipping")
                return SourceOutcome(kind="skipped", source_id=src.source_id)
            return SourceOutcome(kind="changed", source_id=src.source_id, change=change, is_new=existing is None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"sync failed for {src.source_id}: {exc!r}")
            return SourceOutcome(kind="error", source_id=src.source_id, error=exc)

    async def _stream(self, request: SyncPlanInput, delete_orphans: bool) -> AsyncGenerator[SourceOutcome[T], None]:
        """Yield one outcome per source (as completed), then one orphan outcome per orphan.

        **Back-pressure.** Each source's task evaluates under the concurrency semaphore and
        keeps its slot until its outcome has been handed to the consumer through a one-slot
        queue. A slow consumer (the writer in :meth:`run`) therefore stalls production instead
        of letting finished outcomes pile up: at most ``concurrency + 2`` outcomes exist at
        any time (one per slot, one queued, one held by the consumer), independent of corpus
        size. No reference to an outcome is kept once it has been yielded. One lightweight
        task per source is still created up front (waiting on the semaphore).

        The ``finally`` block cancels and gathers every task, so an early abort (consumer
        error, cancellation) never abandons a running producer. A fatal ``BaseException``
        escaping a producer task is re-raised here.
        """
        state = await self._store.list_source_state()
        logger.info(f"sync: {len(request.sources)} sources, {len(state)} known in store")
        sem = resolve_semaphore(self._concurrency)
        handoff: asyncio.Queue[SourceOutcome[T]] = asyncio.Queue(maxsize=1)
        fatal: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def _guarded(src: SyncSource) -> None:
            async with sem:
                await handoff.put(await self._evaluate(src, state))

        def _on_done(task: asyncio.Task[None]) -> None:
            # Only abnormal ends matter: wake the consumer so it never waits on a dead producer.
            if fatal.done():
                return
            if task.cancelled():
                fatal.cancel()
            elif (exc := task.exception()) is not None:
                fatal.set_exception(exc)

        tasks = [asyncio.create_task(_guarded(src)) for src in request.sources]
        for task in tasks:
            task.add_done_callback(_on_done)
        try:
            for _ in range(len(tasks)):
                getter = asyncio.ensure_future(handoff.get())
                try:
                    await asyncio.wait({getter, fatal}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    if not getter.done():
                        getter.cancel()
                if getter.cancelled() or not getter.done():
                    fatal.result()  # raises the producer's fatal exception (or CancelledError)
                yield getter.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if fatal.done() and not fatal.cancelled():
                fatal.exception()  # mark retrieved: no "exception was never retrieved" noise
            else:
                fatal.cancel()

        if delete_orphans:
            for sid in state:
                if sid not in request.live_source_ids:
                    yield SourceOutcome(kind="orphan", source_id=sid)

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def plan(self, request: SyncPlanInput, delete_orphans: bool = False) -> ChangeSet[T]:
        """Buffer the event stream into a reviewable ChangeSet (no store writes).

        ``changed`` outcomes land in ``to_add``/``to_update``; ``orphan`` outcomes in
        ``to_delete``; skips are dropped; resolution (``pre_failed``) and production
        errors are logged (one summary warning) and their sources omitted.
        """
        changeset: ChangeSet[T] = ChangeSet()
        failed: list[str] = [sid for sid, _ in request.pre_failed]
        # aclosing: the stream's cleanup (cancel + gather producers) runs before we leave,
        # not whenever the abandoned generator happens to be finalized.
        async with contextlib.aclosing(self._stream(request, delete_orphans)) as stream:
            async for outcome in stream:
                if outcome.kind == "changed":
                    if outcome.change is None:  # pragma: no cover - _evaluate always sets it
                        raise AssertionError("changed outcome without a SourceChange")
                    (changeset.to_add if outcome.is_new else changeset.to_update).append(outcome.change)
                elif outcome.kind == "orphan":
                    changeset.to_delete.append(outcome.source_id)
                elif outcome.kind == "error":
                    failed.append(outcome.source_id)
        if failed:
            logger.warning(f"plan(): {len(failed)} source(s) failed during processing: {failed}")
        return changeset

    async def run(self, request: SyncPlanInput, delete_orphans: bool = False) -> UpdateResult:
        """Consume the event stream, writing each changed source as it arrives.

        Per source: ``pre_write(items)`` runs first (a failure there leaves the store
        untouched for that source), then the source is written — see the module docstring
        for the two write paths and what a failed write leaves behind. Every failure is
        recorded in ``errors`` and never blocks other sources. Orphan outcomes are deleted
        as they arrive. The corpus is **not** all-or-nothing: a completed source stays
        durable even if a later one fails.

        Memory: producers are back-pressured by the writer, so at most ``concurrency + 2``
        sources' items are alive at once (see :meth:`_stream`); a written source's items are
        released before the next one is taken.
        """
        result = UpdateResult()
        result.skipped.extend(request.pre_skipped)
        result.errors.extend(request.pre_failed)
        try:
            # aclosing: on an abort the producers are cancelled and awaited before run()
            # returns control, instead of lingering until the generator is finalized.
            async with contextlib.aclosing(self._stream(request, delete_orphans)) as stream:
                async for outcome in stream:
                    if outcome.kind == "skipped":
                        result.skipped.append(outcome.source_id)
                    elif outcome.kind == "error":
                        if outcome.error is None:  # pragma: no cover - _evaluate always sets it
                            raise AssertionError("error outcome without an exception")
                        result.errors.append((outcome.source_id, outcome.error))
                    elif outcome.kind == "orphan":
                        await self._delete_orphan(outcome.source_id, result)
                    else:  # "changed"
                        await self._write_change(outcome, result)
                    del outcome  # release the written items before waiting for the next source
        except BaseException:
            # A fatal exception (e.g. asyncio.CancelledError) escaped the stream —
            # collected per-source errors do not reach here; this is the abort path.
            logger.exception("Run failed before completion")
            raise
        logger.info(
            f"run complete: {len(result.processed)} processed, {len(result.skipped)} skipped, "
            f"{len(result.deleted)} deleted, {len(result.errors)} errors"
        )
        return result

    async def apply(self, changeset: ChangeSet[T]) -> UpdateResult:
        """Write an already-buffered *changeset* to the store (whole-corpus pre-write first).

        ``pre_write`` runs over **all** items of ``to_add`` + ``to_update`` before any store
        mutation: if it raises, the store is left untouched and every changed source is
        reported in ``errors``. Then ``to_delete`` is processed, and finally each source is
        written — additions by ``upsert``, updates as described in the module docstring
        (``replace_source`` where the store offers it, else ``delete_by_source`` then
        ``upsert``; a source whose stale-delete failed is **not** upserted — no old/new
        mixing).

        *changeset* itself is **never mutated**: when a ``pre_write`` hook is configured it
        runs on copies of the items, so the changeset still equals its saved copy afterwards
        and can be applied again (by this or another pipeline) with the same result as a
        freshly loaded one. Vectors computed by one ``apply()`` are therefore not kept on the
        plan — every ``apply()`` embeds again.

        Returns:
            :class:`UpdateResult` (``skipped`` is always empty — apply cannot know
            plan-time skips).
        """
        result = UpdateResult()
        changes = [(sc, True) for sc in changeset.to_add] + [(sc, False) for sc in changeset.to_update]

        if self._pre_write is not None:
            # pre_write mutates its items (embedding). Work on copies so the caller's changeset
            # stays exactly the plan that was reviewed / saved — see the docstring.
            changes = [
                (sc.model_copy(update={"items": [item.model_copy(deep=True) for item in sc.items]}), is_new)
                for sc, is_new in changes
            ]
            all_items = [item for sc, _ in changes for item in sc.items]
            try:
                await self._pre_write(all_items)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("apply(): pre-write failed; store left untouched.")
                for sc, _ in changes:
                    result.errors.append((sc.source_id, exc))
                return result

        for sid in changeset.to_delete:
            await self._delete_orphan(sid, result)

        for sc, is_new in changes:
            try:
                await self._write_source(sc, is_new)
                result.processed.append(sc.source_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"apply(): write failed for {sc.source_id}: {exc!r}")
                result.errors.append((sc.source_id, exc))

        logger.info(
            f"apply(): {len(result.processed)} processed, {len(result.deleted)} deleted, {len(result.errors)} errors"
        )
        return result

    # ------------------------------------------------------------------
    # write helpers
    # ------------------------------------------------------------------

    async def _write_source(self, change: SourceChange[T], is_new: bool) -> None:
        """Write one source's items to the store (no ``pre_write``, no error handling).

        * new source — ``upsert`` (nothing stored to replace);
        * update with no items — ``delete_by_source`` (the source now yields nothing);
        * update, store is a :class:`SourceReplaceStore` — ``replace_source`` (new items
          land before stale ones are removed; what a failure leaves is the store's contract,
          but the source must not look "unchanged");
        * update, any other store — ``delete_by_source`` then ``upsert``. A failed delete
          skips the upsert; a failed upsert leaves the source **empty** until the next run
          re-processes it as new.
        """
        if is_new:
            if change.items:
                await self._store.upsert(change.items)
        elif not change.items:
            await self._store.delete_by_source(change.source_id)
        elif isinstance(self._store, SourceReplaceStore):
            replacer = cast("SourceReplaceStore[T]", self._store)
            await replacer.replace_source(change.source_id, change.items)
        else:
            await self._store.delete_by_source(change.source_id)
            await self._store.upsert(change.items)

    async def _write_change(self, outcome: SourceOutcome[T], result: UpdateResult) -> None:
        """Write one changed source for :meth:`run`: ``pre_write`` → :meth:`_write_source`.

        Any ``Exception`` is recorded on *result* for this source only.
        """
        if outcome.change is None:  # pragma: no cover - _evaluate always sets it
            raise AssertionError("changed outcome without a SourceChange")
        change = outcome.change
        try:
            if self._pre_write is not None:
                await self._pre_write(change.items)
            await self._write_source(change, outcome.is_new)
            result.processed.append(change.source_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"run(): write failed for {change.source_id}: {exc!r}")
            result.errors.append((change.source_id, exc))

    async def _delete_orphan(self, source_id: str, result: UpdateResult) -> None:
        """Delete one orphan from the store, recording the outcome on *result*."""
        try:
            await self._store.delete_by_source(source_id)
            result.deleted.append(source_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"delete orphan {source_id} failed: {exc!r}")
            result.errors.append((source_id, exc))
