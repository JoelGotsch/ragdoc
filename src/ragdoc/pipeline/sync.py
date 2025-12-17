"""Generic plan/apply/run sync engine shared by all sync pipelines (Decision D2-A).

One streaming core drives every sync pipeline (``VectorStorePipeline``,
``DocumentStorePipeline``, ``MentionStorePipeline``):

* :meth:`SyncEngine.plan` buffers the per-source event stream into a serializable
  :class:`~ragdoc.pipeline.changeset.ChangeSet` (no store writes).
* :meth:`SyncEngine.run` consumes the **same** event stream, writing each changed source as
  its event is yielded (per-source ``pre_write`` → delete-then-upsert).
* :meth:`SyncEngine.apply` writes an already-buffered ChangeSet, calling ``pre_write`` once
  over the whole corpus first (embed-first semantics: the store is untouched if it raises).

``run()`` therefore produces the same end state as ``apply(plan(...))`` — same change
detection, same final store contents — differing only in *scheduling*: per-source
``pre_write``-and-write versus whole-corpus ``pre_write``-first.

Unified error discipline: production and write failures are caught per source with
``except Exception`` (never ``BaseException``), so ``asyncio.CancelledError``,
``KeyboardInterrupt``, and ``SystemExit`` always propagate.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

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
"""Per-source item production.

Returns a :class:`~ragdoc.pipeline.changeset.SourceChange` with the source's new content
(``items`` may be empty ⇒ "the source now yields nothing" — its stale stored entry is
deleted), or ``None`` when the source cannot yield a payload this run — the engine logs a
warning and counts it skipped, leaving any stored state untouched. ``None`` covers both
"unavailable" (e.g. it vanished from the upstream store between listing and loading) and
"filtered" (a processor dropped the document): a filtered source **preserves** its stored
copy — filtering is potentially transient (an LLM misclassification must not destroy
durable state), so deletion stays reserved for ``delete_orphans`` and for sources that
genuinely vanish from the input set.
"""

TokenSelector = Callable[[SourceState], "str | None"]
"""Selects the stored comparand from :class:`SourceState`.

``None`` ⇒ "always changed" (pins the stored-``content_hash``-is-``None`` rule).
"""

PreWriteHook = Callable[[list[T]], Awaitable[None]]
"""Pre-write hook, called with the items about to be written (mutates in place — embedding).

:meth:`SyncEngine.run` calls it once per source; :meth:`SyncEngine.apply` calls it once with
the whole changeset's items (embed-first).
"""


def source_hash_token(state: SourceState) -> str | None:
    """Direct-path selector: the stored file-byte hash (``''`` ⇒ unknown ⇒ always changed)."""
    return state.source_hash or None


def content_hash_token(state: SourceState) -> str | None:
    """Boundary-2 selector: the stored ``content_hash`` (``None`` ⇒ always changed)."""
    return state.content_hash


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
        processed: source_ids newly added or updated (items written).
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
            logger.error(f"sync failed for {src.source_id}: {exc!r}", exc_info=True)
            return SourceOutcome(kind="error", source_id=src.source_id, error=exc)

    async def _stream(self, request: SyncPlanInput, delete_orphans: bool) -> AsyncIterator[SourceOutcome[T]]:
        """Yield one outcome per source (as completed), then one orphan outcome per orphan.

        Tasks are created eagerly and drained via ``asyncio.as_completed``; the ``finally``
        block cancels and gathers leftovers so an early abort never abandons tasks.
        """
        state = await self._store.list_source_state()
        logger.info(f"sync: {len(request.sources)} sources, {len(state)} known in store")
        sem = resolve_semaphore(self._concurrency)

        async def _guarded(src: SyncSource) -> SourceOutcome[T]:
            async with sem:
                return await self._evaluate(src, state)

        tasks = [asyncio.create_task(_guarded(src)) for src in request.sources]
        try:
            for next_done in asyncio.as_completed(tasks):
                yield await next_done
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
        async for outcome in self._stream(request, delete_orphans):
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

        Per-source atomicity: ``pre_write(items)`` → (update ⇒ ``delete_by_source``) →
        (items ⇒ ``upsert``); a failure at any step leaves that one source's write
        unapplied and records it in ``errors``. Orphan outcomes are deleted as they
        arrive. The corpus is **not** all-or-nothing: a completed source stays durable
        even if a later one fails.
        """
        result = UpdateResult()
        result.skipped.extend(request.pre_skipped)
        result.errors.extend(request.pre_failed)
        try:
            async for outcome in self._stream(request, delete_orphans):
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
        except BaseException:
            # A fatal exception (e.g. asyncio.CancelledError) escaped the stream —
            # collected per-source errors do not reach here; this is the abort path.
            logger.error("Run failed before completion", exc_info=True)
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
        reported in ``errors``. Then ``to_delete`` is processed, stale entries of updates
        are deleted (a source whose stale-delete failed is **not** upserted — no old/new
        mixing), and finally each source's items are upserted.

        Returns:
            :class:`UpdateResult` (``skipped`` is always empty — apply cannot know
            plan-time skips).
        """
        result = UpdateResult()
        changes = list(changeset.to_add) + list(changeset.to_update)

        if self._pre_write is not None:
            all_items = [item for sc in changes for item in sc.items]
            try:
                await self._pre_write(all_items)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("apply(): pre-write failed; store left untouched.", exc_info=True)
                for sc in changes:
                    result.errors.append((sc.source_id, exc))
                return result

        for sid in changeset.to_delete:
            await self._delete_orphan(sid, result)

        errored: set[str] = set()
        for sc in changeset.to_update:
            try:
                await self._store.delete_by_source(sc.source_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"apply(): delete stale {sc.source_id} failed: {exc!r}", exc_info=True)
                result.errors.append((sc.source_id, exc))
                errored.add(sc.source_id)

        for sc in changes:
            if sc.source_id in errored:
                continue
            try:
                if sc.items:
                    await self._store.upsert(sc.items)
                result.processed.append(sc.source_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"apply(): upsert {sc.source_id} failed: {exc!r}", exc_info=True)
                result.errors.append((sc.source_id, exc))

        logger.info(
            f"apply(): {len(result.processed)} processed, {len(result.deleted)} deleted, {len(result.errors)} errors"
        )
        return result

    # ------------------------------------------------------------------
    # write helpers
    # ------------------------------------------------------------------

    async def _write_change(self, outcome: SourceOutcome[T], result: UpdateResult) -> None:
        """Write one changed source: pre_write → delete-stale (updates) → upsert (atomic per source)."""
        if outcome.change is None:  # pragma: no cover - _evaluate always sets it
            raise AssertionError("changed outcome without a SourceChange")
        change = outcome.change
        try:
            if self._pre_write is not None:
                await self._pre_write(change.items)
            if not outcome.is_new:
                await self._store.delete_by_source(change.source_id)
            if change.items:
                await self._store.upsert(change.items)
            result.processed.append(change.source_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"run(): write failed for {change.source_id}: {exc!r}", exc_info=True)
            result.errors.append((change.source_id, exc))

    async def _delete_orphan(self, source_id: str, result: UpdateResult) -> None:
        """Delete one orphan from the store, recording the outcome on *result*."""
        try:
            await self._store.delete_by_source(source_id)
            result.deleted.append(source_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"delete orphan {source_id} failed: {exc!r}", exc_info=True)
            result.errors.append((source_id, exc))
