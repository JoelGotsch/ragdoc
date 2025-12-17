"""DocumentStorePipeline: incremental source-files -> DocumentStore sync (Boundary 1).

Parses and processes source files into :class:`~ragdoc.document.Document` objects and
syncs them into a :class:`~ragdoc.pipeline.stores.DocumentStore`, so they can be edited
and later re-chunked by a :class:`~ragdoc.pipeline.vectorstore.VectorStorePipeline`
(Boundary 2) without re-parsing.

Same plan/apply shape as ``VectorStorePipeline`` but the payload is ``Document`` and there is
no embedding step.  Change detection at this boundary uses the file-byte ``source_hash``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic

from ragdoc.metadata import TMetadata
from ragdoc.pipeline.changeset import ChangeSet, SourceChange
from ragdoc.pipeline.vectorstore import UpdateResult, _file_hash
from ragdoc.processing._concurrency import _resolve_semaphore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import Document
    from ragdoc.pipeline.linear import DocumentPipeline
    from ragdoc.pipeline.stores import DocumentStore


class DocumentStorePipeline(Generic[TMetadata]):
    """Incremental file -> DocumentStore sync via plan/apply.

    This boundary stores **whole Documents, one per source** (via ``parse_and_process``), so a
    splitter or non-default chunker on the given :class:`DocumentPipeline` is not "ignored" — it
    has nowhere to write its output (splits would collide on ``source_id``; chunks aren't
    ``Document``\\ s) and is therefore rejected at construction. Split and chunk at Boundary 2
    (:class:`~ragdoc.pipeline.vectorstore.VectorStorePipeline`) instead.

    Args:
        pipeline: :class:`~ragdoc.pipeline.linear.DocumentPipeline` providing the parser,
            processors, and ``source_id_fn`` (single source of truth for identity).  Must **not**
            carry a splitter or a non-default chunker (raises ``ValueError`` otherwise).
        document_store: target :class:`~ragdoc.pipeline.stores.DocumentStore`.
        hash_fn: ``Path -> str`` file-byte change-detection hash (default SHA-256 of bytes).
        concurrency: max sources processed concurrently in :meth:`plan`.

    Raises:
        ValueError: If *pipeline* has a splitter or a non-default chunker configured.
    """

    def __init__(
        self,
        pipeline: DocumentPipeline[TMetadata],
        document_store: DocumentStore,
        hash_fn: Callable[[Path], str] = lambda p: _file_hash(p),
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        misplaced: list[str] = []
        if pipeline.has_splitter:
            misplaced.append("a splitter")
        if pipeline.has_custom_chunker:
            misplaced.append("a chunker")
        if misplaced:
            raise ValueError(
                f"DocumentStorePipeline (Boundary 1) stores whole Documents, one per source, and "
                f"cannot use {' or '.join(misplaced)} on its DocumentPipeline — splits would "
                f"collide on source_id and chunks aren't Documents. Remove it and split/chunk at "
                f"Boundary 2 (VectorStorePipeline with a document_store) instead."
            )

        self._pipeline = pipeline
        self._document_store = document_store
        self._hash_fn = hash_fn
        self._concurrency = concurrency

    @property
    def _source_id_fn(self) -> Callable[[Path], str]:
        return self._pipeline.source_id_fn

    async def plan(self, sources: Iterable[Path], delete_orphans: bool = False) -> ChangeSet[Document]:
        """Compute the Document change set for *sources* without touching the store."""
        changeset, _skipped, errors = await self._plan_internal(sources, delete_orphans)
        if errors:
            logger.warning(f"plan(): {len(errors)} source(s) failed: {[sid for sid, _ in errors]}")
        return changeset

    async def _plan_internal(
        self, sources: Iterable[Path], delete_orphans: bool
    ) -> tuple[ChangeSet[Document], list[str], list[tuple[str, BaseException]]]:
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

        store_state = await self._document_store.list_source_state()
        logger.info(f"plan(): {len(current)} sources, {len(store_state)} known in doc store")

        to_add: list[SourceChange[Document]] = []
        to_update: list[SourceChange[Document]] = []
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
                    doc = await self._pipeline.parse_and_process(path)
                    if doc is None:
                        logger.info(f"Document filtered out: {path.name}")
                        return
                    doc.source_id = sid
                    doc.source_hash = file_hash
                    change: SourceChange[Document] = SourceChange(
                        source_id=sid,
                        source_hash=file_hash,
                        content_hash=doc.content_hash(),
                        items=[doc],
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

    async def apply(self, changeset: ChangeSet[Document]) -> UpdateResult:
        """Write *changeset* to the document store (delete orphans, then upsert per source).

        ``upsert`` is keyed on ``source_id`` and replaces any existing document, so updates
        need no separate stale-delete.
        """
        result = UpdateResult()

        for sid in changeset.to_delete:
            try:
                await self._document_store.delete_by_source(sid)
                result.deleted.append(sid)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error(f"apply(): delete {sid} failed: {exc!r}", exc_info=True)
                result.errors.append((sid, exc))

        for sc in list(changeset.to_add) + list(changeset.to_update):
            try:
                await self._document_store.upsert(sc.items)
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

    async def run(self, sources: Iterable[Path], delete_orphans: bool = False) -> UpdateResult:
        """Convenience: ``apply(plan(sources, delete_orphans))`` with full reporting."""
        changeset, skipped, plan_errors = await self._plan_internal(sources, delete_orphans)
        result = await self.apply(changeset)
        result.skipped = skipped
        result.errors = plan_errors + result.errors
        logger.info(
            f"run complete: {len(result.processed)} processed, {len(result.skipped)} skipped, "
            f"{len(result.deleted)} deleted, {len(result.errors)} errors"
        )
        return result
