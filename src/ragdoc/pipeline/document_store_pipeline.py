"""DocumentStorePipeline: incremental source-files -> DocumentStore sync (Boundary 1).

Parses and processes source files into :class:`~ragdoc.document.Document` objects and
syncs them into a :class:`~ragdoc.pipeline.stores.DocumentStore`, so they can be edited
and later re-chunked by a Boundary-2
:meth:`~ragdoc.pipeline.vectorstore.VectorStorePipeline.from_document_store` pipeline
without re-parsing.

Same plan/apply/run shape as ``VectorStorePipeline`` (both compose the shared
:class:`~ragdoc.pipeline.sync.SyncEngine`) but the payload is ``Document`` and there is
no embedding step.  Change detection at this boundary uses the file-byte ``source_hash``.

:meth:`DocumentStorePipeline.run` is **streaming**: each source is written as soon as it is
produced (per-source durability), instead of buffering the whole corpus first. A source
whose Document is filtered out by processing (processor returns ``None``) yields an empty
change — it is counted ``processed`` and any stale stored Document for it is deleted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic

from ragdoc.metadata import TMetadata
from ragdoc.pipeline.changeset import ChangeSet, SourceChange
from ragdoc.pipeline.sync import (
    SyncEngine,
    SyncPlanInput,
    SyncSource,
    UpdateResult,
    build_current_map,
    file_hash,
    source_hash_token,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import Document
    from ragdoc.pipeline.linear import IngestPipeline
    from ragdoc.pipeline.stores import DocumentStore


class DocumentStorePipeline(Generic[TMetadata]):
    """Incremental file -> DocumentStore sync via plan/apply/run.

    This boundary stores **whole Documents, one per source**, so it takes an
    :class:`~ragdoc.pipeline.linear.IngestPipeline` (parse → process) — the
    boundary-appropriate type.  An ``IngestPipeline`` cannot carry a splitter or a
    chunker (splits would collide on ``source_id``; chunks aren't ``Document``\\ s), so a
    misplaced Boundary-2 stage is a ``TypeError`` at construction rather than a runtime
    rejection.  Split and chunk at Boundary 2
    (:meth:`~ragdoc.pipeline.vectorstore.VectorStorePipeline.from_document_store`) instead.

    Args:
        ingest: :class:`~ragdoc.pipeline.linear.IngestPipeline` providing the parser,
            processors, and ``source_id_fn`` (single source of truth for identity).
        document_store: target :class:`~ragdoc.pipeline.stores.DocumentStore`.
        hash_fn: ``Path -> str`` file-byte change-detection hash (default
            :func:`~ragdoc.pipeline.sync.file_hash`, SHA-256 of the bytes).
        concurrency: max sources processed concurrently.
    """

    def __init__(
        self,
        ingest: IngestPipeline[TMetadata],
        document_store: DocumentStore,
        hash_fn: Callable[[Path], str] = file_hash,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        self._ingest = ingest
        self._document_store = document_store
        self._hash_fn = hash_fn
        self._concurrency = concurrency
        self._engine: SyncEngine[Document] = SyncEngine(
            store=document_store,
            producer=self._produce,
            token_of=source_hash_token,
            concurrency=concurrency,
        )

    @property
    def _source_id_fn(self) -> Callable[[Path], str]:
        return self._ingest.source_id_fn

    async def _produce(self, src: SyncSource) -> SourceChange[Document] | None:
        """Parse + process one file into a single-Document change (empty when filtered)."""
        if src.path is None:  # pragma: no cover - resolution always sets it
            raise AssertionError("DocumentStorePipeline SyncSource without a path")
        doc = await self._ingest.run(src.path)
        if doc is None:
            logger.info(f"Document filtered out: {src.path.name}")
            # Empty change (NOT None): the source still exists, it just yields nothing —
            # counted processed, and any stale stored Document is deleted.
            return SourceChange(source_id=src.source_id, source_hash=src.change_token, content_hash=None, items=[])
        doc.source_id = src.source_id
        doc.source_hash = src.change_token
        return SourceChange(
            source_id=src.source_id,
            source_hash=src.change_token,
            content_hash=doc.content_hash(),
            items=[doc],
        )

    async def _resolve(self, sources: Iterable[Path]) -> SyncPlanInput:
        current = await build_current_map(sources, self._source_id_fn, self._hash_fn, self._concurrency)
        return SyncPlanInput(
            sources=[
                SyncSource(source_id=sid, change_token=digest, path=path) for sid, (path, digest) in current.items()
            ],
            live_source_ids=frozenset(current),
        )

    async def plan(self, sources: Iterable[Path], delete_orphans: bool = False) -> ChangeSet[Document]:
        """Compute the Document change set for *sources* without touching the store."""
        return await self._engine.plan(await self._resolve(sources), delete_orphans)

    async def apply(self, changeset: ChangeSet[Document]) -> UpdateResult:
        """Write *changeset* to the document store (delete orphans, delete stale, then upsert).

        Updates go delete-then-upsert with per-source error isolation: a source whose
        stale-delete failed is recorded in ``errors`` and **not** upserted.
        """
        return await self._engine.apply(changeset)

    async def run(self, sources: Iterable[Path], delete_orphans: bool = False) -> UpdateResult:
        """Streaming, per-source sync: same end state as ``apply(plan(sources, ...))``.

        Each source is written as soon as it is produced (per-source durability); a completed
        source stays durable even if a later one fails.
        """
        return await self._engine.run(await self._resolve(sources), delete_orphans)
