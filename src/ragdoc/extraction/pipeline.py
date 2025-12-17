"""MentionStorePipeline: incremental file-or-DocumentStore -> MentionStore sync.

Two modes, one constructor each (so an illegal configuration is a ``TypeError`` at
construction, not a runtime rejection):

* **Direct mode** (``MentionStorePipeline(ingest=..., ...)``) — parses + processes source
  files via an :class:`~ragdoc.pipeline.linear.IngestPipeline`, splits each into
  context-sized sub-documents, runs the :class:`~ragdoc.extraction.extractor.Extractor` on
  every split (a direct typed channel — ``extract()`` returns
  :class:`~ragdoc.extraction.mention.Mention` objects; nothing rides on document metadata),
  and syncs them into a :class:`~ragdoc.extraction.stores.MentionStore`. Change detection
  uses the file-byte ``source_hash``: an unchanged file is **skipped with no LLM call**.
  An ``IngestPipeline`` cannot carry a chunker (chunking is meaningless for mention
  extraction), so that misconfiguration is unexpressible.
* **Boundary-2 mode** (:meth:`MentionStorePipeline.from_document_store`) — reads
  already-parsed-and-processed Documents from the DocumentStore by ``source_id``, splits +
  extracts. There is no ingest parameter at all, so re-running the (destructive) processor
  chain or a custom parser on stored documents is unexpressible. Change detection uses the
  Document's ``content_hash``. This mirrors
  :meth:`~ragdoc.pipeline.vectorstore.VectorStorePipeline.from_document_store` so a user
  running both a vector store and a knowledge graph over the same corpus shares the existing
  ``DocumentStore`` substrate without a new boundary.

Shape mirrors the sync pipelines (all compose the shared
:class:`~ragdoc.pipeline.sync.SyncEngine`):

* :meth:`run` — streaming, **per source**. Each source is atomic; a completed source stays
  durable if a later one fails.
* :meth:`plan` / :meth:`apply` — the reviewable path, returning a serializable
  :class:`~ragdoc.pipeline.changeset.ChangeSet` of mentions
  (``ChangeSet[Mention[P]]``; load with ``ChangeSet[Mention[P]].load(path)``).

A source's mentions share one ``content_hash`` (the parent Document's), stamped uniformly via
:func:`~ragdoc.extraction.mention.finalize_mention` with a **running per-source ordinal** across
splits (identical payloads in different splits mint distinct ids); edge-carrying payloads have
their endpoint refs rewritten through the finalization old → new id map
(:func:`~ragdoc.extraction.mention.rewrite_edge_refs`), so KG edges never dangle. This pipeline
emits **mentions, not chunks** — it never touches a vector store.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, cast

from ragdoc.extraction.extractor import Extractor
from ragdoc.extraction.mention import Mention, PayloadT, finalize_mention, rewrite_edge_refs
from ragdoc.pipeline.changeset import ChangeSet, SourceChange
from ragdoc.pipeline.splitter import TokenSplitter
from ragdoc.pipeline.sync import (
    SyncEngine,
    SyncPlanInput,
    SyncSource,
    UpdateResult,
    build_current_map,
    content_hash_token,
    file_hash,
    source_hash_token,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import Document
    from ragdoc.extraction.stores import MentionStore
    from ragdoc.pipeline.linear import IngestPipeline
    from ragdoc.pipeline.stores import DocumentStore
    from ragdoc.splitting.base import Splitter


class MentionStorePipeline(Generic[PayloadT]):
    """Incremental file-or-DocumentStore -> MentionStore sync for one payload type.

    Two modes, one constructor each:

    * **Direct** (``MentionStorePipeline(...)``): parse + process + split + extract per
      file. Change detection uses the file-byte ``source_hash``.
    * **Boundary-2** (:meth:`from_document_store`): split + extract on Documents loaded
      from the DocumentStore by ``source_id``. Processors and a custom parser have no
      parameter to arrive through — processing already happened at Boundary 1 (re-running
      it is destructive), and the source is the store, not a file.

    Args:
        ingest: :class:`~ragdoc.pipeline.linear.IngestPipeline` providing the parser,
            pre-split processors, and ``source_id_fn`` (single source of truth for
            identity). An ``IngestPipeline`` cannot carry a chunker (chunking is
            meaningless for mention extraction).
        extractor: the :class:`~ragdoc.extraction.extractor.Extractor` run on each split
            (e.g. ``StructuredExtractor`` or ``KnowledgeGraphExtractor``); its ``extract()``
            returns the typed mentions directly.
        mention_store: target :class:`MentionStore`.
        splitter: ``Document -> list[Document]`` splitter run before extraction. ``None`` uses
            :class:`~ragdoc.pipeline.splitter.TokenSplitter` with its documented defaults,
            resolved once at construction.
        hash_fn: ``Path -> str`` file-byte change-detection hash (default
            :func:`~ragdoc.pipeline.sync.file_hash`, SHA-256 of bytes).
        concurrency: max sources processed concurrently.

    Raises:
        TypeError: if *extractor* does not implement the ``Extractor`` protocol.
    """

    def __init__(
        self,
        ingest: IngestPipeline,
        extractor: Extractor[PayloadT],
        mention_store: MentionStore,
        splitter: Splitter | None = None,
        hash_fn: Callable[[Path], str] = file_hash,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        self._wire(
            ingest=ingest,
            extractor=extractor,
            mention_store=mention_store,
            document_store=None,
            splitter=splitter,
            hash_fn=hash_fn,
            concurrency=concurrency,
        )

    @classmethod
    def from_document_store(
        cls,
        extractor: Extractor[PayloadT],
        mention_store: MentionStore,
        document_store: DocumentStore,
        splitter: Splitter | None = None,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> MentionStorePipeline[PayloadT]:
        """Boundary-2 constructor: split + extract Documents read from *document_store*.

        There is deliberately no ingest/parser/processors parameter: Documents in the store
        were parsed and processed at Boundary 1
        (:class:`~ragdoc.pipeline.document_store_pipeline.DocumentStorePipeline`), and
        re-running the processor chain is destructive.

        Args:
            extractor: the :class:`~ragdoc.extraction.extractor.Extractor` run on each split.
            mention_store: target :class:`MentionStore`.
            document_store: source :class:`~ragdoc.pipeline.stores.DocumentStore` holding
                already-parsed-and-processed Documents.
            splitter: ``Document -> list[Document]`` splitter run before extraction
                (``None`` uses :class:`~ragdoc.pipeline.splitter.TokenSplitter` defaults).
            concurrency: max sources processed concurrently.

        Returns:
            A Boundary-2 ``MentionStorePipeline``; its :meth:`plan` / :meth:`run` take
            ``source_id`` strings (or ``None`` for every document in the store).
        """
        self = cls.__new__(cls)
        self._wire(
            ingest=None,
            extractor=extractor,
            mention_store=mention_store,
            document_store=document_store,
            splitter=splitter,
            hash_fn=file_hash,
            concurrency=concurrency,
        )
        return self

    def _wire(
        self,
        *,
        ingest: IngestPipeline | None,
        extractor: Extractor[PayloadT],
        mention_store: MentionStore,
        document_store: DocumentStore | None,
        splitter: Splitter | None,
        hash_fn: Callable[[Path], str],
        concurrency: int | asyncio.Semaphore,
    ) -> None:
        """Shared initialisation for both constructors."""
        if not isinstance(extractor, Extractor):
            raise TypeError("extractor must implement the Extractor protocol: async extract(document) -> list[Mention]")
        self._ingest = ingest
        self._extractor = extractor
        self._store = mention_store
        self._document_store = document_store
        # Resolved once at construction — the documented default splitter, not a per-document rebuild.
        self._splitter: Splitter = splitter if splitter is not None else TokenSplitter()
        self._hash_fn = hash_fn
        self._concurrency = concurrency
        self._engine: SyncEngine[Mention[PayloadT]] = SyncEngine(
            store=mention_store,
            producer=self._produce_from_doc_store if document_store is not None else self._produce_from_path,
            token_of=content_hash_token if document_store is not None else source_hash_token,
            concurrency=concurrency,
        )

    @property
    def _source_id_fn(self) -> Callable[[Path], str]:
        if self._ingest is None:  # pragma: no cover - only read on the direct path
            raise AssertionError("source_id_fn is a direct-mode concern; Boundary 2 has no paths")
        return self._ingest.source_id_fn

    # ------------------------------------------------------------------
    # extraction primitives
    # ------------------------------------------------------------------

    async def _extract_from_document(self, parent: Document) -> list[Mention[PayloadT]]:
        """Split + extract on an already-parsed-and-processed Document.

        The Document must carry ``source_id`` and ``source_hash`` (caller's responsibility;
        ``source_hash`` may honestly be ``None`` — it is never faked from the content hash).
        Source-of-truth path: both file-mode and Boundary-2 mode delegate here.

        Splits are extracted concurrently, then finalized **in split order** with a running
        per-source ordinal (so identical payloads in different splits mint distinct
        ``mention_id``\\ s) while recording the old → new mention-id map; every edge-carrying
        payload's refs are rewritten through that map, so KG edges keep pointing at the
        finalized node mention ids even though finalization re-mints them with the parent's
        ``content_hash``.
        """
        source_id = parent.source_id or parent.source_path or parent.id
        source_hash = parent.source_hash
        content_hash = parent.content_hash()

        # Splitting renders per element/group to measure token budgets — sync CPU/pandoc work,
        # so hop off the event loop at this orchestration boundary (see IngestPipeline).
        splits = await asyncio.to_thread(self._splitter, parent)
        for split in splits:
            split.source_id = source_id
            split.source_hash = source_hash
        results = await asyncio.gather(*(self._extractor.extract(split) for split in splits), return_exceptions=True)
        for res in results:
            if isinstance(res, BaseException):
                raise res  # first failing split fails the source, as before

        out: list[Mention[PayloadT]] = []
        id_map: dict[str, str] = {}
        ordinal = 0
        for split_mentions in cast("list[list[Mention[PayloadT]]]", results):
            for mention in split_mentions:
                old_id = mention.mention_id
                mention.ordinal = ordinal
                ordinal += 1
                finalize_mention(mention, content_hash=content_hash, source_id=source_id, source_hash=source_hash)
                id_map[old_id] = mention.mention_id
                out.append(mention)
        for mention in out:
            rewrite_edge_refs(mention.payload, id_map)
        return out

    async def _extract_source(self, path: Path, source_id: str, file_hash: str) -> list[Mention[PayloadT]]:
        """Parse → process → split → extract one source file."""
        if self._ingest is None:  # pragma: no cover - wired only in direct mode
            raise AssertionError("_extract_source called without an IngestPipeline")
        parent: Document | None = await self._ingest.run(path)
        if parent is None:
            logger.info(f"MentionStorePipeline: document filtered out: {path.name}")
            return []
        parent.source_id = source_id
        parent.source_hash = file_hash
        return await self._extract_from_document(parent)

    # ------------------------------------------------------------------
    # producers (per-source production for the engine)
    # ------------------------------------------------------------------

    async def _produce_from_path(self, src: SyncSource) -> SourceChange[Mention[PayloadT]] | None:
        """Direct mode: parse → process → split → extract one file into a mention change."""
        if src.path is None:  # pragma: no cover - direct-mode resolution always sets it
            raise AssertionError("direct-mode SyncSource without a path")
        mentions = await self._extract_source(src.path, src.source_id, src.change_token)
        return SourceChange(
            source_id=src.source_id,
            source_hash=src.change_token,
            content_hash=mentions[0].content_hash if mentions else None,
            items=mentions,
        )

    async def _produce_from_doc_store(self, src: SyncSource) -> SourceChange[Mention[PayloadT]] | None:
        """Boundary-2 mode: split + extract a Document loaded from the DocumentStore."""
        if self._document_store is None:  # pragma: no cover - wired only when set
            raise AssertionError("_produce_from_doc_store called without a document_store")
        doc = await self._document_store.get_document(src.source_id)
        if doc is None:
            return None  # vanished between list_source_state and get_document → engine warns + skips
        mentions = await self._extract_from_document(doc)
        return SourceChange(
            source_id=src.source_id,
            source_hash=doc.source_hash or "",
            content_hash=mentions[0].content_hash if mentions else doc.content_hash(),
            items=mentions,
        )

    # ------------------------------------------------------------------
    # resolution (request → SyncPlanInput)
    # ------------------------------------------------------------------

    async def _resolve(self, sources: Iterable[Path] | Iterable[str] | None) -> SyncPlanInput:
        """Resolve paths (direct) or source_ids (Boundary 2) into a :class:`SyncPlanInput`."""
        if self._document_store is None:
            if sources is None:
                raise ValueError("Direct mode (no document_store) requires `sources` paths.")
            current = await build_current_map(
                cast("Iterable[Path]", sources), self._source_id_fn, self._hash_fn, self._concurrency
            )
            return SyncPlanInput(
                sources=[
                    SyncSource(source_id=sid, change_token=digest, path=path) for sid, (path, digest) in current.items()
                ],
                # Failed ids stay in live_source_ids: an unreadable file is not an orphan.
                live_source_ids=frozenset(current) | frozenset(current.errors),
                pre_failed=tuple(current.errors.items()),
            )

        doc_state = await self._document_store.list_source_state()
        targets = [str(sid) for sid in sources] if sources is not None else list(doc_state)
        resolved: list[SyncSource] = []
        pre_skipped: list[str] = []
        for sid in targets:
            entry = doc_state.get(sid)
            if entry is None:
                logger.warning(f"sync: source_id {sid!r} not in document store; skipping")
                pre_skipped.append(sid)
                continue
            resolved.append(SyncSource(source_id=sid, change_token=entry.content_hash or ""))
        # Orphans are compared against the FULL document store (not the target subset).
        return SyncPlanInput(sources=resolved, live_source_ids=frozenset(doc_state), pre_skipped=tuple(pre_skipped))

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def run(
        self,
        sources: Iterable[Path] | Iterable[str] | None = None,
        delete_orphans: bool = False,
    ) -> UpdateResult:
        """Streaming, per-source sync.

        * **Direct**: *sources* is an iterable of file ``Path``\\ s (required). Unchanged
          files (by file-byte ``source_hash``) are skipped with no LLM call.
        * **Boundary-2** (:meth:`from_document_store`): *sources* is an iterable of
          ``source_id`` strings, or ``None`` for every document in the store. Unchanged
          Documents (by ``content_hash``) are skipped with no LLM call.
        """
        return await self._engine.run(await self._resolve(sources), delete_orphans)

    async def plan(
        self,
        sources: Iterable[Path] | Iterable[str] | None = None,
        delete_orphans: bool = False,
    ) -> ChangeSet[Mention[PayloadT]]:
        """Compute the mention change set without touching the mention store.

        * **Direct**: *sources* is an iterable of file ``Path``\\ s.
        * **Boundary-2** (:meth:`from_document_store`): *sources* is an iterable of
          ``source_id`` strings, or ``None`` for every document in the store.

        Returns:
            A ``ChangeSet[Mention[P]]``; reload from disk with
            ``ChangeSet[Mention[P]].load(path)`` (the concrete parametrization).
        """
        return await self._engine.plan(await self._resolve(sources), delete_orphans)

    async def apply(self, changeset: ChangeSet[Mention[PayloadT]]) -> UpdateResult:
        """Write *changeset* to the mention store (delete orphans, delete stale, then upsert).

        Updates go delete-then-upsert with per-source error isolation: a source whose
        stale-delete failed is recorded in ``errors`` and **not** upserted (no old/new mixing).
        """
        return await self._engine.apply(changeset)
