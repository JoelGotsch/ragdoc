"""MentionStorePipeline: incremental file-or-DocumentStore -> MentionStore sync.

Two modes, selected by whether a ``document_store`` is provided:

* **Direct mode** (no ``document_store``) — parses + processes source files, splits each into
  context-sized sub-documents, runs the :class:`~ragdoc.extraction.extractor.Extractor` on every
  split (a direct typed channel — ``extract()`` returns
  :class:`~ragdoc.extraction.mention.Mention` objects; nothing rides on document metadata), and
  syncs them into a :class:`~ragdoc.extraction.stores.MentionStore`. Change detection uses the
  file-byte ``source_hash``: an unchanged file is **skipped with no LLM call**.
* **Boundary-2 mode** (``document_store`` given) — reads already-parsed-and-processed Documents
  from the DocumentStore by ``source_id``, splits + extracts (never re-runs the processor chain —
  doing so on an already-processed document is destructive). Change detection uses the Document's
  ``content_hash``. This mirrors :class:`~ragdoc.pipeline.vectorstore.VectorStorePipeline`'s
  Boundary-2 mode so a user running both a vector store and a knowledge graph over the same
  corpus shares the existing ``DocumentStore`` substrate without a new boundary.

Shape mirrors the sync pipelines (all compose the shared
:class:`~ragdoc.pipeline.sync.SyncEngine`):

* :meth:`run` — streaming, **per source**. Each source is atomic; a completed source stays
  durable if a later one fails.
* :meth:`plan` / :meth:`apply` — the reviewable path, returning a serializable
  :class:`~ragdoc.pipeline.changeset.ChangeSet` of mentions
  (``ChangeSet[Mention[P]]``; load with ``ChangeSet[Mention[P]].load(path)``).

A source's mentions share one ``content_hash`` (the parent Document's), stamped uniformly via
:func:`~ragdoc.extraction.mention.finalize_mention`. This pipeline emits **mentions, not
chunks** — it never touches a vector store.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, cast

from ragdoc.extraction.extractor import Extractor
from ragdoc.extraction.mention import Mention, PayloadT, finalize_mention
from ragdoc.pipeline.changeset import ChangeSet, SourceChange
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
    from ragdoc.pipeline.linear import DocumentPipeline
    from ragdoc.pipeline.stores import DocumentStore
    from ragdoc.splitting.base import Splitter


class MentionStorePipeline(Generic[PayloadT]):
    """Incremental file-or-DocumentStore -> MentionStore sync for one payload type.

    Two modes:

    * **Direct** (no ``document_store``): parse + process + split + extract per file.
      Change detection uses the file-byte ``source_hash``.
    * **Boundary-2** (``document_store`` given): split + extract on Documents loaded from the
      DocumentStore by ``source_id``. Processors and a custom parser are **rejected at
      construction** — processing already happened at Boundary 1 (re-running it is destructive),
      and the source is the store, not a file.

    Args:
        pipeline: :class:`~ragdoc.pipeline.linear.DocumentPipeline` providing the parser,
            pre-split processors, and ``source_id_fn`` (single source of truth for identity). A
            non-default chunker is always rejected (chunking is meaningless for mention extraction).
            In Boundary-2 mode, processors and a custom parser are rejected as well.
        extractor: the :class:`~ragdoc.extraction.extractor.Extractor` run on each split
            (e.g. ``StructuredExtractor`` or ``KnowledgeGraphExtractor``); its ``extract()``
            returns the typed mentions directly.
        mention_store: target :class:`MentionStore`.
        document_store: optional :class:`~ragdoc.pipeline.stores.DocumentStore`. When given,
            the pipeline runs in Boundary-2 mode.
        splitter: ``Document -> list[Document]`` splitter run before extraction. ``None`` builds a
            default token splitter (``split_document`` with a Markdown prompt renderer).
        hash_fn: ``Path -> str`` file-byte change-detection hash (default
            :func:`~ragdoc.pipeline.sync.file_hash`, SHA-256 of bytes). Used only in direct mode.
        concurrency: max sources processed concurrently.

    Raises:
        TypeError: if *extractor* does not implement the ``Extractor`` protocol.
        ValueError: if *pipeline* carries a non-default chunker, or — in Boundary-2 mode — if it
            carries processors or a custom parser.
    """

    def __init__(
        self,
        pipeline: DocumentPipeline,
        extractor: Extractor[PayloadT],
        mention_store: MentionStore,
        document_store: DocumentStore | None = None,
        splitter: Splitter | None = None,
        hash_fn: Callable[[Path], str] = file_hash,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        if not isinstance(extractor, Extractor):
            raise TypeError("extractor must implement the Extractor protocol: async extract(document) -> list[Mention]")
        if pipeline.has_custom_chunker:
            raise ValueError(
                "MentionStorePipeline extracts mentions, not chunks; its DocumentPipeline must not "
                "carry a chunker. Configure chunking on a VectorStorePipeline instead."
            )
        if document_store is not None:
            misplaced: list[str] = []
            if pipeline.has_processors:
                misplaced.append("processors (they ran once at Boundary 1; re-running is destructive)")
            if pipeline.has_custom_parser:
                misplaced.append("a custom parser (the source is the document_store, not a file)")
            if misplaced:
                raise ValueError(
                    "A Boundary-2 MentionStorePipeline (document_store given) splits and extracts "
                    "Documents loaded from the store; its DocumentPipeline must not carry "
                    + " or ".join(misplaced)
                    + ". Configure those on the DocumentStorePipeline (Boundary 1) instead."
                )

        self._pipeline = pipeline
        self._extractor = extractor
        self._store = mention_store
        self._document_store = document_store
        self._splitter = splitter
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
        return self._pipeline.source_id_fn

    def _get_splitter(self) -> Splitter:
        if self._splitter is not None:
            return self._splitter
        from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
        from ragdoc.splitting.token import split_document
        from ragdoc.utils import GPTTokenizer

        renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
        tokenizer = GPTTokenizer()
        return lambda document: split_document(document, renderer, tokenizer)

    # ------------------------------------------------------------------
    # extraction primitives
    # ------------------------------------------------------------------

    async def _extract_from_document(self, parent: Document) -> list[Mention[PayloadT]]:
        """Split + extract on an already-parsed-and-processed Document.

        The Document must carry ``source_id`` and ``source_hash`` (caller's responsibility).
        Source-of-truth path: both file-mode and Boundary-2 mode delegate here.
        """
        source_id = parent.source_id or parent.source_path or parent.id
        source_hash = parent.source_hash or ""
        content_hash = parent.content_hash()

        splitter = self._get_splitter()
        out: list[Mention[PayloadT]] = []
        for split in splitter(parent):
            split.source_id = source_id
            split.source_hash = source_hash
            for mention in await self._extractor.extract(split):
                finalize_mention(mention, content_hash=content_hash, source_id=source_id, source_hash=source_hash)
                out.append(mention)
        return out

    async def _extract_source(self, path: Path, source_id: str, file_hash: str) -> list[Mention[PayloadT]]:
        """Parse → process → split → extract one source file."""
        parent: Document | None = await self._pipeline.parse_and_process(path)
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
                live_source_ids=frozenset(current),
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

        * **Direct** (no ``document_store``): *sources* is an iterable of file ``Path``\\ s
          (required). Unchanged files (by file-byte ``source_hash``) are skipped with no LLM call.
        * **Boundary-2** (``document_store`` given): *sources* is an iterable of ``source_id``
          strings, or ``None`` for every document in the store. Unchanged Documents (by
          ``content_hash``) are skipped with no LLM call.
        """
        return await self._engine.run(await self._resolve(sources), delete_orphans)

    async def plan(
        self,
        sources: Iterable[Path] | Iterable[str] | None = None,
        delete_orphans: bool = False,
    ) -> ChangeSet[Mention[PayloadT]]:
        """Compute the mention change set without touching the mention store.

        * **Direct** (no ``document_store``): *sources* is an iterable of file ``Path``\\ s.
        * **Boundary-2** (``document_store`` given): *sources* is an iterable of ``source_id``
          strings, or ``None`` for every document in the store.

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
