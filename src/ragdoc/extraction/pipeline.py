"""MentionStorePipeline: incremental file-or-DocumentStore -> MentionStore sync.

Two modes, selected by whether a ``document_store`` is provided:

* **Direct mode** (no ``document_store``) — parses + processes source files, splits each into
  context-sized sub-documents, runs the
  :class:`~ragdoc.extraction.processor.StructuredExtractionProcessor` on every split, and
  syncs the harvested :class:`~ragdoc.extraction.mention.Mention`\\ s into a
  :class:`~ragdoc.extraction.stores.MentionStore`. Change detection uses the file-byte
  ``source_hash``: an unchanged file is **skipped with no LLM call**.
* **Boundary-2 mode** (``document_store`` given) — reads already-parsed-and-processed Documents
  from the DocumentStore by ``source_id``, splits + extracts (never re-runs the processor chain —
  doing so on an already-processed document is destructive). Change detection uses the Document's
  ``content_hash``. This mirrors :class:`~ragdoc.pipeline.vectorstore.VectorStorePipeline`'s
  Boundary-2 mode so a user running both a vector store and a knowledge graph over the same
  corpus shares the existing ``DocumentStore`` substrate without a new boundary.

Shape mirrors the sync pipelines:

* :meth:`run` — streaming, **per source**. Each source is atomic; a completed source stays
  durable if a later one fails.
* :meth:`plan` / :meth:`apply` — the reviewable path, returning a serializable
  :class:`~ragdoc.extraction.changeset.MentionChangeSet`.

A source's mentions share one ``content_hash`` (the parent Document's), stamped uniformly via
:func:`~ragdoc.extraction.mention.finalize_mention`. This pipeline emits **mentions, not
chunks** — it never touches a vector store.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic

from ragdoc.extraction.changeset import MentionChangeSet, MentionSourceChange
from ragdoc.extraction.mention import Mention, PayloadT, finalize_mention
from ragdoc.pipeline.vectorstore import UpdateResult, _file_hash
from ragdoc.processing._concurrency import _resolve_semaphore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import Document
    from ragdoc.extraction.processor import StructuredExtractionProcessor
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
        extractor: the :class:`StructuredExtractionProcessor` run on each split.
        mention_store: target :class:`MentionStore`.
        document_store: optional :class:`~ragdoc.pipeline.stores.DocumentStore`. When given,
            the pipeline runs in Boundary-2 mode.
        splitter: ``Document -> list[Document]`` splitter run before extraction. ``None`` builds a
            default token splitter (``split_document`` with a Markdown prompt renderer).
        hash_fn: ``Path -> str`` file-byte change-detection hash (default SHA-256 of bytes). Used
            only in direct mode.
        concurrency: max sources processed concurrently.

    Raises:
        ValueError: if *pipeline* carries a non-default chunker, or — in Boundary-2 mode — if it
            carries processors or a custom parser.
    """

    def __init__(
        self,
        pipeline: DocumentPipeline,
        extractor: StructuredExtractionProcessor[PayloadT],
        mention_store: MentionStore,
        document_store: DocumentStore | None = None,
        splitter: Splitter | None = None,
        hash_fn: Callable[[Path], str] = lambda p: _file_hash(p),
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
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

    def _build_current_map(self, sources: Iterable[Path]) -> dict[str, tuple[Path, str]]:
        """Map ``source_id -> (path, file_hash)``, raising on id collisions (as the sync pipelines do)."""
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
        splits = splitter(parent)
        payload_model = self._extractor.payload_model
        mention_type = Mention[payload_model]

        out: list[Mention[PayloadT]] = []
        for split in splits:
            split.source_id = source_id
            split.source_hash = source_hash
            processed = await self._extractor.process(split)
            for raw in processed.metadata.get(self._extractor.metadata_key, []):
                mention = mention_type.model_validate(raw)
                finalize_mention(mention, content_hash=content_hash, source_id=source_id, source_hash=source_hash)
                out.append(mention)
        return out

    async def _extract_source(self, path: Path, source_id: str, file_hash: str) -> list[Mention[PayloadT]]:
        """Parse → process → split → extract one source file."""
        parent: Document | None = await self._pipeline.parse_and_process(path)
        if parent is None:
            logger.info("MentionStorePipeline: document filtered out: %s", path.name)
            return []
        parent.source_id = source_id
        parent.source_hash = file_hash
        return await self._extract_from_document(parent)

    # ------------------------------------------------------------------
    # run (streaming, per source)
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
        if self._document_store is not None:
            return await self._run_from_doc_store(sources, delete_orphans)  # type: ignore[arg-type]
        if sources is None:
            raise ValueError("Direct mode (no document_store) requires `sources` paths.")
        return await self._run_from_paths(sources, delete_orphans)  # type: ignore[arg-type]

    async def _run_from_paths(self, sources: Iterable[Path], delete_orphans: bool) -> UpdateResult:
        current = self._build_current_map(sources)
        store_state = await self._store.list_source_state()
        logger.info("run(): %d sources, %d known in mention store", len(current), len(store_state))

        result = UpdateResult()
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str, path: Path, file_hash: str) -> None:
            async with sem:
                try:
                    existing = store_state.get(sid)
                    if existing is not None and existing.source_hash == file_hash:
                        result.skipped.append(sid)
                        return
                    mentions = await self._extract_source(path, sid, file_hash)
                    if existing is not None:
                        await self._store.delete_by_source(sid)
                    if mentions:
                        await self._store.upsert(mentions)
                    result.processed.append(sid)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error("run() failed for %s: %r", path.name, exc, exc_info=True)
                    result.errors.append((sid, exc))

        await asyncio.gather(*[_process(sid, p, h) for sid, (p, h) in current.items()])

        if delete_orphans:
            await self._delete_orphans([sid for sid in store_state if sid not in current], result)
        logger.info(
            "run complete: %d processed, %d skipped, %d deleted, %d errors",
            len(result.processed),
            len(result.skipped),
            len(result.deleted),
            len(result.errors),
        )
        return result

    async def _run_from_doc_store(self, source_ids: Iterable[str] | None, delete_orphans: bool) -> UpdateResult:
        assert self._document_store is not None
        doc_state = await self._document_store.list_source_state()
        store_state = await self._store.list_source_state()
        targets = list(source_ids) if source_ids is not None else list(doc_state.keys())
        logger.info("run(): %d doc-store sources, %d in mention store", len(targets), len(store_state))

        result = UpdateResult()
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str) -> None:
            async with sem:
                try:
                    doc_entry = doc_state.get(sid)
                    if doc_entry is None:
                        logger.warning("run(): source_id %r not in document store; skipping", sid)
                        return
                    existing = store_state.get(sid)
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
                    mentions = await self._extract_from_document(doc)
                    if existing is not None:
                        await self._store.delete_by_source(sid)
                    if mentions:
                        await self._store.upsert(mentions)
                    result.processed.append(sid)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error("run() failed for %s: %r", sid, exc, exc_info=True)
                    result.errors.append((sid, exc))

        await asyncio.gather(*[_process(sid) for sid in targets])

        if delete_orphans:
            # Orphans compared against the FULL document store (not the target subset).
            await self._delete_orphans([sid for sid in store_state if sid not in doc_state], result)
        logger.info(
            "run complete: %d processed, %d skipped, %d deleted, %d errors",
            len(result.processed),
            len(result.skipped),
            len(result.deleted),
            len(result.errors),
        )
        return result

    async def _delete_orphans(self, orphan_ids: list[str], result: UpdateResult) -> None:
        for sid in orphan_ids:
            try:
                await self._store.delete_by_source(sid)
                result.deleted.append(sid)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error("run(): delete orphan %s failed: %r", sid, exc, exc_info=True)
                result.errors.append((sid, exc))

    # ------------------------------------------------------------------
    # plan / apply (reviewable)
    # ------------------------------------------------------------------

    async def plan(
        self,
        sources: Iterable[Path] | Iterable[str] | None = None,
        delete_orphans: bool = False,
    ) -> MentionChangeSet[PayloadT]:
        """Compute the mention change set without touching the mention store.

        * **Direct** (no ``document_store``): *sources* is an iterable of file ``Path``\\ s.
        * **Boundary-2** (``document_store`` given): *sources* is an iterable of ``source_id``
          strings, or ``None`` for every document in the store.
        """
        changeset, _skipped, errors = await self._plan_internal(sources, delete_orphans)
        if errors:
            logger.warning("plan(): %d source(s) failed: %s", len(errors), [sid for sid, _ in errors])
        return changeset

    async def _plan_internal(
        self,
        sources: Iterable[Path] | Iterable[str] | None,
        delete_orphans: bool,
    ) -> tuple[MentionChangeSet[PayloadT], list[str], list[tuple[str, BaseException]]]:
        if self._document_store is not None:
            return await self._plan_from_doc_store(sources, delete_orphans)  # type: ignore[arg-type]
        if sources is None:
            raise ValueError("Direct mode (no document_store) requires `sources` paths.")
        return await self._plan_from_paths(sources, delete_orphans)  # type: ignore[arg-type]

    async def _plan_from_paths(
        self, sources: Iterable[Path], delete_orphans: bool
    ) -> tuple[MentionChangeSet[PayloadT], list[str], list[tuple[str, BaseException]]]:
        current = self._build_current_map(sources)
        store_state = await self._store.list_source_state()

        to_add: list[MentionSourceChange[PayloadT]] = []
        to_update: list[MentionSourceChange[PayloadT]] = []
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
                    mentions = await self._extract_source(path, sid, file_hash)
                    content_hash = mentions[0].content_hash if mentions else None
                    change: MentionSourceChange[PayloadT] = MentionSourceChange(
                        source_id=sid, source_hash=file_hash, content_hash=content_hash, items=mentions
                    )
                    (to_add if existing is None else to_update).append(change)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error("plan() failed for %s: %r", path.name, exc, exc_info=True)
                    errors.append((sid, exc))

        await asyncio.gather(*[_process(sid, p, h) for sid, (p, h) in current.items()])
        to_delete = [sid for sid in store_state if sid not in current] if delete_orphans else []
        return MentionChangeSet(to_add=to_add, to_update=to_update, to_delete=to_delete), skipped, errors

    async def _plan_from_doc_store(
        self, source_ids: Iterable[str] | None, delete_orphans: bool
    ) -> tuple[MentionChangeSet[PayloadT], list[str], list[tuple[str, BaseException]]]:
        assert self._document_store is not None
        doc_state = await self._document_store.list_source_state()
        store_state = await self._store.list_source_state()
        targets = list(source_ids) if source_ids is not None else list(doc_state.keys())

        to_add: list[MentionSourceChange[PayloadT]] = []
        to_update: list[MentionSourceChange[PayloadT]] = []
        skipped: list[str] = []
        errors: list[tuple[str, BaseException]] = []
        sem = _resolve_semaphore(self._concurrency)

        async def _process(sid: str) -> None:
            async with sem:
                try:
                    doc_entry = doc_state.get(sid)
                    if doc_entry is None:
                        return
                    existing = store_state.get(sid)
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
                    mentions = await self._extract_from_document(doc)
                    content_hash = mentions[0].content_hash if mentions else doc.content_hash()
                    change: MentionSourceChange[PayloadT] = MentionSourceChange(
                        source_id=sid,
                        source_hash=doc.source_hash or "",
                        content_hash=content_hash,
                        items=mentions,
                    )
                    (to_add if existing is None else to_update).append(change)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.error("plan() failed for %s: %r", sid, exc, exc_info=True)
                    errors.append((sid, exc))

        await asyncio.gather(*[_process(sid) for sid in targets])
        to_delete = [sid for sid in store_state if sid not in doc_state] if delete_orphans else []
        return MentionChangeSet(to_add=to_add, to_update=to_update, to_delete=to_delete), skipped, errors

    async def apply(self, changeset: MentionChangeSet[PayloadT]) -> UpdateResult:
        """Write *changeset* to the mention store (delete orphans, replace updates, upsert)."""
        result = UpdateResult()

        for sid in changeset.to_delete:
            try:
                await self._store.delete_by_source(sid)
                result.deleted.append(sid)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error("apply(): delete %s failed: %r", sid, exc, exc_info=True)
                result.errors.append((sid, exc))

        for sc in changeset.to_update:
            try:
                await self._store.delete_by_source(sc.source_id)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error("apply(): delete stale %s failed: %r", sc.source_id, exc, exc_info=True)
                result.errors.append((sc.source_id, exc))

        for sc in list(changeset.to_add) + list(changeset.to_update):
            try:
                if sc.items:
                    await self._store.upsert(sc.items)
                result.processed.append(sc.source_id)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.error("apply(): upsert %s failed: %r", sc.source_id, exc, exc_info=True)
                result.errors.append((sc.source_id, exc))

        logger.info(
            "apply(): %d processed, %d deleted, %d errors",
            len(result.processed),
            len(result.deleted),
            len(result.errors),
        )
        return result
