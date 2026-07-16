"""VectorStorePipeline: incremental vector-store sync via plan() / apply() / run().

:class:`VectorStorePipeline` composes a pipeline with a
:class:`~ragdoc.pipeline.stores.VectorStore` over the shared
:class:`~ragdoc.pipeline.sync.SyncEngine`.  It serves two boundaries through two
constructors, so an illegal configuration is a ``TypeError`` at construction:

* ``VectorStorePipeline(pipeline=DocumentPipeline(...), vector_store=...)`` — the **direct
  path** (parse → process → split → chunk per file).
* :meth:`VectorStorePipeline.from_document_store` — **Boundary 2**: takes a
  :class:`~ragdoc.pipeline.linear.ChunkPipeline` (split → chunk only) and reads
  already-parsed-and-processed Documents from a
  :class:`~ragdoc.pipeline.stores.DocumentStore`.  A ``ChunkPipeline`` cannot carry
  processors or a parser, so re-running the (destructive) processor chain on stored
  documents is unexpressible.

Two entry points:

* :meth:`run` — the **standard** path. Syncs **per source** (produce → embed →
  delete-then-upsert), writing each source as soon as it is ready. Each source is atomic
  (embed-first per source); the corpus is **not** all-or-nothing — a completed source stays
  durable even if a later one fails. Flat peak memory (one source's chunks at a time).
* :meth:`plan` / :meth:`apply` — the **reviewable** path. :meth:`plan` determines what changed
  and computes chunks for new/modified sources, returning a serializable
  :class:`~ragdoc.pipeline.changeset.ChangeSet` without touching the store (safe to call
  repeatedly). :meth:`apply` embeds the **whole corpus first** (store untouched if embedding
  fails), then writes per source.

:meth:`run` consumes the same event stream :meth:`plan` buffers; it produces the same end
state, differing only in scheduling (per-source embed-and-write versus whole-corpus
embed-first). Use :meth:`plan` / :meth:`apply` when you need to review changes or want
whole-corpus embed-first semantics.

Provenance: each chunk carries ``source_id`` (from the ``IngestPipeline``'s
``source_id_fn``) and ``source_hash`` (file-byte SHA-256, set here via ``hash_fn``).
``chunk.metadata`` is never modified.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, cast

from ragdoc.metadata import TMetadata
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
    from ragdoc.chunking.chunk import Chunk
    from ragdoc.pipeline.embedders import EmbedderConfig
    from ragdoc.pipeline.linear import ChunkPipeline, DocumentPipeline
    from ragdoc.pipeline.stores import DocumentStore, VectorStore


class VectorStorePipeline(Generic[TMetadata]):
    """Incremental vector-store synchronisation via plan/apply/run.

    Two constructors, one per boundary:

    * ``VectorStorePipeline(...)`` — **direct mode**: every stage runs per file
      (parse → process → split → chunk → embed → write).  Change detection uses the
      file-byte ``source_hash``.
    * :meth:`from_document_store` — **Boundary-2 mode**: split + chunk Documents read
      from a ``DocumentStore``.  Change detection uses the Document's ``content_hash``.

    Args:
        pipeline: :class:`~ragdoc.pipeline.linear.DocumentPipeline` for parsing,
            processing, splitting, and chunking.  Its ``source_id_fn`` is the single source
            of truth for source identity (read here for collision/orphan checks).
        vector_store: :class:`~ragdoc.pipeline.stores.VectorStore` implementation.
        embedders: Named embedder configs.  Each key becomes an entry in
            ``chunk.named_embeddings``.  Embedders run concurrently.
        hash_fn: ``Path -> str`` producing the file-byte change-detection hash.  Defaults to
            :func:`~ragdoc.pipeline.sync.file_hash` (SHA-256 of the raw bytes).  Override for
            custom filesystems (e.g. S3) where ``path.read_bytes()`` is unavailable.
        concurrency: Max sources processed concurrently (``int`` or a shared
            ``asyncio.Semaphore``).  Defaults to 10.
    """

    def __init__(
        self,
        pipeline: DocumentPipeline[TMetadata],
        vector_store: VectorStore,
        embedders: dict[str, EmbedderConfig] | None = None,
        hash_fn: Callable[[Path], str] = file_hash,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        self._wire(
            pipeline=pipeline,
            chunk=None,
            vector_store=vector_store,
            document_store=None,
            embedders=embedders,
            hash_fn=hash_fn,
            concurrency=concurrency,
        )

    @classmethod
    def from_document_store(
        cls,
        chunk: ChunkPipeline[TMetadata],
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedders: dict[str, EmbedderConfig] | None = None,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> VectorStorePipeline[TMetadata]:
        """Boundary-2 constructor: chunk + embed Documents read from *document_store*.

        Takes a :class:`~ragdoc.pipeline.linear.ChunkPipeline` — the boundary-appropriate
        type.  Processing already happened at Boundary 1
        (:class:`~ragdoc.pipeline.document_store_pipeline.DocumentStorePipeline`) and a
        ``ChunkPipeline`` cannot carry processors or a parser, so re-running them here is
        unexpressible rather than merely rejected.

        Args:
            chunk: :class:`~ragdoc.pipeline.linear.ChunkPipeline` (splitter + chunker +
                ``chunk_id_fn`` + ``metadata_type``).
            vector_store: Target :class:`~ragdoc.pipeline.stores.VectorStore`.
            document_store: Source :class:`~ragdoc.pipeline.stores.DocumentStore` holding
                already-parsed-and-processed Documents.
            embedders: Named embedder configs (see class docstring).
            concurrency: Max sources processed concurrently.

        Returns:
            A Boundary-2 ``VectorStorePipeline``; its :meth:`plan` / :meth:`run` take
            ``source_id`` strings (or ``None`` for every document in the store).
        """
        self = cls.__new__(cls)
        self._wire(
            pipeline=None,
            chunk=chunk,
            vector_store=vector_store,
            document_store=document_store,
            embedders=embedders,
            hash_fn=file_hash,
            concurrency=concurrency,
        )
        return self

    def _wire(
        self,
        *,
        pipeline: DocumentPipeline[TMetadata] | None,
        chunk: ChunkPipeline[TMetadata] | None,
        vector_store: VectorStore,
        document_store: DocumentStore | None,
        embedders: dict[str, EmbedderConfig] | None,
        hash_fn: Callable[[Path], str],
        concurrency: int | asyncio.Semaphore,
    ) -> None:
        """Shared initialisation for both constructors."""
        self._pipeline = pipeline
        self._chunk = chunk
        self._vector_store = vector_store
        self._embedders = embedders or {}
        self._document_store = document_store
        self._hash_fn = hash_fn
        self._concurrency = concurrency
        self._engine: SyncEngine[Chunk] = SyncEngine(
            store=vector_store,
            producer=self._produce_from_doc_store if document_store is not None else self._produce_from_path,
            token_of=content_hash_token if document_store is not None else source_hash_token,
            pre_write=self._embed_chunks,
            concurrency=concurrency,
        )

    @property
    def _source_id_fn(self) -> Callable[[Path], str]:
        # Single source of truth — the DocumentPipeline's IngestPipeline owns it.
        if self._pipeline is None:  # pragma: no cover - only read on the direct path
            raise AssertionError("source_id_fn is a direct-mode concern; Boundary 2 has no paths")
        return self._pipeline.source_id_fn

    # ------------------------------------------------------------------
    # producers (per-source production for the engine)
    # ------------------------------------------------------------------

    async def _produce_from_path(self, src: SyncSource) -> SourceChange[Chunk] | None:
        """Direct mode: parse → process → split → chunk one file."""
        if src.path is None:  # pragma: no cover - direct-mode resolution always sets it
            raise AssertionError("direct-mode SyncSource without a path")
        if self._pipeline is None:  # pragma: no cover - wired only in direct mode
            raise AssertionError("_produce_from_path called without a DocumentPipeline")
        chunks = await self._pipeline.run(src.path)
        # source_id is already stamped by the pipeline (via doc.source_id, same
        # source_id_fn). source_hash is a sync-only concern unknown at chunk time.
        for chunk in chunks:
            chunk.source_hash = src.change_token
        return SourceChange(
            source_id=src.source_id,
            source_hash=src.change_token,
            content_hash=chunks[0].content_hash if chunks else None,
            items=chunks,
        )

    async def _produce_from_doc_store(self, src: SyncSource) -> SourceChange[Chunk] | None:
        """Boundary-2 mode: split + chunk a Document loaded from the DocumentStore."""
        if self._document_store is None or self._chunk is None:  # pragma: no cover - wired only when set
            raise AssertionError("_produce_from_doc_store called without a document_store/ChunkPipeline")
        doc = await self._document_store.get_document(src.source_id)
        if doc is None:
            return None  # vanished between list_source_state and get_document → engine warns + skips
        # Already processed at Boundary 1 — split/chunk only.
        chunks = await self._chunk.run(doc)
        return SourceChange(
            source_id=src.source_id,
            source_hash=doc.source_hash or "",
            # ChunkPipeline.run just stamped doc.content_hash() on every chunk — reuse it.
            content_hash=chunks[0].content_hash if chunks else doc.content_hash(),
            items=chunks,
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
                # Sources whose hashing failed are still live (unreadable ≠ vanished):
                # they must never become orphan-deletion candidates.
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
        # Orphans are compared against the FULL document store (not the target subset),
        # so a subset run never deletes chunks whose documents still exist upstream.
        return SyncPlanInput(sources=resolved, live_source_ids=frozenset(doc_state), pre_skipped=tuple(pre_skipped))

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def plan(
        self, sources: Iterable[Path] | Iterable[str] | None = None, delete_orphans: bool = False
    ) -> ChangeSet[Chunk]:
        """Compute the change set without touching the vector store.

        Two modes, selected by which constructor built this pipeline:

        * **Direct:** *sources* is an iterable of file ``Path``\\ s (required).  Files are
          hashed and compared via ``vector_store.list_source_state()``; changed files are
          parsed/chunked.  ``source_hash`` (file bytes) is the change token.
        * **Boundary 2** (:meth:`from_document_store`): *sources* is an iterable of
          ``source_id`` strings, or ``None`` for every document in the store.  Documents
          whose ``content_hash`` differs from the vector store's are re-chunked.
          ``content_hash`` is the change token.

        Args:
            sources: Paths (direct) or source_ids / ``None`` (from DocumentStore).
            delete_orphans: Opt-in removal.  **Footgun (direct):** only safe when *sources* is
                your complete corpus.  From a DocumentStore, orphans are source_ids in the
                vector store but absent from the document store.

        Returns:
            A :class:`~ragdoc.pipeline.changeset.ChangeSet` of chunks.
        """
        return await self._engine.plan(await self._resolve(sources), delete_orphans)

    async def apply(self, changeset: ChangeSet[Chunk]) -> UpdateResult:
        """Embed and write *changeset* to the store.

        Embeds **all** chunks first (non-destructive): if embedding fails, the store is left
        untouched and every changed source is reported in ``errors``.  Then, per source:
        delete orphans, delete stale chunks for updates, and upsert.

        Args:
            changeset: The (possibly hand-edited) ChangeSet to apply.

        Returns:
            :class:`~ragdoc.pipeline.sync.UpdateResult` (``skipped`` is empty — apply does
            not know plan-time skips).
        """
        return await self._engine.apply(changeset)

    async def run(
        self,
        sources: Iterable[Path] | Iterable[str] | None = None,
        delete_orphans: bool = False,
    ) -> UpdateResult:
        """Streaming, **per-source** sync (the standard path).

        Produces the **same end result** as ``apply(plan(sources, delete_orphans))`` — same
        change detection (skip/add/update/orphan) and the same final store contents. The
        difference is *scheduling*, not outcome: ``apply(plan(...))`` chunks the whole corpus
        and embeds *all* chunks before any write, whereas ``run()`` processes each source
        independently and writes it as soon as it is ready (parse → chunk → embed →
        delete-then-upsert).

        **Per-source atomicity.** Each source is written delete-then-upsert, so an individual
        source is never left half-replaced, and its embed runs *before* its write (an embed
        failure leaves that one source untouched and records it in ``errors``). The **corpus**
        is *not* all-or-nothing: a source that succeeds is durable in the store even if a later
        source fails. Use ``apply(plan(...))`` when you need whole-corpus embed-first semantics
        or a reviewable :class:`~ragdoc.pipeline.changeset.ChangeSet`.

        Args:
            sources: Paths (direct mode) or source_ids / ``None`` for all (Boundary-2
                mode). See :meth:`plan`.
            delete_orphans: opt-in orphan removal (footgun in direct mode: pass the full
                corpus).

        Returns:
            :class:`~ragdoc.pipeline.sync.UpdateResult` (``skipped`` = unchanged sources,
            ``errors`` = per-source parse/chunk/embed/write failures).
        """
        return await self._engine.run(await self._resolve(sources), delete_orphans)

    async def run_directory(self, directory: Path, glob: str = "**/*") -> UpdateResult:
        """Discover files matching *glob* under *directory*, then :meth:`run` them."""
        # Directory traversal is blocking filesystem I/O — run it off the event loop.
        sources = await asyncio.to_thread(lambda: [p for p in directory.glob(glob) if p.is_file()])
        return await self.run(sources)

    # ------------------------------------------------------------------
    # embedding (the engine's pre_write hook)
    # ------------------------------------------------------------------

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
