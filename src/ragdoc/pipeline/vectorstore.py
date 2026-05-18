"""VectorStorePipeline: incremental vector-store updates with hash-based change detection.

:class:`VectorStorePipeline` **composes** a
:class:`~ragdoc.pipeline.linear.DocumentPipeline` with a
:class:`~ragdoc.pipeline.stores.VectorStore` to provide incremental updates.

The **vector store is the single source of truth**.  Each chunk carries
``source_id`` and ``source_hash`` as first-class fields, so the pipeline can
detect unchanged sources, clean up stale chunks, and discover orphans — all
without any local state file.

Typical usage::

    from pathlib import Path
    from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline

    vs = VectorStorePipeline(
        pipeline=DocumentPipeline(splitter=TokenSplitter()),
        vector_store=my_vector_store,
    )

    # First run: all files are new → processed
    r1 = await vs.run([Path("a.docx"), Path("b.html")])
    assert len(r1.processed) == 2 and len(r1.skipped) == 0

    # Second run without changes → all skipped
    r2 = await vs.run([Path("a.docx"), Path("b.html")])
    assert len(r2.skipped) == 2 and len(r2.processed) == 0
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Iterable, cast

from ragdoc.metadata import BaseMetadata, TMetadata
from ragdoc.processing._concurrency import _resolve_semaphore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.chunking.chunk import Chunk
    from ragdoc.pipeline.embedders import EmbedderConfig
    from ragdoc.pipeline.linear import DocumentPipeline
    from ragdoc.pipeline.stores import VectorStore


@dataclass
class UpdateResult:
    """Aggregated result from :meth:`VectorStorePipeline.run`.

    Attributes:
        processed: Paths that were newly processed or re-processed (file changed).
        skipped: Paths whose file-byte hash matched the vector store (no change).
        deleted: Source IDs that existed in the vector store but were absent from
            the current source list (their chunks were removed).
        errors: ``(path, exception)`` pairs for files that raised during processing.
            ``Exception`` subclasses are always collected here rather than propagated,
            so a single failing file does not abort the entire run.
            ``asyncio.CancelledError`` is re-raised immediately and never appears in
            this list.
    """

    processed: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: list[tuple[Path, BaseException]] = field(default_factory=list)


class VectorStorePipeline(Generic[TMetadata]):
    """Wraps a :class:`~ragdoc.pipeline.linear.DocumentPipeline` with incremental
    vector-store synchronisation.

    The vector store is the **single source of truth** for provenance.  Each
    chunk carries two first-class fields set by this pipeline before upsert:

    * ``source_id`` — the source identity key derived from the source ``Path``
      via ``source_id_fn`` (default: ``path.name``).
    * ``source_hash`` — SHA-256 hex digest of the raw file bytes.

    ``chunk.metadata`` is never modified by this pipeline; it is exclusively
    for user-defined custom data propagated through the document pipeline.

    **source_id strategies**

    The default ``source_id_fn = lambda p: p.name`` uses the bare filename,
    which is simple but will collide if the same filename appears in multiple
    directories.  Common alternatives:

    * Full path: ``lambda p: str(p)`` — unique per machine, not portable.
    * Relative path: ``lambda p: str(p.relative_to(base_dir))`` — portable
      when anchored to a known base directory.
    * Custom: any ``Callable[[Path], str]`` — for S3 keys, external IDs, etc.

    If ``source_id_fn`` maps two different ``Path`` objects to the same string,
    :meth:`run` raises ``ValueError`` before any processing begins.

    On each call to :meth:`run`:

    * **Unchanged** sources (vector store already has chunks with the same hash)
      are skipped.
    * **New or modified** sources are re-processed; stale chunks are deleted by
      source_id and new chunks are upserted.
    * Sources present in the vector store but **absent** from the current source
      list are deleted.

    Per-file exceptions from the underlying pipeline are collected in
    :attr:`UpdateResult.errors` rather than propagated, so a single failing file
    does not abort the entire run.

    Args:
        pipeline: :class:`~ragdoc.pipeline.linear.DocumentPipeline` to use
            for parsing, processing, splitting, and chunking.
        vector_store: :class:`~ragdoc.pipeline.stores.VectorStore` implementation.
        source_id_fn: Callable that maps a ``Path`` to its source identity string.
            Defaults to ``lambda p: p.name`` (bare filename).
        embedders: Named embedding configurations.  Each key becomes an entry in
            ``chunk.named_embeddings`` after embedding.  Embedders run concurrently.
            If ``None`` or empty, no embeddings are computed and ``named_embeddings``
            stays ``{}``.

            Typical usage — dense on ``embedding_content``, sparse on ``prompt_content``::

                from ragdoc.pipeline import EmbedderConfig, prompt_content_text

                VectorStorePipeline(
                    pipeline=...,
                    vector_store=...,
                    embedders={
                        "dense": EmbedderConfig(my_dense_embedder),
                        "sparse": EmbedderConfig(my_sparse_embedder, text_fn=prompt_content_text),
                    },
                )
        concurrency: Maximum number of sources processed concurrently.  Accepts
            either an ``int`` (a private semaphore is created) or a pre-built
            ``asyncio.Semaphore`` (shared across pipelines).  Defaults to ``10``:
            sources that hash-match the store are cheap reads and benefit from
            parallelism, while the default cap prevents rate-limiting on LLM-heavy
            pipelines.
    """

    def __init__(
        self,
        pipeline: "DocumentPipeline[TMetadata]",
        vector_store: "VectorStore",
        source_id_fn: Callable[[Path], str] = lambda p: p.name,
        embedders: "dict[str, EmbedderConfig] | None" = None,
        concurrency: int | asyncio.Semaphore = 10,
    ) -> None:
        self._pipeline = pipeline
        self._vector_store = vector_store
        self._source_id_fn = source_id_fn
        self._embedders = embedders or {}
        self._concurrency = concurrency

    async def run(self, sources: Iterable[Path]) -> UpdateResult:
        """Incrementally update the vector store from *sources*.

        Algorithm:

        1. Build ``current``: ``{source_id → (Path, sha256(bytes))}``.
           Raises ``ValueError`` immediately if any two paths produce the same
           ``source_id`` (collision detection).
        2. Query the vector store for all known source IDs; delete chunks for
           any source not in ``current``.
        3. For each source in ``current``:
           - Query the vector store for the existing hash.
           - If hash matches → **skip**.
           - Otherwise: delete stale chunks by source_id, process,
             set ``chunk.source_id`` and ``chunk.source_hash``, upsert.

        Args:
            sources: Iterable of :class:`~pathlib.Path` objects to process.

        Returns:
            :class:`UpdateResult` with ``processed``, ``skipped``, ``deleted``,
            and ``errors`` lists.

        Raises:
            ValueError: If ``source_id_fn`` maps two different paths to the
                same source_id.
        """
        result = UpdateResult()

        # Step 1 — build current-source map (source_id → (Path, file_hash))
        # and detect collisions up front.
        current: dict[str, tuple[Path, str]] = {}
        collisions: dict[str, list[Path]] = {}
        for path in sources:
            sid = self._source_id_fn(path)
            if sid in current:
                collisions.setdefault(sid, [current[sid][0]]).append(path)
            else:
                current[sid] = (path, _file_hash(path))

        if collisions:
            lines = [
                f"  {sid!r}: {[str(p) for p in paths]}"
                for sid, paths in collisions.items()
            ]
            raise ValueError(
                "source_id_fn produced duplicate source_ids for different paths.\n"
                + "\n".join(lines)
                + "\nConsider using a relative-path strategy: "
                "lambda p: str(p.relative_to(base_dir))"
            )

        logger.info(f"Run starting: {len(current)} sources")

        # Step 2 — delete sources that have been removed
        existing_ids = await self._vector_store.list_source_ids()
        for source_id in existing_ids:
            if source_id not in current:
                await self._vector_store.delete_by_source(source_id)
                result.deleted.append(source_id)

        if result.deleted:
            logger.info(f"Run: deleted {len(result.deleted)} orphaned sources")

        # Step 3 — process new / changed sources
        sem = _resolve_semaphore(self._concurrency)

        async def _process_one(source_id: str, path: Path, file_hash: str) -> None:
            async with sem:
                try:
                    existing_hash = await self._vector_store.get_source_hash(source_id)

                    if existing_hash == file_hash:
                        logger.info(f"Run: skipped {path.name} (unchanged)")
                        result.skipped.append(path)
                        return

                    # Remove stale chunks (changed file or missing hash).
                    if existing_hash is not None:
                        await self._vector_store.delete_by_source(source_id)

                    logger.info(f"Run: processing {path.name} (new/changed)")
                    chunks = await self._pipeline.run(path)
                    for chunk in chunks:
                        chunk.source_id = source_id
                        chunk.source_hash = file_hash

                    await self._embed_chunks(chunks)
                    # Cast to the widest accepted type at the vector-store boundary.
                    # Sound because TMetadata is bounded by BaseMetadata.
                    await self._vector_store.upsert(cast("list[Chunk[BaseMetadata]]", chunks))
                    result.processed.append(path)
                except BaseException as exc:
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    logger.error(f"Run failed for {path.name}: {exc!r}", exc_info=True)
                    result.errors.append((path, exc))

        try:
            await asyncio.gather(
                *[_process_one(sid, path, fhash) for sid, (path, fhash) in current.items()]
            )
        except BaseException:
            logger.error(
                f"Run failed: {len(result.processed)} processed, "
                f"{len(result.skipped)} skipped before failure",
                exc_info=True,
            )
            raise
        logger.info(
            f"Run complete: {len(result.processed)} processed, "
            f"{len(result.skipped)} skipped, {len(result.deleted)} deleted, "
            f"{len(result.errors)} errors"
        )
        return result

    async def _embed_chunks(self, chunks: "list[Chunk]") -> None:
        """Populate ``chunk.named_embeddings`` for all configured embedders (in-place).

        All embedders run concurrently via :func:`asyncio.gather`.  Each embedder
        extracts its own text slice from the chunks, embeds them, and writes the
        resulting vectors back by name.

        Args:
            chunks: Chunks to embed.  Modified in place.
        """
        if not self._embedders or not chunks:
            return

        async def _run_one(name: str, config: "EmbedderConfig") -> tuple[str, list[list[float]]]:
            texts = [config.text_fn(chunk) for chunk in chunks]
            vectors = await config.embedder.embed(texts)
            return name, vectors

        results = await asyncio.gather(
            *(_run_one(name, cfg) for name, cfg in self._embedders.items())
        )
        for name, vectors in results:
            for chunk, vec in zip(chunks, vectors):
                chunk.named_embeddings[name] = vec

    async def run_directory(
        self,
        directory: Path,
        glob: str = "**/*",
    ) -> UpdateResult:
        """Discover all files matching *glob* in *directory*, then call :meth:`run`.

        Args:
            directory: Root directory to search.
            glob: Glob pattern relative to *directory* (default ``"**/*"`` — all files).

        Returns:
            :class:`UpdateResult` from :meth:`run`.
        """
        sources = [p for p in directory.glob(glob) if p.is_file()]
        return await self.run(sources)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the raw bytes of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
