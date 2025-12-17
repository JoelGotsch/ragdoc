"""IngestPipeline / ChunkPipeline and their direct-path composition, DocumentPipeline.

The pipeline is split at the two sync boundaries:

* :class:`IngestPipeline` — Boundary 1 and the front of the direct path:
  parse → process → stamp ``source_id``.  Cannot carry a splitter or chunker.
* :class:`ChunkPipeline` — Boundary 2 and the back of the direct path:
  split → chunk → mint chunk ids.  Cannot carry a parser or processors.
* :class:`DocumentPipeline` — the direct path: ``IngestPipeline`` ∘ ``ChunkPipeline``,
  plus fan-out (:meth:`DocumentPipeline.run_many` / :meth:`DocumentPipeline.stream`).

Misplacing a stage is therefore a ``TypeError`` at construction (there is no parameter
for it), not a runtime rejection.

All methods are async.  For one-off synchronous use::

    import asyncio
    chunks = asyncio.run(pipeline.run(Path("report.docx")))

Typical usage::

    from pathlib import Path
    from ragdoc.pipeline import DocumentPipeline, TokenSplitter

    pipeline = DocumentPipeline(splitter=TokenSplitter(max_tokens=4000))
    chunks = await pipeline.run(Path("report.docx"))

For concurrent processing of many files::

    result = await pipeline.run_many(paths, concurrency=8)

For memory-efficient streaming::

    async for batch in pipeline.stream(paths):
        await vector_store.upsert(batch)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, cast

from ragdoc.chunking.provenance import resolve_chunk_provenance
from ragdoc.metadata import TMetadata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from ragdoc.chunking import Chunk, Chunker
    from ragdoc.chunking.provenance import ChunkIdFn
    from ragdoc.document import Document
    from ragdoc.pipeline.splitter import TokenSplitter
    from ragdoc.processing.base import DocumentProcessor, ProcessingPipeline


@dataclass
class PipelineResult:
    """Aggregated result from :meth:`DocumentPipeline.run_many`.

    Attributes:
        chunks: All chunks produced across all successfully processed files.
        errors: ``(path, exception)`` pairs for files that failed when
            ``on_error="skip"`` is set. ``asyncio.CancelledError`` is always
            re-raised and never appears here.
    """

    chunks: list[Chunk] = field(default_factory=list)
    errors: list[tuple[Path, BaseException]] = field(default_factory=list)


class IngestPipeline(Generic[TMetadata]):
    """Boundary 1 + front of the direct path: parse → process → stamp ``source_id``.

    Produces whole :class:`~ragdoc.document.Document` objects — it has **no** splitter or
    chunker parameter, so a Boundary-2 stage cannot be misconfigured onto it
    (``TypeError`` at construction).  Used standalone by
    :class:`~ragdoc.pipeline.document_store_pipeline.DocumentStorePipeline` and composed
    into :class:`DocumentPipeline` for the direct path.

    Args:
        parser: Async callable converting a :class:`~pathlib.Path` to a
            :class:`~ragdoc.document.Document`.  Defaults to
            :func:`~ragdoc.parsing.load` (registry-based parser selection).
        processors: List of processors **or** a
            :class:`~ragdoc.processing.ProcessingPipeline`.  Optional — omit
            when no processing is needed.
        source_id_fn: ``Path -> source_id`` identity function stamped onto every
            parsed document (default: ``p.name``).
    """

    def __init__(
        self,
        parser: Callable[[Path], Awaitable[Document]] | None = None,
        processors: list[DocumentProcessor] | ProcessingPipeline | None = None,
        source_id_fn: Callable[[Path], str] | None = None,
    ) -> None:
        from ragdoc.processing.base import ProcessingPipeline as PP

        if parser is None:
            from ragdoc.parsing import load

            self._parser: Callable[[Path], Awaitable[Document]] = load
        else:
            self._parser = parser
        self._source_id_fn: Callable[[Path], str] = source_id_fn if source_id_fn is not None else (lambda p: p.name)

        if processors is None:
            self._processing_pipeline: ProcessingPipeline = PP()
        elif isinstance(processors, PP):
            self._processing_pipeline = processors
        else:
            self._processing_pipeline = PP(processors)

    @property
    def source_id_fn(self) -> Callable[[Path], str]:
        """The Path -> source_id function this pipeline stamps onto documents.

        Read by the sync pipelines (``VectorStorePipeline`` / ``DocumentStorePipeline`` /
        ``MentionStorePipeline``) so the pre-parse collision/orphan checks use the same
        identity function — single source of truth, no drift.
        """
        return self._source_id_fn

    async def run(self, source: Path) -> Document | None:
        """Parse + process one file into a Document; ``None`` if a processor drops it.

        ``source_id`` is stamped from ``source_id_fn`` (a pure function of the path); the
        file-byte ``source_hash`` is a sync concern stamped by the sync pipelines.
        """
        logger.debug(f"Parsing {source.name}")
        doc = await self._parser(source)
        logger.debug(f"Parsed {source.name}: {len(doc.elements)} elements")
        if not doc.source_path:
            logger.warning(
                f"Parser did not set source_path on document from {source}. "
                "Set doc.source_path in your parser function."
            )
        doc.source_id = self._source_id_fn(source)
        return await self._processing_pipeline.process(doc)


class ChunkPipeline(Generic[TMetadata]):
    """Boundary 2 + back of the direct path: split → chunk an already-processed Document.

    Consumes :class:`~ragdoc.document.Document` objects that were parsed and processed
    elsewhere (the direct path's :class:`IngestPipeline`, or a
    :class:`~ragdoc.pipeline.stores.DocumentStore` at Boundary 2) — it has **no** parser
    or processors parameter, so re-running the (destructive, non-idempotent) processor
    chain on an already-processed document cannot be misconfigured (``TypeError`` at
    construction).

    Args:
        splitter: Splits a :class:`~ragdoc.document.Document` into sub-documents.
            Optional — when ``None`` the whole document becomes one chunk.
            :class:`~ragdoc.pipeline.splitter.TokenSplitter` is the recommended default.
        chunker: Converts a :class:`~ragdoc.document.Document` to
            :class:`~ragdoc.chunking.Chunk` (s).  Defaults to
            :class:`~ragdoc.chunking.SimpleChunker`.
        chunk_id_fn: ``(source_id, split_sequence, chunk_ordinal, content_hash) -> id``
            minted onto every chunk by :meth:`run`.  Defaults to
            :func:`~ragdoc.chunking.provenance.mint_chunk_id` (deterministic,
            collision-free across splits).
        metadata_type: Optional TypedDict subclass of
            :class:`~ragdoc.metadata.BaseMetadata`; when set, :meth:`run` validates that
            all ``Required`` keys are present on the document's metadata.
    """

    def __init__(
        self,
        splitter: TokenSplitter | None = None,
        chunker: Chunker | None = None,
        chunk_id_fn: ChunkIdFn | None = None,
        metadata_type: type[TMetadata] | None = None,
    ) -> None:
        from ragdoc.chunking import SimpleChunker
        from ragdoc.chunking.provenance import mint_chunk_id

        self._splitter = splitter
        self._chunker: Chunker = chunker or SimpleChunker()
        self._chunk_id_fn: ChunkIdFn = chunk_id_fn or mint_chunk_id
        self._metadata_type = metadata_type

    async def run(self, document: Document) -> list[Chunk[TMetadata]]:
        """Split → chunk an already parsed-and-processed Document (no parsing, no processing).

        Back half of :meth:`DocumentPipeline.run` and the Boundary-2 entry point for
        re-chunking a Document read back from a
        :class:`~ragdoc.pipeline.stores.DocumentStore`
        (``VectorStorePipeline.from_document_store``).

        **Processing is the caller's responsibility, not this method's.** The processor chain
        runs exactly once — at ingest time on the direct path (:class:`IngestPipeline`) or at
        Boundary 1 (:class:`~ragdoc.pipeline.document_store_pipeline.DocumentStorePipeline`).
        This class deliberately cannot re-run it: doing so on an already-processed document is
        both wasteful and **destructive**, because non-idempotent processors corrupt it (e.g.
        ``LLMHeadingResolver(remove_elements_before_title=True)`` re-detects a title and strips
        most elements; ``FootnoteProcessor`` re-resolves already-anchored footnotes onto the
        wrong numerals, inserting duplicate refs).

        Provenance is materialized **uniformly** here via
        :func:`~ragdoc.chunking.provenance.resolve_chunk_provenance`: ``content_hash`` is
        computed once from *document* and stamped on every chunk, and ``source_id`` /
        ``source_hash`` are propagated to each split so a multi-split source produces chunks
        that share one ``content_hash`` (required for Boundary-2 change detection).
        ``source_hash`` may be ``None`` — it is never faked from the content hash.

        **Chunk ids are minted here** (single id authority), by ``chunk_id_fn`` (default
        :func:`~ragdoc.chunking.provenance.mint_chunk_id`) over ``(source_id,
        split_sequence, chunk_ordinal, content_hash)``.  ``split_sequence`` comes from
        positional enumeration of the splitter output (1-based — equal to
        ``metadata["split_sequence"]`` whenever the splitter is ``split_document``, and
        defined for any custom splitter); ``chunk_ordinal`` is the 0-based chunk index
        within a split.  Identical-content splits of one source therefore get distinct ids.

        Args:
            document: A parsed, processed, and (optionally) provenance-stamped Document.

        Returns:
            List of chunks (one or more per split, depending on the chunker).
        """
        doc = document

        if self._metadata_type is not None:
            required = self._metadata_type.__required_keys__ - {"filename"}
            missing = required - set(doc.metadata.keys())
            if missing:
                raise ValueError(
                    f"Processors did not set required metadata fields: {missing}. "
                    f"Expected by {self._metadata_type.__name__}."
                )

        # Source-level provenance (uniform across all of this document's chunks).
        prov = resolve_chunk_provenance(doc)

        typed_doc: Document[TMetadata] = cast("Document[TMetadata]", doc)

        # Splitting renders per element/group to measure token budgets — a pandoc-subprocess
        # storm for non-HTML formats. Splitters are sync callables, so hop off the event loop
        # at this orchestration boundary.
        splits = await asyncio.to_thread(self._splitter, typed_doc) if self._splitter else [typed_doc]
        logger.debug(f"Split into {len(splits)} sub-documents")

        chunks: list[Chunk[TMetadata]] = []
        for seq, split in enumerate(splits, 1):
            split.source_id = prov.source_id
            split.source_hash = prov.source_hash
            split_chunks = await self._chunker.chunk(split)
            for ordinal, chunk in enumerate(split_chunks):
                chunk.content_hash = prov.content_hash
                chunk.id = self._chunk_id_fn(prov.source_id, seq, ordinal, prov.content_hash)
            chunks.extend(cast("list[Chunk[TMetadata]]", split_chunks))
        logger.info(f"Chunked document {prov.source_id}: {len(chunks)} chunks produced")
        return chunks


class DocumentPipeline(Generic[TMetadata]):
    """Direct path: :class:`IngestPipeline` ∘ :class:`ChunkPipeline`, plus fan-out.

    Each call to :meth:`run` executes parse → process (``self.ingest``) then
    split → chunk (``self.chunk``) for a single file.  :meth:`run_many` fans out across
    multiple files with an :class:`asyncio.Semaphore`-based concurrency limit.
    :meth:`stream` is a memory-efficient alternative that yields chunk batches as each
    file completes.

    Construct either from the two sub-pipelines (``ingest=`` / ``chunk=``) or from the
    flat per-stage kwargs, which delegate into freshly built sub-pipelines.  Mixing a
    sub-pipeline with its own stages' flat kwargs is a ``TypeError``.

    Args:
        ingest: Pre-built :class:`IngestPipeline` (parse → process).  Mutually exclusive
            with ``parser`` / ``processors`` / ``source_id_fn``.
        chunk: Pre-built :class:`ChunkPipeline` (split → chunk).  Mutually exclusive with
            ``splitter`` / ``chunker`` / ``chunk_id_fn`` / ``metadata_type``.
        parser: See :class:`IngestPipeline`.
        processors: See :class:`IngestPipeline`.
        source_id_fn: See :class:`IngestPipeline`.
        splitter: See :class:`ChunkPipeline`.
        chunker: See :class:`ChunkPipeline`.
        chunk_id_fn: See :class:`ChunkPipeline`.
        metadata_type: See :class:`ChunkPipeline`.
        concurrency: Max files processed in parallel by :meth:`run_many` and
            :meth:`stream` (default ``1`` — sequential).
        on_error: ``"raise"`` (default) re-raises exceptions immediately.
            ``"skip"`` catches per-file exceptions and collects them in
            :attr:`PipelineResult.errors`.
    """

    def __init__(
        self,
        ingest: IngestPipeline[TMetadata] | None = None,
        chunk: ChunkPipeline[TMetadata] | None = None,
        *,
        parser: Callable[[Path], Awaitable[Document]] | None = None,
        processors: list[DocumentProcessor] | ProcessingPipeline | None = None,
        source_id_fn: Callable[[Path], str] | None = None,
        splitter: TokenSplitter | None = None,
        chunker: Chunker | None = None,
        chunk_id_fn: ChunkIdFn | None = None,
        metadata_type: type[TMetadata] | None = None,
        concurrency: int = 1,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        if ingest is not None and not (parser is None and processors is None and source_id_fn is None):
            raise TypeError(
                "DocumentPipeline: parser/processors/source_id_fn belong to the IngestPipeline — "
                "configure them on the ingest= sub-pipeline, not alongside it."
            )
        if chunk is not None and not (
            splitter is None and chunker is None and chunk_id_fn is None and metadata_type is None
        ):
            raise TypeError(
                "DocumentPipeline: splitter/chunker/chunk_id_fn/metadata_type belong to the ChunkPipeline — "
                "configure them on the chunk= sub-pipeline, not alongside it."
            )
        self.ingest: IngestPipeline[TMetadata] = (
            ingest
            if ingest is not None
            else IngestPipeline(parser=parser, processors=processors, source_id_fn=source_id_fn)
        )
        self.chunk: ChunkPipeline[TMetadata] = (
            chunk
            if chunk is not None
            else ChunkPipeline(splitter=splitter, chunker=chunker, chunk_id_fn=chunk_id_fn, metadata_type=metadata_type)
        )
        self._concurrency = concurrency
        self._on_error = on_error

    @property
    def source_id_fn(self) -> Callable[[Path], str]:
        """The Path -> source_id function (delegates to :attr:`ingest`)."""
        return self.ingest.source_id_fn

    async def run(self, source: Path) -> list[Chunk[TMetadata]]:
        """Parse, process, split, and chunk a single file.

        Args:
            source: Path to the source file.

        Returns:
            List of :class:`~ragdoc.chunking.Chunk` objects (empty if a processor
            filtered the document out).
        """
        doc = await self.ingest.run(source)
        if doc is None:
            logger.info("Document filtered out by processing pipeline")
            return []
        return await self.chunk.run(doc)

    async def run_many(
        self,
        sources: Iterable[Path],
        *,
        concurrency: int | None = None,
    ) -> PipelineResult:
        """Process multiple files, respecting the concurrency limit.

        Args:
            sources: Iterable of file paths to process.
            concurrency: Override the instance-level concurrency for this call.

        Returns:
            :class:`PipelineResult` with all chunks and any collected errors.
        """
        limit = concurrency if concurrency is not None else self._concurrency
        source_list = list(sources)
        n_total = len(source_list)
        logger.info(f"Starting batch: {n_total} files, concurrency={limit}")
        result = PipelineResult()
        semaphore = asyncio.Semaphore(limit)
        completed = 0

        async def _run_one(path: Path) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    chunks = await self.run(path)
                    result.chunks.extend(chunks)
                except BaseException as exc:
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if self._on_error == "skip":
                        logger.error(f"Failed to process {path.name}: {exc!r}", exc_info=True)
                        result.errors.append((path, exc))
                    else:
                        raise
                finally:
                    completed += 1
                    logger.info(f"Progress: {completed}/{n_total} complete ({path.name})")

        try:
            await asyncio.gather(*[_run_one(p) for p in source_list])
        except BaseException:
            logger.error(
                f"Batch failed: {len(result.chunks)} chunks, {len(result.errors)} errors before failure",
                exc_info=True,
            )
            raise
        logger.info(f"Batch complete: {len(result.chunks)} chunks, {len(result.errors)} errors")
        return result

    async def stream(
        self,
        sources: Iterable[Path],
        *,
        concurrency: int | None = None,
    ) -> AsyncIterator[list[Chunk]]:
        """Yield chunk batches as each file completes.

        Memory-efficient alternative to :meth:`run_many` for large file sets:
        each batch can be handed off (e.g. to a vector store) before the next
        file is started.

        Args:
            sources: Iterable of file paths to process.
            concurrency: Override the instance-level concurrency for this call.

        Yields:
            Lists of :class:`~ragdoc.chunking.Chunk` objects, one list per
            completed file.
        """
        limit = concurrency if concurrency is not None else self._concurrency
        source_list = list(sources)
        logger.info(f"Streaming {len(source_list)} files, concurrency={limit}")
        semaphore = asyncio.Semaphore(limit)

        async def _run_one(path: Path) -> list[Chunk]:
            async with semaphore:
                return await self.run(path)

        tasks = [asyncio.create_task(_run_one(p)) for p in source_list]
        try:
            for coro in asyncio.as_completed(tasks):
                batch = await coro
                logger.debug(f"Stream batch yielded: {len(batch)} chunks")
                yield batch
        finally:
            # If the consumer stops iterating early or a task raises, the remaining
            # tasks must not be abandoned: cancel and await them all.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
