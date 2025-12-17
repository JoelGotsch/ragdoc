"""DocumentPipeline: linear facade over parse → process → split → chunk.

:class:`DocumentPipeline` is the entry point for Scenario A (linear pipeline)
and Scenario D (concurrent / streaming processing).  It wires together all
four pipeline stages behind a single async interface.

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


class DocumentPipeline(Generic[TMetadata]):
    """Linear facade over parse → process → split → chunk.

    Each call to :meth:`run` executes the four stages in sequence for a single
    file.  :meth:`run_many` fans out across multiple files with an
    :class:`asyncio.Semaphore`-based concurrency limit.  :meth:`stream` is a
    memory-efficient alternative that yields chunk batches as each file
    completes.

    Args:
        parser: Async callable converting a :class:`~pathlib.Path` to a
            :class:`~ragdoc.document.Document`.  Defaults to
            :func:`~ragdoc.parsing.load` (registry-based parser selection).
        processors: List of processors **or** a
            :class:`~ragdoc.processing.ProcessingPipeline`.  Optional — omit
            when no processing is needed.
        splitter: Splits a :class:`~ragdoc.document.Document` into
            sub-documents.  Optional — when ``None`` the whole document
            becomes one chunk.  :class:`~ragdoc.pipeline.splitter.TokenSplitter`
            is the recommended default.
        chunker: Converts a :class:`~ragdoc.document.Document` to
            :class:`~ragdoc.chunking.Chunk` (s).  Defaults to
            :class:`~ragdoc.chunking.SimpleChunker`.
        source_id_fn: ``Path -> source_id`` identity function stamped onto every
            parsed document (default: ``p.name``).
        chunk_id_fn: ``(source_id, split_sequence, chunk_ordinal, content_hash) -> id``
            minted onto every chunk by :meth:`chunk_document`.  Defaults to
            :func:`~ragdoc.chunking.provenance.mint_chunk_id` (deterministic,
            collision-free across splits).
        concurrency: Max files processed in parallel by :meth:`run_many` and
            :meth:`stream` (default ``1`` — sequential).
        on_error: ``"raise"`` (default) re-raises exceptions immediately.
            ``"skip"`` catches per-file exceptions and collects them in
            :attr:`PipelineResult.errors`.
    """

    def __init__(
        self,
        parser: Callable[[Path], Awaitable[Document]] | None = None,
        processors: list[DocumentProcessor] | ProcessingPipeline | None = None,
        splitter: TokenSplitter | None = None,
        chunker: Chunker | None = None,
        source_id_fn: Callable[[Path], str] = lambda p: p.name,
        chunk_id_fn: ChunkIdFn | None = None,
        concurrency: int = 1,
        on_error: Literal["raise", "skip"] = "raise",
        metadata_type: type[TMetadata] | None = None,
    ) -> None:
        from ragdoc.chunking import SimpleChunker
        from ragdoc.chunking.provenance import mint_chunk_id
        from ragdoc.processing.base import ProcessingPipeline as PP

        self._parser: Callable[[Path], Awaitable[Document]]
        if parser is None:
            from ragdoc.parsing import load

            self._parser = load
            self._parser_is_default = True
        else:
            self._parser = parser
            self._parser_is_default = False
        self._splitter = splitter
        self._chunker: Chunker = chunker or SimpleChunker()
        self._source_id_fn = source_id_fn
        self._chunk_id_fn: ChunkIdFn = chunk_id_fn or mint_chunk_id
        self._concurrency = concurrency
        self._on_error = on_error
        self._metadata_type = metadata_type

        if processors is None:
            self._processing_pipeline: ProcessingPipeline = PP()
        elif isinstance(processors, PP):
            self._processing_pipeline = processors
        else:
            self._processing_pipeline = PP(processors)

    @property
    def source_id_fn(self) -> Callable[[Path], str]:
        """The Path -> source_id function this pipeline stamps onto documents.

        Read by sync pipelines (``VectorStorePipeline`` / ``DocumentStorePipeline``) so the
        pre-parse collision/orphan checks use the same identity function — single source of
        truth, no drift.
        """
        return self._source_id_fn

    # The following predicates let the sync pipelines validate that a DocumentPipeline carries
    # only the stages meaningful at their boundary, and fail loudly (not silently) otherwise.
    # See DocumentStorePipeline (Boundary 1) and VectorStorePipeline (Boundary 2).

    @property
    def has_splitter(self) -> bool:
        """True if a splitter is configured (Boundary 1 forbids one — it stores whole Documents)."""
        return self._splitter is not None

    @property
    def has_custom_chunker(self) -> bool:
        """True if the chunker is not the default :class:`SimpleChunker` (Boundary 1 forbids one).

        A default ``SimpleChunker`` (whether implicit or passed explicitly) can't be told apart
        and is harmless — only a deliberately-configured chunker (e.g. ``LLMChunker``) signals a
        misplaced Boundary-2 stage.
        """
        from ragdoc.chunking import SimpleChunker

        return not isinstance(self._chunker, SimpleChunker)

    @property
    def has_processors(self) -> bool:
        """True if any processor is configured (Boundary 2 forbids them — docs are pre-processed)."""
        return len(self._processing_pipeline) > 0

    @property
    def has_custom_parser(self) -> bool:
        """True if the parser is not the default :func:`~ragdoc.parsing.load` (Boundary 2 forbids
        one — its source is the document store, not a file)."""
        return not self._parser_is_default

    async def run(self, source: Path) -> list[Chunk[TMetadata]]:
        """Parse, process, split, and chunk a single file.

        Args:
            source: Path to the source file.

        Returns:
            List of :class:`~ragdoc.chunking.Chunk` objects.
        """
        return await self._process_one(source)

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
                    chunks = await self._process_one(path)
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
                return await self._process_one(path)

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

    async def _parse(self, source: Path) -> Document:
        """Parse *source* and stamp ``source_id`` (a pure function of the path). Shared parse prefix."""
        logger.debug(f"Parsing {source.name}")
        doc = await self._parser(source)
        logger.debug(f"Parsed {source.name}: {len(doc.elements)} elements")
        if not doc.source_path:
            logger.warning(
                f"Parser did not set source_path on document from {source}. "
                "Set doc.source_path in your parser function."
            )
        # source_id is a pure function of the path; the file-byte source_hash is a sync
        # concern stamped by the sync pipeline (VectorStorePipeline / DocumentStorePipeline).
        doc.source_id = self._source_id_fn(source)
        return doc

    async def parse_and_process(self, source: Path) -> Document | None:
        """Parse + process a file into a Document, **without** splitting or chunking.

        Boundary-1 entry point for :class:`~ragdoc.pipeline.DocumentStorePipeline`, which
        stores Documents (not chunks).  Returns ``None`` if a processor drops the document.
        """
        return await self._processing_pipeline.process(await self._parse(source))

    async def _process_one(self, source: Path) -> list[Chunk[TMetadata]]:
        # Direct parse path: a freshly parsed document is not yet processed, so run the
        # processor chain here before chunking. Documents read back from a DocumentStore are
        # already processed (Boundary 1) and go straight to chunk_document, bypassing this.
        doc = await self._processing_pipeline.process(await self._parse(source))
        if doc is None:
            logger.info("Document filtered out by processing pipeline")
            return []
        return await self.chunk_document(doc)

    async def chunk_document(self, document: Document) -> list[Chunk[TMetadata]]:
        """Split → chunk an already parsed-and-processed Document (no parsing, no processing).

        Shared tail of :meth:`run` and the entry point for re-chunking a Document read
        back from a :class:`~ragdoc.pipeline.stores.DocumentStore` (Boundary 2 / Mode 2).

        **Processing is the caller's responsibility, not this method's.** The processor chain
        runs exactly once — at parse time on the direct path (:meth:`_process_one`) or at
        Boundary 1 (:class:`~ragdoc.pipeline.document_store_pipeline.DocumentStorePipeline`). This
        method deliberately never re-runs it: doing so on an already-processed document is both
        wasteful and **destructive**, because non-idempotent processors corrupt it (e.g.
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
