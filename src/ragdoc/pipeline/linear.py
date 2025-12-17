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
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, cast

from ragdoc.metadata import TMetadata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.chunking import Chunk, Chunker
    from ragdoc.document import Document
    from ragdoc.pipeline.parser import Parser
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
        parser: Converts a :class:`~pathlib.Path` to a
            :class:`~ragdoc.document.Document`.  Defaults to
            :class:`~ragdoc.pipeline.parser.AutoParser`.
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
        concurrency: Max files processed in parallel by :meth:`run_many` and
            :meth:`stream` (default ``1`` — sequential).
        on_error: ``"raise"`` (default) re-raises exceptions immediately.
            ``"skip"`` catches per-file exceptions and collects them in
            :attr:`PipelineResult.errors`.
    """

    def __init__(
        self,
        parser: Parser | None = None,
        processors: list[DocumentProcessor] | ProcessingPipeline | None = None,
        splitter: TokenSplitter | None = None,
        chunker: Chunker | None = None,
        concurrency: int = 1,
        on_error: Literal["raise", "skip"] = "raise",
        metadata_type: type[TMetadata] | None = None,
    ) -> None:
        from ragdoc.chunking import SimpleChunker
        from ragdoc.pipeline.parser import AutoParser
        from ragdoc.processing.base import ProcessingPipeline as PP

        self._parser: Parser = parser or AutoParser()
        self._splitter = splitter
        self._chunker: Chunker = chunker or SimpleChunker()
        self._concurrency = concurrency
        self._on_error = on_error
        self._metadata_type = metadata_type

        if processors is None:
            self._processing_pipeline: ProcessingPipeline = PP()
        elif isinstance(processors, PP):
            self._processing_pipeline = processors
        else:
            self._processing_pipeline = PP(processors)

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
        for coro in asyncio.as_completed(tasks):
            batch = await coro
            logger.debug(f"Stream batch yielded: {len(batch)} chunks")
            yield batch

    async def _process_one(self, source: Path) -> list[Chunk[TMetadata]]:
        logger.debug(f"Parsing {source.name}")
        doc = await self._parser(source)
        logger.debug(f"Parsed {source.name}: {len(doc.elements)} elements")
        if not doc.source_path:
            logger.warning(
                f"Parser did not set source_path on document from {source}. "
                "Chunks will have null provenance fields. "
                "Set doc.source_path in your parser function."
            )

        doc = await self._processing_pipeline.process(doc)
        if doc is None:
            logger.info(f"Document filtered out by processing pipeline: {source.name}")
            return []

        if self._metadata_type is not None:
            required = self._metadata_type.__required_keys__ - {"filename"}
            missing = required - set(doc.metadata.keys())
            if missing:
                raise ValueError(
                    f"Processors did not set required metadata fields: {missing}. "
                    f"Expected by {self._metadata_type.__name__}."
                )

        typed_doc: Document[TMetadata] = cast("Document[TMetadata]", doc)

        logger.debug(f"Splitting {source.name}")
        splits = self._splitter(typed_doc) if self._splitter else [typed_doc]
        logger.debug(f"Split {source.name} into {len(splits)} sub-documents")

        chunks: list[Chunk[TMetadata]] = [
            c
            for split in splits
            for c in await self._chunker.chunk(split)  # type: ignore[misc]
        ]
        logger.info(f"Completed {source.name}: {len(chunks)} chunks produced")
        return chunks
