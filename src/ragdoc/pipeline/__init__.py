"""ragdoc.pipeline — high-level pipeline facade.

Scenario A: linear pipeline
    :class:`DocumentPipeline` composes parse → process → split → chunk into a
    single async call.

Scenario D: concurrency + streaming
    :meth:`DocumentPipeline.run_many` fans out with an
    :class:`asyncio.Semaphore`-based concurrency limit.
    :meth:`DocumentPipeline.stream` yields chunk batches as each file
    completes, keeping memory usage flat.

Scenario B: incremental update
    :class:`VectorStorePipeline` composes :class:`DocumentPipeline` with
    hash-based change detection via
    :meth:`VectorStorePipeline.run`.

Quick start::

    from pathlib import Path
    from ragdoc.pipeline import DocumentPipeline, TokenSplitter

    pipeline = DocumentPipeline(splitter=TokenSplitter(max_tokens=4000))

    # single file
    chunks = await pipeline.run(Path("report.docx"))

    # many files, 4 at a time
    result = await pipeline.run_many(paths, concurrency=4)
"""

from ragdoc.pipeline.embedders import Embedder, EmbedderConfig, embedding_content_text, prompt_content_text
from ragdoc.pipeline.linear import DocumentPipeline, PipelineResult
from ragdoc.pipeline.parser import AutoParser, Parser
from ragdoc.pipeline.splitter import TokenSplitter
from ragdoc.pipeline.stores import VectorStore
from ragdoc.pipeline.vectorstore import UpdateResult, VectorStorePipeline

__all__ = [
    "AutoParser",
    "DocumentPipeline",
    "Embedder",
    "EmbedderConfig",
    "Parser",
    "PipelineResult",
    "TokenSplitter",
    "UpdateResult",
    "VectorStore",
    "VectorStorePipeline",
    "embedding_content_text",
    "prompt_content_text",
]
