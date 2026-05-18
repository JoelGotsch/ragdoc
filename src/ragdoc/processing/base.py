"""
Core processing infrastructure.

This module defines the base classes and protocols for document processing:
- DocumentProcessor: Async ABC for document transformation
- ProcessingPipeline: Chain multiple processors together

Design Principles:
------------------
- Async-only: All processors implement an async process() method
- Immutable transforms: Processors return new Document instances (or mutate in place)
- Universal: Processors work on Document objects, not parser-specific contexts
- Composable: ProcessingPipeline chains processors in sequence
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import Document


class DocumentProcessor(ABC):
    """
    Abstract base class for async document processors.

    DocumentProcessor transforms a Document and returns the (possibly modified)
    Document, or ``None`` to signal that the document should be dropped entirely.
    All processors are async to support LLM calls and other I/O. Pure-sync
    processors simply implement ``async def process()`` without any ``await``.

    Returning ``None`` causes ``ProcessingPipeline`` to short-circuit: no
    subsequent processors run, and ``DocumentPipeline`` produces zero chunks
    for that source file.

    Example:
        ```python
        class MyProcessor(DocumentProcessor):
            async def process(self, document: Document) -> Document:
                for element in document.elements:
                    if isinstance(element, Heading):
                        element.level = min(element.level, 3)
                return document

        processor = MyProcessor()
        doc = await processor.process(doc)
        ```

    Example (filter):
        ```python
        class NonEmptyFilter(DocumentProcessor):
            async def process(self, document: Document) -> Document | None:
                return document if document.elements else None
        ```
    """

    @abstractmethod
    async def process(self, document: "Document") -> "Document | None":
        """
        Process a document and return the result.

        Args:
            document: The document to process.

        Returns:
            The processed document (may be the same instance, modified in
            place), or ``None`` to drop the document from the pipeline.
        """
        ...

    async def __call__(self, document: "Document") -> "Document | None":
        """Allow processors to be called directly."""
        return await self.process(document)


class ProcessingPipeline:
    """
    Chain multiple processors together into a single pipeline.

    Example:
        ```python
        pipeline = ProcessingPipeline([
            HeadingLevelProcessor(),
            TitleDetectionProcessor(),
            LLMHeadingResolver(client),
        ])

        doc = await pipeline.process(doc)
        ```
    """

    def __init__(
        self,
        processors: list[DocumentProcessor] | None = None,
    ):
        self._processors: list[DocumentProcessor] = processors or []

    def add(self, processor: DocumentProcessor) -> "ProcessingPipeline":
        """Add a processor to the pipeline.  Returns self for chaining."""
        self._processors.append(processor)
        return self

    async def process(self, document: "Document") -> "Document | None":
        """Run all processors in sequence.

        Short-circuits as soon as any processor returns ``None``, meaning the
        document is dropped and no further processors run.

        Args:
            document: The document to process.

        Returns:
            The processed document, or ``None`` if a processor dropped it.
        """
        n = len(self._processors)
        if n:
            logger.debug(f"Running {n} processors")
        for i, processor in enumerate(self._processors):
            logger.debug(f"Running processor {i + 1}/{n}: {type(processor).__name__}")
            result = await processor.process(document)
            if result is None:
                source = getattr(document, "source_path", None)
                logger.debug(
                    f"{type(processor).__name__} dropped document"
                    + (f": {source}" if source else "")
                )
                return None
            document = result
        return document

    async def __call__(self, document: "Document") -> "Document | None":
        """Allow pipeline to be called directly."""
        return await self.process(document)

    def __len__(self) -> int:
        return len(self._processors)

    def __iter__(self):
        return iter(self._processors)
