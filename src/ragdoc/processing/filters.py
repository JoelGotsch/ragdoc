"""Document filter processors.

Filters are :class:`~ragdoc.processing.base.DocumentProcessor` subclasses that
return ``None`` to drop a document from the pipeline rather than transforming it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragdoc.processing.base import DocumentProcessor

if TYPE_CHECKING:
    from ragdoc.document import Document


class EmptyDocumentFilter(DocumentProcessor):
    """Drop documents that have no elements after parsing.

    A document with zero elements produces no useful chunks.  Use this filter
    early in a :class:`~ragdoc.processing.base.ProcessingPipeline` to skip
    empty or failed parses before any expensive LLM-based processors run.

    Returns ``None`` for empty documents, causing the pipeline to short-circuit
    and :class:`~ragdoc.pipeline.DocumentPipeline` to return zero chunks for
    that source file.

    Example:
        ```python
        from ragdoc.pipeline import DocumentPipeline
        from ragdoc.processing.filters import EmptyDocumentFilter

        pipeline = DocumentPipeline(
            processors=[EmptyDocumentFilter()],
        )
        chunks = await pipeline.run(path)  # [] if document was empty
        ```
    """

    async def process(self, document: Document) -> Document | None:
        """Return the document unchanged, or ``None`` if it has no elements.

        Args:
            document: The document to inspect.

        Returns:
            ``document`` if it has at least one element, otherwise ``None``.
        """
        if not document.elements:
            return None
        return document
