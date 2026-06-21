"""Base class for all chunkers.

A :class:`Chunker` converts a single :class:`~ragdoc.document.Document` into
one or more :class:`~ragdoc.chunking.chunk.Chunk` objects.

The library is async-only.  Implement :meth:`chunk`::

    class MyChunker(Chunker):
        async def chunk(self, document: Document) -> list[Chunk]:
            return [Chunk(prompt_content="...", embedding_content="...")]

    chunker = MyChunker()
    chunks = await chunker.chunk(doc)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragdoc.chunking.chunk import Chunk
    from ragdoc.document import Document


class Chunker(ABC):
    """Abstract base class for all chunkers.

    Implement :meth:`chunk`.  See the module docstring for usage examples.
    """

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        """Convert *document* into one or more :class:`~ragdoc.chunking.chunk.Chunk` objects."""
        ...
