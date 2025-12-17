"""SimpleChunker: pure-rendering, no-LLM chunker.

Converts a :class:`~ragdoc.document.Document` into a single
:class:`~ragdoc.chunking.chunk.Chunk` using a :class:`~ragdoc.rendering.Renderer`
for prompt content.  ``embedding_content`` is set equal to ``prompt_content`` —
use :class:`~ragdoc.chunking.llm.LLMChunker` when distinct embedding content
is needed.  No LLM calls are made.

Typical usage::

    from ragdoc.chunking import SimpleChunker

    chunker = SimpleChunker()
    chunks = await chunker.chunk(doc)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ragdoc.chunking.base import Chunker
from ragdoc.chunking.chunk import Chunk

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import Document
    from ragdoc.rendering import Renderer


def _default_prompt_renderer() -> Renderer:
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


class SimpleChunker(Chunker):
    """Converts a document into a single :class:`~ragdoc.chunking.chunk.Chunk`.

    The renderer defaults to a sensible value; inject a replacement when you
    need custom formats, metadata headers, or external-ref providers.

    ``embedding_content`` is always identical to ``prompt_content``.  For
    distinct embedding content, use :class:`~ragdoc.chunking.llm.LLMChunker`.

    Args:
        prompt_renderer: Renders ``prompt_content`` (LLM context).
            Defaults to MARKDOWN with ``render_for_prompt``.
        metadata_fn: Called with the document to build the ``Chunk.metadata``
            dict.  Defaults to using the document's metadata.
        id_fn: Called with the document to produce the ``Chunk.id``.
            Defaults to :meth:`~ragdoc.document.Document.content_hash` — the same
            content always produces the same ID, making upserts idempotent.
    """

    def __init__(
        self,
        prompt_renderer: Renderer | None = None,
        metadata_fn: Callable[[Document], dict] | None = None,
        id_fn: Callable[[Document], str] | None = None,
    ) -> None:
        self._prompt_renderer = prompt_renderer
        self._metadata_fn: Callable[[Document], dict] = metadata_fn or (lambda doc: doc.metadata)
        self._id_fn: Callable[[Document], str] = id_fn or (lambda doc: doc.content_hash())

    def _get_prompt_renderer(self) -> Renderer:
        return self._prompt_renderer if self._prompt_renderer is not None else _default_prompt_renderer()

    async def chunk(self, document: Document) -> list[Chunk]:
        prompt_content = self._get_prompt_renderer().render(document)
        content_hash = document.content_hash()
        chunk = Chunk(
            id=self._id_fn(document),
            source_path=document.source_path or None,
            source_id=document.source_id or document.source_path or document.id,
            source_hash=document.source_hash or content_hash,
            content_hash=content_hash,
            prompt_content=prompt_content,
            embedding_content=prompt_content,
            metadata=self._metadata_fn(document),  # type: ignore[reportArgumentType]  # metadata_fn returns MetadataDict
        )
        chunk_label = f"{chunk.source_path} ({chunk.id})" if chunk.source_path else chunk.id
        logger.debug(f"SimpleChunker: produced chunk {chunk_label} ({len(prompt_content)} chars)")
        return [chunk]
