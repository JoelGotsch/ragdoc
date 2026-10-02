"""DocumentDumpProcessor — serialize Document objects to .ragdoc.json files.

Useful for caching parsed documents, debugging pipelines, and sharing
intermediate results.  The companion parser :mod:`ragdoc.parsing.ragdoc_json`
reads them back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ragdoc.document import Document
from ragdoc.processing.base import DocumentProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FileNamer = Callable[[Document], str]
"""Callable that returns a filename (not a full path) for a given Document."""


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def default_file_namer(document: Document) -> str:
    """Return ``{stem}_{id[:8]}.ragdoc.json`` for *document*.

    The stem is derived from :attr:`~ragdoc.document.Document.source_path`.
    When ``source_path`` is empty the stem is omitted and only the id prefix is used.

    Args:
        document: The document to name.

    Returns:
        A filename string ending in ``.ragdoc.json``.
    """
    stem = Path(document.source_path).stem if document.source_path else ""
    id_prefix = document.id[:8]
    if stem:
        return f"{stem}_{id_prefix}.ragdoc.json"
    return f"{id_prefix}.ragdoc.json"


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class DocumentDumpProcessor(DocumentProcessor):
    """Serialize each document to a ``.ragdoc.json`` file and pass it through.

    The processor writes one JSON file per document into *output_dir* and
    returns the document unchanged.  It is safe to use as a pass-through stage
    anywhere in a :class:`~ragdoc.processing.base.ProcessingPipeline`.

    File I/O is async (via :mod:`aiofiles`) with a synchronous fallback in case
    aiofiles is unavailable or raises.

    Args:
        output_dir: Directory to write ``.ragdoc.json`` files into.
            Created automatically (including parents) if it does not exist.
        file_namer: Callable that maps a :class:`~ragdoc.document.Document` to
            a filename string.  Defaults to :func:`default_file_namer`.
        indent: JSON indentation level passed to
            :meth:`~pydantic.BaseModel.model_dump_json`.  Defaults to ``2``.

    Example:
        ```python
        proc = DocumentDumpProcessor(output_dir=Path("cache/"))
        doc = await proc.process(doc)   # writes cache/{stem}_{id[:8]}.ragdoc.json
        ```
    """

    def __init__(
        self,
        output_dir: Path | str,
        file_namer: FileNamer | None = None,
        indent: int = 2,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._file_namer: FileNamer = file_namer or default_file_namer
        self._indent = indent

    async def process(self, document: Document) -> Document:
        """Write *document* to a ``.ragdoc.json`` file and return it unchanged.

        Args:
            document: The document to serialize.

        Returns:
            The same *document* instance, unmodified.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / self._file_namer(document)
        content = document.model_dump_json(indent=self._indent)

        try:
            import aiofiles

            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.debug(f"DocumentDumpProcessor: wrote {path}")
        except OSError:
            logger.debug(f"DocumentDumpProcessor: async write failed, falling back to sync write for {path}")
            path.write_text(content, encoding="utf-8")

        return document
