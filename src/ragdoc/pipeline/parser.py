"""Parser Protocol and AutoParser for the pipeline module.

The :class:`Parser` Protocol defines the interface expected by
:class:`~ragdoc.pipeline.linear.DocumentPipeline`.  The protocol is defined in
:mod:`ragdoc.parsing.parser` and re-exported here for convenience.

:class:`AutoParser` is the batteries-included default: it delegates to
:func:`~ragdoc.parsing.load`, selecting the right parser via the registry.

.. deprecated::
    Import :class:`Parser` from :mod:`ragdoc.parsing.parser` instead.
    Use :func:`~ragdoc.parsing.load` directly instead of ``AutoParser``.
"""

from __future__ import annotations

from pathlib import Path

from ragdoc.document import Document

# Re-export the canonical Parser protocol from parsing.parser
from ragdoc.parsing.parser import Parser

__all__ = ["AutoParser", "Parser"]


class AutoParser:
    """Selects the appropriate parser based on file extension via the registry.

    Wraps :func:`~ragdoc.parsing.load` (the registry-based entry point).

    Raises:
        ValueError: If no parser is registered for the file extension.
    """

    async def __call__(self, path: Path) -> Document:
        from ragdoc.parsing import load

        return await load(path)
