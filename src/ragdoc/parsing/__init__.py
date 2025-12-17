"""Parsing — convert files to :class:`~ragdoc.document.Document` objects.

Public API
----------

.. autofunction:: load
.. autofunction:: register_parser
.. autofunction:: get_parser
.. autofunction:: get_registered_parsers
.. autofunction:: describe_registry
"""

import logging
from pathlib import Path

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.registry import (
    ParserRegistration,
    _resolve_parser,
    describe_registry,
    get_parser,
    get_registered_parsers,
    register_parser,
    stamp_provenance,
    unregister_parser,
)
from ragdoc.parsing.xlsx import ExcelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in parser registrations
# ---------------------------------------------------------------------------


def _register_builtin_parsers() -> None:
    """Register all built-in parsers. Called once at import time."""
    from ragdoc.parsing.azure_di import _register as _reg_azure_di
    from ragdoc.parsing.html import _register as _reg_html
    from ragdoc.parsing.mineru import _register as _reg_mineru
    from ragdoc.parsing.pandoc import _register as _reg_pandoc
    from ragdoc.parsing.ragdoc_json import _register as _reg_ragdoc_json
    from ragdoc.parsing.xlsx import _register as _reg_xlsx

    _reg_html()
    _reg_pandoc()
    _reg_xlsx()
    _reg_azure_di()
    _reg_mineru()
    _reg_ragdoc_json()


_register_builtin_parsers()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


async def load(path: Path | str, parser: str | None = None) -> Document:
    """Load a file into a Document.

    Resolution order:

    1. Explicit ``parser=`` argument (e.g. ``parser="azure_di"``).
    2. Config ``parser_preferences`` from :func:`~ragdoc.config.configure`.
    3. Highest-priority matching parser from the registry.

    Args:
        path: Path to the file to parse.
        parser: Force a specific parser by name (must be registered).

    Returns:
        A parsed :class:`~ragdoc.document.Document`.

    Raises:
        ValueError: If no matching parser is found.
    """
    path = Path(path)
    resolved = _resolve_parser(path, parser_name=parser)
    logger.debug(f"load: using parser {resolved.name!r} for {path.name}")
    doc = await resolved(path)
    stamp_provenance(doc, path, resolved.name)
    logger.info(f"load: parsed {path.name} -> Document({len(doc.elements)} elements)")
    return doc
