"""Parsing — convert files to :class:`~ragdoc.document.Document` objects.

Public API
----------

.. autofunction:: load
.. autofunction:: register_parser
.. autofunction:: get_parser
.. autofunction:: get_registered_parsers
.. autofunction:: describe_registry
.. autofunction:: load_document
.. autofunction:: from_path
"""

import logging
from pathlib import Path
from typing import Union

from ragdoc.document import Document
from ragdoc.parsing.azure_di import AzureAnalyzeRun, AzureJSONFile
from ragdoc.parsing.base import load_file
from ragdoc.parsing.html import HTMLFile, HTMLSource
from ragdoc.parsing.pandoc import PandocFile, WordFile
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.registry import (
    ParserRegistration,
    _resolve_parser,
    describe_registry,
    get_parser,
    get_registered_parsers,
    register_parser,
    unregister_parser,
)
from ragdoc.parsing.xlsx import ExcelConfig, ExcelPackage, ExcelSource

# Union of all concrete source types accepted by load_document / from_path
DocumentSource = HTMLSource | PandocFile | WordFile | ExcelSource | AzureJSONFile | AzureAnalyzeRun

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
# New unified entry point
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
    logger.info(f"load: parsed {path.name} -> Document({len(doc.elements)} elements)")
    return doc


# ---------------------------------------------------------------------------
# Legacy API (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def load_document(source: DocumentSource) -> Document:
    """Load a Document from a typed DocumentSource.

    .. deprecated::
        Use :func:`load` instead.
    """
    return load_file(source)


def from_path(path: Path | str) -> DocumentSource:
    """Create the appropriate DocumentSource for a file path based on its extension.

    Supported extensions: .html, .docx, .doc, .xlsx, .json (azure: *.azure.json)

    .. deprecated::
        Use :func:`load` instead.

    Args:
        path: Path to the source file

    Returns:
        A DocumentSource instance ready to pass to load_document()

    Raises:
        ValueError: If the file extension is not supported
    """
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    match suffixes:
        case [*_, ".azure", ".json"]:
            return AzureJSONFile(file_path=path)
        case [*_, ".html"]:
            return HTMLSource(file_path=path)
        case [*_, ".docx"] | [*_, ".doc"]:
            return PandocFile(file_path=str(path))
        case [*_, ".xlsx"]:
            return ExcelSource(file_path=path)
        case _:
            raise ValueError(f"Unsupported file type: {''.join(path.suffixes)!r} for {path}")
