"""RagdocJsonParser — deserialize ``.ragdoc.json`` files back into Documents.

``.ragdoc.json`` files are produced by
:class:`~ragdoc.processing.dump.DocumentDumpProcessor`.  This parser reads them
back, with control over how provenance fields (``source_path``, ``parser``,
``metadata["filename"]``) are set on the resulting document.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic import Field

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProvenanceMode
# ---------------------------------------------------------------------------


class ProvenanceMode(str, Enum):
    """Controls how provenance fields are set after deserializing a ``.ragdoc.json`` file.

    Attributes:
        ORIGINAL: Preserve ``source_path``, ``parser``, and ``metadata["filename"]``
            exactly as they appear in the JSON (i.e. the original document's provenance).
            Use this when you want to continue processing as if the document had
            just been parsed from its original source.
        JSON_FILE: Overwrite fields to reflect the ``.ragdoc.json`` file itself:
            ``metadata["filename"]`` becomes the JSON file's name,
            ``source_path`` becomes the full path to the JSON file, and
            ``parser`` is set to ``"ragdoc_json"``.  Use this when you want to
            treat the serialized file as the canonical source.
    """

    ORIGINAL = "ORIGINAL"
    JSON_FILE = "JSON_FILE"


# ---------------------------------------------------------------------------
# Standalone parse function
# ---------------------------------------------------------------------------


async def parse_ragdoc_json(path: Path, provenance_mode: ProvenanceMode) -> Document:
    """Deserialize a ``.ragdoc.json`` file into a :class:`~ragdoc.document.Document`.

    Reads the file asynchronously via :mod:`aiofiles`.

    Args:
        path: Path to the ``.ragdoc.json`` file.
        provenance_mode: Controls how ``source_path``, ``parser``, and
            ``metadata["filename"]`` are set on the returned document.
            See :class:`ProvenanceMode`.

    Returns:
        The deserialized :class:`~ragdoc.document.Document`.
    """
    import aiofiles

    async with aiofiles.open(path, encoding="utf-8") as f:
        content = await f.read()
    logger.debug(f"parse_ragdoc_json: read {path} via aiofiles")

    document = Document.model_validate_json(content)

    if provenance_mode is ProvenanceMode.JSON_FILE:
        document.metadata["filename"] = path.name
        document.source_path = str(path)
        document.parser = "ragdoc_json"

    return document


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------


class RagdocJsonParser(Parser):
    """Parser for ``.ragdoc.json`` files produced by DocumentDumpProcessor.

    Registered automatically at priority 60, which is higher than all built-in
    parsers, so ``.ragdoc.json`` files are always dispatched here by
    :func:`~ragdoc.parsing.load`.

    ``provenance_mode`` controls how provenance fields are handled after
    deserialization (see the field description); the default
    :attr:`ProvenanceMode.ORIGINAL` preserves the original document's
    ``source_path``, ``parser``, and ``metadata["filename"]``.

    Example:
        ```python
        from ragdoc.parsing import load

        # Uses ORIGINAL mode by default — provenance from the original source
        doc = await load("snapshot.ragdoc.json")

        # Or force JSON_FILE mode for a specific parser instance:
        from ragdoc.parsing.ragdoc_json import RagdocJsonParser, ProvenanceMode
        parser = RagdocJsonParser(provenance_mode=ProvenanceMode.JSON_FILE)
        doc = await parser(Path("snapshot.ragdoc.json"))
        ```
    """

    name: str = "ragdoc_json"
    patterns: list[str] = [".ragdoc.json"]
    priority: int = 60
    description: str = "ragdoc JSON snapshot files (.ragdoc.json)"

    provenance_mode: ProvenanceMode = Field(
        default=ProvenanceMode.ORIGINAL,
        description=(
            'Controls how source_path, parser, and metadata["filename"] are set on '
            "the deserialized Document. ORIGINAL preserves values from the JSON; "
            "JSON_FILE overwrites them to reflect the .ragdoc.json file itself."
        ),
    )

    async def __call__(self, path: Path) -> Document:
        return await parse_ragdoc_json(path, self.provenance_mode)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def _register() -> None:
    """Register RagdocJsonParser in the global parser registry."""
    from ragdoc.parsing.registry import register_parser

    register_parser(RagdocJsonParser())
