import asyncio
from pathlib import Path

from ragdoc.document import Document
from ragdoc.parsing.pandoc.load import PandocHTML, generate_document as generate_pandoc_document
from ragdoc.parsing.parser import Parser


def load_pandoc(path: Path | str) -> Document:
    """Parse a file via Pandoc into a Document.

    Provenance (``source_path``, ``metadata["filename"]``) is stamped centrally by
    :func:`ragdoc.parsing.load` — not here.
    """
    return generate_pandoc_document(PandocHTML.from_file(Path(path)))


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class PandocParser(Parser):
    name: str = "pandoc"
    patterns: list[str] = [".docx", ".doc"]
    description: str = "Word documents via pandoc"

    async def __call__(self, path: Path) -> Document:
        # pypandoc forks a pandoc subprocess and blocks on it — run off the event loop.
        return await asyncio.to_thread(load_pandoc, path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(PandocParser())
