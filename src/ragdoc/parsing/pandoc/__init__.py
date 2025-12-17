from pathlib import Path

from ragdoc.document import Document
from ragdoc.parsing.pandoc.load import PandocHTML, generate_document as generate_pandoc_document
from ragdoc.parsing.parser import Parser


def load_pandoc(path: Path | str) -> Document:
    """Parse a file via Pandoc into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    file_path = Path(path)
    document = generate_pandoc_document(PandocHTML.from_file(file_path))
    document.metadata["filename"] = file_path.name
    document.source_path = str(file_path)
    return document


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class PandocParser(Parser):
    name: str = "pandoc"
    patterns: list[str] = [".docx", ".doc"]
    description: str = "Word documents via pandoc"

    async def __call__(self, path: Path) -> Document:
        return load_pandoc(path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(PandocParser())
