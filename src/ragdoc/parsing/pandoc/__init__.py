from pathlib import Path

from pydantic import BaseModel

from ragdoc.document import Document
from ragdoc.parsing.base import load_file
from ragdoc.parsing.pandoc.load import PandocHTML, generate_document as generate_pandoc_document
from ragdoc.parsing.parser import Parser


class PandocFile(BaseModel):
    file_path: str


class WordFile(BaseModel):
    file_path: str


@load_file.register
def load_pandoc(file_obj: PandocFile) -> Document:
    """Parse a file via Pandoc into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    file_path = Path(file_obj.file_path)
    document = generate_pandoc_document(PandocHTML.from_file(file_path))
    document.metadata["filename"] = file_path.name
    document.source_path = str(file_path)
    return document


@load_file.register
def load_wordfile(file_obj: WordFile) -> Document:
    pandoc_file = PandocFile(file_path=file_obj.file_path)
    return load_file(pandoc_file)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class PandocParser(Parser):
    name: str = "pandoc"
    patterns: list[str] = [".docx", ".doc"]
    description: str = "Word documents via pandoc"

    async def __call__(self, path: Path) -> Document:
        return load_pandoc(PandocFile(file_path=str(path)))


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(PandocParser())
