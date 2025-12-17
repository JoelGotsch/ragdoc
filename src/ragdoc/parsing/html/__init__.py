from pathlib import Path

from pydantic import BaseModel

from ragdoc.document import Document
from ragdoc.parsing.base import load_file
from ragdoc.parsing.html.load import HTML, generate_document as generate_html_document
from ragdoc.parsing.parser import Parser


class HTMLSource(BaseModel):
    file_path: str | Path


@load_file.register
def load_html(file_obj: HTMLSource) -> Document:
    """Parse an HTML file into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    file_path = Path(file_obj.file_path)
    document = generate_html_document(HTML.from_file(file_path))
    document.metadata["filename"] = file_path.name
    document.source_path = str(file_path)
    return document


# Backwards-compatible alias removed — use HTMLSource
HTMLFile = HTMLSource


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class HTMLParser(Parser):
    name: str = "html"
    patterns: list[str] = [".html", ".htm"]
    description: str = "HTML files"

    async def __call__(self, path: Path) -> Document:
        return load_html(HTMLSource(file_path=path))


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(HTMLParser())
