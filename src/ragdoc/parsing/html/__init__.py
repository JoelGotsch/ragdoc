from pathlib import Path

from ragdoc.document import Document
from ragdoc.parsing.html.load import HTML, generate_document as generate_html_document
from ragdoc.parsing.parser import Parser


def load_html(path: Path | str) -> Document:
    """Parse an HTML file into a Document.

    Provenance (``source_path``, ``metadata["filename"]``) is stamped centrally by
    :func:`ragdoc.parsing.load` — not here.
    """
    return generate_html_document(HTML.from_file(Path(path)))


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class HTMLParser(Parser):
    name: str = "html"
    patterns: list[str] = [".html", ".htm"]
    description: str = "HTML files"

    async def __call__(self, path: Path) -> Document:
        return load_html(path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(HTMLParser())
