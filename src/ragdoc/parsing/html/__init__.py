import asyncio
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
        # File read + bs4 parse + (with download_images) one synchronous httpx.get per remote
        # image — run off the event loop. to_thread copies the context, so the active
        # configure(...) settings are visible in the worker thread.
        return await asyncio.to_thread(load_html, path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(HTMLParser())
