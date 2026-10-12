"""``pdf_basic`` — zero-config local PDF parsing behind the ``pdf`` extra.

Registered at priority 10, below ``azure_di`` (40) and ``mineru`` (50), so it
acts as the fallback when no higher-fidelity PDF path is configured.
"""

import asyncio
import importlib.util
from pathlib import Path

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.pdf_basic.load import parse_pdf_basic


class PdfBasicParser(Parser):
    name: str = "pdf_basic"
    patterns: list[str] = [".pdf"]
    priority: int = 10  # below azure_di (40) and mineru (50)
    description: str = "Basic local PDF parsing (text + font-size headings) via pymupdf"

    def is_available(self) -> bool:
        return importlib.util.find_spec("pymupdf") is not None

    def unavailable_reason(self) -> str:
        return "pdf_basic: install 'ragdoc[pdf]' for basic local PDF parsing"

    async def __call__(self, path: Path) -> Document:
        # pymupdf parsing is CPU/IO blocking — run it off the event loop.
        return await asyncio.to_thread(parse_pdf_basic, path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(PdfBasicParser())
