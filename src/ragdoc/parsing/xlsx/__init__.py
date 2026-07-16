import asyncio
import importlib.util
from pathlib import Path

from pydantic import Field

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.xlsx.load import (
    _XLSX_IMPORT_ERROR,
    ExcelConfig,
    generate_document as generate_xlsx_documents,
)


def load_excel(path: Path | str, config: ExcelConfig | None = None) -> Document:
    """Parse an Excel workbook into a Document.

    Provenance (``source_path``, ``metadata["filename"]``) is stamped centrally by
    :func:`ragdoc.parsing.load` — not here.

    Raises:
        ImportError: If pandas is not installed (the ``xlsx`` extra).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(_XLSX_IMPORT_ERROR) from exc
    return generate_xlsx_documents(pd.ExcelFile(Path(path)), config or ExcelConfig())


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class XlsxParser(Parser):
    name: str = "xlsx"
    patterns: list[str] = [".xlsx"]
    description: str = "Excel spreadsheets"
    config: ExcelConfig = Field(default_factory=ExcelConfig, description="Per-parser Excel parsing configuration.")

    def is_available(self) -> bool:
        return importlib.util.find_spec("pandas") is not None

    def unavailable_reason(self) -> str:
        return "xlsx: install 'ragdoc[xlsx]' for Excel parsing"

    async def __call__(self, path: Path) -> Document:
        # pandas parses the workbook synchronously (CPU + file I/O) — run it off the event loop.
        return await asyncio.to_thread(load_excel, path, self.config)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(XlsxParser())
