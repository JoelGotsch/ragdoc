from pathlib import Path

import pandas as pd
from pydantic import Field

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.xlsx.load import ExcelConfig, generate_document as generate_xlsx_documents


def load_excel(path: Path | str, config: ExcelConfig | None = None) -> Document:
    """Parse an Excel workbook into a Document.

    Provenance (``source_path``, ``metadata["filename"]``) is stamped centrally by
    :func:`ragdoc.parsing.load` — not here.
    """
    return generate_xlsx_documents(pd.ExcelFile(Path(path)), config or ExcelConfig())


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class XlsxParser(Parser):
    name: str = "xlsx"
    patterns: list[str] = [".xlsx"]
    description: str = "Excel spreadsheets"
    config: ExcelConfig = Field(default_factory=ExcelConfig, description="Per-parser Excel parsing configuration.")

    async def __call__(self, path: Path) -> Document:
        return load_excel(path, self.config)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(XlsxParser())
