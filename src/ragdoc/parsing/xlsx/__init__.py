from pathlib import Path

import pandas as pd
from pydantic import Field

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.xlsx.load import ExcelConfig, generate_document as generate_xlsx_documents


def load_excel(path: Path | str, config: ExcelConfig | None = None) -> Document:
    """Parse an Excel workbook into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    file_path = Path(path)
    document = generate_xlsx_documents(pd.ExcelFile(file_path), config or ExcelConfig())
    document.metadata["filename"] = file_path.name
    document.source_path = str(file_path)
    return document


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
