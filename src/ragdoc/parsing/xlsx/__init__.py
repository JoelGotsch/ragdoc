from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from ragdoc.document import Document
from ragdoc.parsing.base import load_file
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.xlsx.load import ExcelConfig, generate_document as generate_xlsx_documents


class ExcelSource(BaseModel):
    file_path: str | Path
    config: ExcelConfig = Field(default_factory=ExcelConfig)

    model_config = {"arbitrary_types_allowed": True}


@load_file.register
def load_excel(file_obj: ExcelSource) -> Document:
    """Parse an Excel workbook into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    document = generate_xlsx_documents(pd.ExcelFile(file_obj.file_path), file_obj.config)
    document.metadata["filename"] = Path(file_obj.file_path).name
    document.source_path = str(file_obj.file_path)
    return document


# Backwards-compatible alias removed — use ExcelSource
ExcelPackage = ExcelSource


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class XlsxParser(Parser):
    name: str = "xlsx"
    patterns: list[str] = [".xlsx"]
    description: str = "Excel spreadsheets"

    async def __call__(self, path: Path) -> Document:
        return load_excel(ExcelSource(file_path=path))


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(XlsxParser())
