import json
from pathlib import Path

from pydantic import BaseModel, Field

from ragdoc.document import Document
from ragdoc.parsing.azure_di.load import AzureDIBundle, generate_document_azure_di
from ragdoc.parsing.parser import Parser


class AzureJSONFile(BaseModel):
    """Internal model describing an Azure DI JSON result file on disk (not part of the public API)."""

    file_path: str | Path
    source_path: str | Path | None = Field(
        default=None, description="Path to the source pdf file which lead to this json being produced"
    )


class AzureAnalyzeRun(BaseModel):
    """Internal model carrying an in-memory Azure DI analyze payload (not part of the public API)."""

    analyze_dict: dict
    source_path: str | Path | None = Field(
        default=None, description="Path to the source pdf file which lead to this json being produced"
    )


def load_azure_json(file_obj: AzureJSONFile) -> Document:
    """Parse an Azure DI JSON result file into a Document.

    Provenance (``source_path``, ``metadata["filename"]``) is stamped centrally by
    :func:`ragdoc.parsing.load` — not here.
    """
    file_path = Path(file_obj.file_path)
    bundle_source = Path(file_obj.source_path) if file_obj.source_path is not None else None
    with open(file_path) as fh:
        azure_bundle = AzureDIBundle(analyze_result=json.load(fh), source_path=bundle_source)
    return generate_document_azure_di(azure_bundle)


def load_azure_analyze_result(analyze_run: AzureAnalyzeRun) -> Document:
    """Parse an in-memory Azure DI analyze result into a Document. Sets ``source_path`` and ``metadata["filename"]`` from ``source_path``."""
    source_path = Path(analyze_run.source_path) if analyze_run.source_path is not None else None
    azure_bundle = AzureDIBundle(analyze_result=analyze_run.analyze_dict, source_path=source_path)
    document = generate_document_azure_di(azure_bundle)
    if source_path:
        document.metadata["filename"] = source_path.name
        document.source_path = str(source_path)
    return document


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def parse_azure_json(path: Path) -> Document:
    """Parse an Azure DI JSON file (standalone function for reuse/testing)."""
    return load_azure_json(AzureJSONFile(file_path=path))


def parse_pdf_azure_di(path: Path) -> Document:
    """Parse a PDF via Azure Document Intelligence (standalone function for reuse/testing).

    Requires the ``azure_di`` extra and credentials configured via
    :func:`~ragdoc.config.configure`.
    """
    from ragdoc.parsing.azure_di import client as _client

    analyze_dict = _client.get_analyze_result(path)
    return load_azure_analyze_result(AzureAnalyzeRun(analyze_dict=analyze_dict, source_path=path))


class AzureJSONParser(Parser):
    name: str = "azure_json"
    patterns: list[str] = [".azure.json"]
    description: str = "Azure Document Intelligence JSON result files"

    async def __call__(self, path: Path) -> Document:
        return parse_azure_json(path)


class AzureDIParser(Parser):
    name: str = "azure_di"
    patterns: list[str] = [".pdf"]
    priority: int = 40
    description: str = "PDF via Azure Document Intelligence"

    async def __call__(self, path: Path) -> Document:
        return parse_pdf_azure_di(path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(AzureJSONParser())
    register_parser(AzureDIParser())
