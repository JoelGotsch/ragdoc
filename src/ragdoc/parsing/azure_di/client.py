"""Azure Document Intelligence SDK client — isolated so it can be mocked without importing automatic.py."""

di_available = True

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentContentFormat
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    di_available = False

from pathlib import Path

from ragdoc.config import get_config


def get_analyze_result(file_path: Path) -> dict:
    """Call Azure Document Intelligence on *file_path* and return the raw result dict.

    Raises:
        ImportError: If the ``azure_di`` extra is not installed.
    """
    if not di_available:
        raise ImportError("Azure SDK not available, cannot process PDF files. Please install the `azure_di` extra.")
    config = get_config()
    credential = AzureKeyCredential(config.azure_key)
    document_intelligence_client = DocumentIntelligenceClient(config.azure_endpoint, credential)
    with open(file_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", body=f, output_content_format=DocumentContentFormat.MARKDOWN
        )
    return poller.result().as_dict()
