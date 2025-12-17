"""ragdoc — parse, process, split, and chunk documents for LLM/RAG pipelines."""

import logging

logging.getLogger("ragdoc").addHandler(logging.NullHandler())

from ragdoc.chunking import Chunk, LLMChunker, SimpleChunker
from ragdoc.config import RagdocConfig, configure
from ragdoc.document import (
    BaseElement,
    Document,
    DocumentList,
    ExternalRef,
    Footnote,
    Heading,
    Image,
    InlineRef,
    Paragraph,
    RawText,
    Table,
)
from ragdoc.metadata import BaseMetadata
from ragdoc.parsing import load
from ragdoc.pipeline import (
    ChangeSet,
    DocumentPipeline,
    DocumentStorePipeline,
    TokenSplitter,
    UpdateResult,
    VectorStorePipeline,
)
from ragdoc.splitting import split_document

__all__ = [
    "BaseElement",
    "BaseMetadata",
    "ChangeSet",
    "Chunk",
    "Document",
    "DocumentList",
    "DocumentPipeline",
    "DocumentStorePipeline",
    "ExternalRef",
    "Footnote",
    "Heading",
    "Image",
    "InlineRef",
    "LLMChunker",
    "Paragraph",
    "RagdocConfig",
    "RawText",
    "SimpleChunker",
    "Table",
    "TokenSplitter",
    "UpdateResult",
    "VectorStorePipeline",
    "configure",
    "load",
    "split_document",
]
