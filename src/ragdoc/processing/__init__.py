"""
Document processing infrastructure.

This module provides processors that transform Document objects after parsing.
Processors are used for tasks like:
- Heading level refinement based on visual properties
- Title detection and extraction
- LLM-based heading resolution
- Summarization of images and documents

Architecture:
-------------
- DocumentProcessor: Async ABC for document transformation
- ProcessingPipeline: Chain multiple processors together

Processors operate on Document objects directly, making them universal
across all parsers (MinerU, Azure DI, HTML, etc.).
"""

from ragdoc.processing.base import (
    DocumentProcessor,
    ProcessingPipeline,
)
from ragdoc.processing.heading import (
    HeadingLevelProcessor,
    HeadingVisualInfo,
    TitleDetectionProcessor,
    # Standalone functions
    extract_font_size,
    is_centered,
    is_bold,
    is_all_caps,
    compute_size_to_level_mapping,
    SizeToLevelMapper,
)
try:
    from ragdoc.processing.heading_llm import (
        LLMHeadingResolver,
        LLMHeadingResolverSettings,
        HeadingInfo,
    )
except ImportError:
    pass  # pydantic_settings not installed; LLMHeadingResolver unavailable
from ragdoc.processing.footnote import (
    FootnoteCandidate,
    FootnoteResolver,
    SimpleFootnoteResolver,
    LLMFootnoteResolver,
    FootnoteProcessor,
    SyncFootnoteProcessor,
    build_footnote_pattern,
    find_footnote_candidates,
    score_footnote_candidates,
    apply_ref_patches,
)
from ragdoc.processing.summary_base import ImageSummary
from ragdoc.processing.summary_image import (
    ImageSummaryProcessor,
    ImageSummarizeFn,
    build_image_messages,
    openai_image_summarizer,
    DEFAULT_TRANSFORMATIONS,
)
from ragdoc.processing.dump import DocumentDumpProcessor, FileNamer, default_file_namer
from ragdoc.processing.filters import EmptyDocumentFilter

__all__ = [
    # Base classes
    "DocumentProcessor",
    "ProcessingPipeline",
    # Heading processors
    "HeadingLevelProcessor",
    "HeadingVisualInfo",
    "TitleDetectionProcessor",
    # Standalone functions
    "extract_font_size",
    "is_centered",
    "is_bold",
    "is_all_caps",
    "compute_size_to_level_mapping",
    "SizeToLevelMapper",
    # LLM heading processor
    "LLMHeadingResolver",
    "LLMHeadingResolverSettings",
    "HeadingInfo",
    # Footnote processors
    "FootnoteCandidate",
    "FootnoteResolver",
    "SimpleFootnoteResolver",
    "LLMFootnoteResolver",
    "FootnoteProcessor",
    "SyncFootnoteProcessor",
    "find_footnote_candidates",
    "apply_ref_patches",
    # Summary result models
    "ImageSummary",
    # Image summary
    "ImageSummarizeFn",
    "build_image_messages",
    "openai_image_summarizer",
    "DEFAULT_TRANSFORMATIONS",
    "ImageSummaryProcessor",
    # Document dump
    "DocumentDumpProcessor",
    "FileNamer",
    "default_file_namer",
    # Filters
    "EmptyDocumentFilter",
]
