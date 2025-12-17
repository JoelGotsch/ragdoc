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
    SizeToLevelMapper,
    TitleDetectionProcessor,
    compute_size_to_level_mapping,
    # Standalone functions
    extract_font_size,
    is_all_caps,
    is_bold,
    is_centered,
)

try:
    from ragdoc.processing.heading_llm import (
        HeadingInfo,
        LLMHeadingResolver,
        LLMHeadingResolverSettings,
    )
except ImportError:
    pass  # pydantic_settings not installed; LLMHeadingResolver unavailable
from ragdoc.processing.dump import DocumentDumpProcessor, FileNamer, default_file_namer
from ragdoc.processing.filters import EmptyDocumentFilter
from ragdoc.processing.footnote import (
    FootnoteCandidate,
    FootnoteProcessor,
    FootnoteResolver,
    LLMFootnoteResolver,
    SimpleFootnoteResolver,
    apply_ref_patches,
    build_footnote_pattern,
    find_footnote_candidates,
    score_footnote_candidates,
)
from ragdoc.processing.summary_base import DocumentSummary, ImageSummary
from ragdoc.processing.summary_document import (
    SUMMARY_SYSTEM_PROMPT,
    DocumentSummarizerProcessor,
    DocumentSummarizerSettings,
    build_summary_messages,
    pack_summaries,
)
from ragdoc.processing.summary_image import (
    DEFAULT_TRANSFORMATIONS,
    ImageSummarizeFn,
    ImageSummaryProcessor,
    build_image_messages,
    openai_image_summarizer,
)

__all__ = [
    "DEFAULT_TRANSFORMATIONS",
    "SUMMARY_SYSTEM_PROMPT",
    # Document dump
    "DocumentDumpProcessor",
    # Base classes
    "DocumentProcessor",
    # Document summary
    "DocumentSummarizerProcessor",
    "DocumentSummarizerSettings",
    "DocumentSummary",
    # Filters
    "EmptyDocumentFilter",
    "FileNamer",
    # Footnote processors
    "FootnoteCandidate",
    "FootnoteProcessor",
    "FootnoteResolver",
    "HeadingInfo",
    # Heading processors
    "HeadingLevelProcessor",
    "HeadingVisualInfo",
    # Image summary
    "ImageSummarizeFn",
    # Summary result models
    "ImageSummary",
    "ImageSummaryProcessor",
    "LLMFootnoteResolver",
    # LLM heading processor
    "LLMHeadingResolver",
    "LLMHeadingResolverSettings",
    "ProcessingPipeline",
    "SimpleFootnoteResolver",
    "SizeToLevelMapper",
    "TitleDetectionProcessor",
    "apply_ref_patches",
    "build_image_messages",
    "build_summary_messages",
    "compute_size_to_level_mapping",
    "default_file_namer",
    # Standalone functions
    "extract_font_size",
    "find_footnote_candidates",
    "is_all_caps",
    "is_bold",
    "is_centered",
    "openai_image_summarizer",
    "pack_summaries",
]
