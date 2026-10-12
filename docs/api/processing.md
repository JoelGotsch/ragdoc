# Processing API

Processors transform `Document` objects. They run between parsing and splitting in the pipeline.

## Base Classes

### DocumentProcessor

::: ragdoc.processing.base.DocumentProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ProcessingPipeline

::: ragdoc.processing.base.ProcessingPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Filters

### EmptyDocumentFilter

::: ragdoc.processing.filters.EmptyDocumentFilter
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Heading Processors

### HeadingLevelProcessor

::: ragdoc.processing.heading.HeadingLevelProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### TitleDetectionProcessor

::: ragdoc.processing.heading.TitleDetectionProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### HeadingNormalizationProcessor

::: ragdoc.processing.heading.HeadingNormalizationProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### HeadingVisualInfo

::: ragdoc.processing.heading.HeadingVisualInfo
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Heading Utility Functions

### extract_font_size

::: ragdoc.processing.heading.extract_font_size
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### is_centered

::: ragdoc.processing.heading.is_centered
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### is_bold

::: ragdoc.processing.heading.is_bold
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### is_all_caps

::: ragdoc.processing.heading.is_all_caps
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### compute_size_to_level_mapping

::: ragdoc.processing.heading.compute_size_to_level_mapping
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### normalize_heading_levels

::: ragdoc.processing.heading.normalize_heading_levels
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## LLM Heading Resolution

### LLMHeadingResolver

::: ragdoc.processing.heading_llm.LLMHeadingResolver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LLMHeadingResolverSettings

::: ragdoc.processing.heading_llm.LLMHeadingResolverSettings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Footnote Processing

### FootnoteProcessor

::: ragdoc.processing.footnote.FootnoteProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### FootnoteResolver

::: ragdoc.processing.footnote.FootnoteResolver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SimpleFootnoteResolver

::: ragdoc.processing.footnote.SimpleFootnoteResolver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LLMFootnoteResolver

::: ragdoc.processing.footnote.LLMFootnoteResolver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Summarization Processors

### ImageSummaryProcessor

::: ragdoc.processing.summary_image.ImageSummaryProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### openai_image_summarizer

::: ragdoc.processing.summary_image.openai_image_summarizer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ImageSummarizeFn

::: ragdoc.processing.summary_image.ImageSummarizeFn
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### build_image_messages

::: ragdoc.processing.summary_image.build_image_messages
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentSummarizerProcessor

::: ragdoc.processing.summary_document.DocumentSummarizerProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentSummarizerSettings

::: ragdoc.processing.summary_document.DocumentSummarizerSettings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_summary_messages

::: ragdoc.processing.summary_document.build_summary_messages
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### pack_summaries

::: ragdoc.processing.summary_document.pack_summaries
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Summary Models

### ImageSummary

::: ragdoc.processing.summary_base.ImageSummary
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentSummary

::: ragdoc.processing.summary_base.DocumentSummary
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Document Dump

### DocumentDumpProcessor

::: ragdoc.processing.dump.DocumentDumpProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### FileNamer

::: ragdoc.processing.dump.FileNamer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### default_file_namer

::: ragdoc.processing.dump.default_file_namer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

