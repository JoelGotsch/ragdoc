# Pipeline API

The `ragdoc.pipeline` module provides the high-level facade over all four
pipeline stages.  See the [Pipeline Guide](../guide/pipeline.md) for usage
examples.

## DocumentPipeline

::: ragdoc.pipeline.linear.DocumentPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## PipelineResult

::: ragdoc.pipeline.linear.PipelineResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## TokenSplitter

::: ragdoc.pipeline.splitter.TokenSplitter
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## AutoParser

::: ragdoc.pipeline.parser.AutoParser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Parser

The canonical `Parser` protocol lives in `ragdoc.parsing`. It is re-exported from
`ragdoc.pipeline.parser` for compatibility, but new code should import from `ragdoc.parsing`.

::: ragdoc.parsing.parser.Parser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## VectorStorePipeline

::: ragdoc.pipeline.vectorstore.VectorStorePipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## UpdateResult

::: ragdoc.pipeline.vectorstore.UpdateResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## VectorStore

::: ragdoc.pipeline.stores.VectorStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
