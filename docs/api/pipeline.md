# Pipeline API

The `ragdoc.pipeline` module provides the high-level facade over all four
pipeline stages.  The pipeline is split at the two sync boundaries —
`IngestPipeline` (parse → process) and `ChunkPipeline` (split → chunk) — with
`DocumentPipeline` composing both for the direct path.  See the
[Pipeline Guide](../guide/pipeline.md) for usage examples.

## IngestPipeline

::: ragdoc.pipeline.linear.IngestPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## ChunkPipeline

::: ragdoc.pipeline.linear.ChunkPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

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

## Parser

The canonical `Parser` base class lives in `ragdoc.parsing`.

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

::: ragdoc.pipeline.sync.UpdateResult
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
