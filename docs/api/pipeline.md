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

The canonical `Parser` base class lives in `ragdoc.parsing` — see the
[Parsing API](parsing.md#parser) (it is re-exported from `ragdoc.pipeline` for
convenience).

## Sync pipelines

### VectorStorePipeline

::: ragdoc.pipeline.vectorstore.VectorStorePipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentStorePipeline

::: ragdoc.pipeline.document_store_pipeline.DocumentStorePipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Sync engine

The generic plan/apply/run core shared by all sync pipelines. See the
[Sync Engine Guide](../guide/sync-engine.md).

### SyncEngine

::: ragdoc.pipeline.sync.SyncEngine
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SourceSyncStore

::: ragdoc.pipeline.sync.SourceSyncStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SyncSource

::: ragdoc.pipeline.sync.SyncSource
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SyncPlanInput

::: ragdoc.pipeline.sync.SyncPlanInput
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SourceOutcome

::: ragdoc.pipeline.sync.SourceOutcome
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### UpdateResult

::: ragdoc.pipeline.sync.UpdateResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### file_hash

::: ragdoc.pipeline.sync.file_hash
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Change sets

### ChangeSet

::: ragdoc.pipeline.changeset.ChangeSet
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SourceChange

::: ragdoc.pipeline.changeset.SourceChange
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Stores

### VectorStore

::: ragdoc.pipeline.stores.VectorStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentStore

::: ragdoc.pipeline.stores.DocumentStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SourceState

::: ragdoc.pipeline.stores.SourceState
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LocalDocumentStore

::: ragdoc.pipeline.local_document_store.LocalDocumentStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Embedders

### Embedder

::: ragdoc.pipeline.embedders.Embedder
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### EmbedderConfig

::: ragdoc.pipeline.embedders.EmbedderConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### OpenAIEmbedder

::: ragdoc.pipeline.embedders.OpenAIEmbedder
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### embedding_content_text

::: ragdoc.pipeline.embedders.embedding_content_text
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### prompt_content_text

::: ragdoc.pipeline.embedders.prompt_content_text
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
