# Extraction API

Typed structured extraction and knowledge graphs. Three layers: **mentions** (raw,
provenance-tagged extraction occurrences), **entities** (canonical, deduplicated records
produced by resolution), and the **typed knowledge graph** (nodes + edges against a
`GraphSchema`). See the [Extraction Guide](../guide/extraction.md) for the workflow.

Requires the `extraction` extra for the LLM extractors and settings
(`pip install 'ragdoc[extraction]'`); the data models and stores import without it.

## Layer 1 — Mentions and extractors

### Extractor

::: ragdoc.extraction.extractor.Extractor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Mention

::: ragdoc.extraction.mention.Mention
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### StructuredExtractor

::: ragdoc.extraction.structured.StructuredExtractor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ExtractionSettings

::: ragdoc.extraction.structured.ExtractionSettings
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### as_processor

::: ragdoc.extraction.extractor.as_processor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### mint_mention_id

::: ragdoc.extraction.mention.mint_mention_id
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### finalize_mention

::: ragdoc.extraction.mention.finalize_mention
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_extraction_messages

::: ragdoc.extraction.structured.build_extraction_messages
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Mention stores

### MentionStore

::: ragdoc.extraction.stores.MentionStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LocalMentionStore

::: ragdoc.extraction.stores.LocalMentionStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### MentionStorePipeline

::: ragdoc.extraction.pipeline.MentionStorePipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Dates

### FuzzyDate

::: ragdoc.extraction.dates.FuzzyDate
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Precision

::: ragdoc.extraction.dates.Precision
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

## Layer 2 — Entities and resolution

### Entity

::: ragdoc.extraction.entity.Entity
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### EntityStore

::: ragdoc.extraction.entity.EntityStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LocalEntityStore

::: ragdoc.extraction.entity.LocalEntityStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### EntityResolutionPipeline

::: ragdoc.extraction.resolution.EntityResolutionPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ResolutionResult

::: ragdoc.extraction.resolution.ResolutionResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ReviewGroup

::: ragdoc.extraction.resolution.ReviewGroup
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ReviewResult

::: ragdoc.extraction.resolution.ReviewResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_entity_embedder

::: ragdoc.extraction.resolution.build_entity_embedder
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### make_llm_reviewer

::: ragdoc.extraction.resolution.make_llm_reviewer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### default_identity_text

::: ragdoc.extraction.resolution.default_identity_text
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Entity queries

### EntityQuery

::: ragdoc.extraction.query.EntityQuery
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DateRange

::: ragdoc.extraction.query.DateRange
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### entity_matches

::: ragdoc.extraction.query.entity_matches
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### filter_entities

::: ragdoc.extraction.query.filter_entities
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Layer 3 — Typed knowledge graph

### GraphSchema

::: ragdoc.extraction.schema.GraphSchema
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### EdgeRef

::: ragdoc.extraction.schema.EdgeRef
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### KnowledgeGraphExtractor

::: ragdoc.extraction.kg.KnowledgeGraphExtractor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### KnowledgeGraphResolutionPipeline

::: ragdoc.extraction.kg_resolution.KnowledgeGraphResolutionPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### KGResolutionResult

::: ragdoc.extraction.kg_resolution.KGResolutionResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### GraphStore

::: ragdoc.extraction.graph_store.GraphStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### LocalGraphStore

::: ragdoc.extraction.graph_store.LocalGraphStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_node_union

::: ragdoc.extraction.schema.build_node_union
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_edge_union

::: ragdoc.extraction.schema.build_edge_union
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### kind_of

::: ragdoc.extraction.schema.kind_of
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### allowed_pattern_kinds

::: ragdoc.extraction.schema.allowed_pattern_kinds
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### render_patterns_prompt

::: ragdoc.extraction.schema.render_patterns_prompt
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_kg_messages

::: ragdoc.extraction.kg.build_kg_messages
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### build_graph_batch_model

::: ragdoc.extraction.kg.build_graph_batch_model
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
