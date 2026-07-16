# Integrations API

Integration helpers for third-party frameworks.

---

## Vector Stores

Concrete implementations of the [`VectorStore`](pipeline.md#vectorstore) protocol.
See the [Pipeline Guide — Qdrant](../guide/pipeline.md#scenario-c-qdrant-vector-store)
for usage examples.

### QdrantVectorStore

::: ragdoc.integrations.vector_stores.qdrant.QdrantVectorStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### QdrantIndex

::: ragdoc.integrations.vector_stores.qdrant.QdrantIndex
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ServerSideVector

::: ragdoc.integrations.vector_stores.qdrant.ServerSideVector
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### register_indexes_from_type

::: ragdoc.integrations.vector_stores.qdrant.register_indexes_from_type
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

---

## Document Stores

Concrete implementations of the [`DocumentStore`](pipeline.md#documentstore) protocol.
See the [Pipeline Guide — Scenario E](../guide/pipeline.md#scenario-e-two-stage-with-a-documentstore).

### QdrantDocumentStore

::: ragdoc.integrations.document_stores.qdrant.QdrantDocumentStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentTooLargeError

::: ragdoc.integrations.document_stores.qdrant.DocumentTooLargeError
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

---

## Extraction Stores

Qdrant adapters for the extraction-layer store protocols
([`MentionStore`](extraction.md#mentionstore), [`EntityStore`](extraction.md#entitystore),
[`GraphStore`](extraction.md#graphstore)). See the
[Extraction Guide](../guide/extraction.md).

### QdrantMentionStore

::: ragdoc.integrations.mention_stores.qdrant.QdrantMentionStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### QdrantEntityStore

::: ragdoc.integrations.entity_stores.qdrant.QdrantEntityStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### QdrantGraphStore

::: ragdoc.integrations.graph_stores.qdrant.QdrantGraphStore
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
