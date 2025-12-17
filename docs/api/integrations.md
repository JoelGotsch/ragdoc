# Integrations API

Integration helpers for third-party frameworks.

---

## Vector Stores

Concrete implementations of the [`VectorStore`](pipeline.md#VectorStore) protocol.
See the [Pipeline Guide — Qdrant](../guide/pipeline.md#scenario-c--qdrant-vector-store)
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

### register_indexes_from_type

::: ragdoc.integrations.vector_stores.qdrant.register_indexes_from_type
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

---

## LlamaIndex

Helpers for converting between `Chunk` and LlamaIndex node dictionaries.

### node_dict_to_document_fragment

::: ragdoc.integrations.llama_index.node_dict_to_document_fragment
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### document_fragment_to_node_dict

::: ragdoc.integrations.llama_index.document_fragment_to_node_dict
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
