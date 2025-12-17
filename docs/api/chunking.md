# Chunking API

The `Chunk` model is the final output of the ragdoc pipeline — a flat, self-contained
record ready for loading into a vector store.

## Chunk

::: ragdoc.chunking.chunk.Chunk
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Chunk provenance

Chunk provenance is resolved uniformly (one fallback chain) and chunk ids are minted
pipeline-side by `ChunkPipeline.run`.

### ChunkProvenance

::: ragdoc.chunking.provenance.ChunkProvenance
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### resolve_chunk_provenance

::: ragdoc.chunking.provenance.resolve_chunk_provenance
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### mint_chunk_id

::: ragdoc.chunking.provenance.mint_chunk_id
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
