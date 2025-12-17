# Tutorials

Interactive notebooks for the ragdoc library. Each notebook runs fully in the browser
via WebAssembly — no installation required.

To run locally instead:

```bash
marimo edit docs/notebooks/<notebook>.py
```

---

## Getting started

| Notebook | Description |
|---|---|
| [Quickstart](../notebooks/quickstart/) | Parse, split, and chunk a document end-to-end |
| [Pipeline Walkthrough](../notebooks/pipeline_walkthrough/) | Full walkthrough using a real MinerU-parsed document |

## Guide notebooks

| Notebook | Description |
|---|---|
| [Document Model](../notebooks/document_model/) | `Document`, element types, inline refs, cross-document links |
| [Parsing](../notebooks/parsing/) | All parser types: Pandoc, HTML, Excel, Azure DI |
| [Rendering](../notebooks/rendering/) | `render_for_prompt` vs `render_raw`, output formats |
| [Splitting](../notebooks/splitting/) | `split_by_headings`, `split_hierarchical`, post-split processing |
| [Chunking](../notebooks/chunking/) | Creating `Chunk` objects with `SimpleChunker` and `LLMChunker` |
| [Pipeline](../notebooks/pipeline/) | `DocumentPipeline`, `VectorStorePipeline`, streaming |
| [Merging](../notebooks/merging/) | Two-parser merge with `compute_patch` and `compute_html_patch` |

## Advanced

| Notebook | Description |
|---|---|
| [Qdrant Sync Pipeline](../notebooks/qdrant_sync/) | Production incremental sync: MinerU + LLMChunker + QdrantVectorStore |
| [Custom Elements](../notebooks/custom_elements/) | Define custom element types, parsers, and renderers |
| [MinerU Example](../notebooks/mineru_example/) | Minimal MinerU parser usage |
