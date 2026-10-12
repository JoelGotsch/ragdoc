# Tutorials

Interactive [marimo](https://marimo.io) notebooks for the ragdoc library. The published
pages are WebAssembly exports of the notebooks in `docs/notebooks/`. Each notebook is also
a plain Python file and is executed by the test suite, so the code shown is known to run
against the current library.

To run and edit a notebook locally (after `uv sync --all-extras --group docs`):

```bash
uv run marimo edit docs/notebooks/<notebook>.py
```

---

## Getting started

| Notebook | Description |
|---|---|
| [Quickstart](../notebooks/quickstart/) | Parse, split, and chunk a document end-to-end |
| [Pipeline Walkthrough](../notebooks/pipeline_walkthrough/) | Stage-by-stage walkthrough using a real MinerU-parsed paper (LLM stages need an API key) |

## Guide notebooks

| Notebook | Description |
|---|---|
| [Document Model](../notebooks/document_model/) | `Document`, element types, inline refs, cross-document links, `concat_documents` |
| [Parsing](../notebooks/parsing/) | `load()`, the parser registry, and the built-in parsers |
| [Rendering](../notebooks/rendering/) | `render_for_prompt` vs `render_raw`, output formats |
| [Splitting](../notebooks/splitting/) | `split_by_headings`, `split_hierarchical`, post-split processing |
| [Chunking](../notebooks/chunking/) | Building `Chunk` objects by hand; how `SimpleChunker` / `LLMChunker` fill the two content fields |
| [Pipeline](../notebooks/pipeline/) | `DocumentPipeline`, streaming, and incremental sync: `plan()` / `apply()` / `run()`, `ChangeSet`, `DocumentStorePipeline` |
| [Merging](../notebooks/merging/) | Two-parser merge with `compute_patch` and `compute_html_patch` |

## Advanced

| Notebook | Description |
|---|---|
| [Qdrant Sync Pipeline](../notebooks/qdrant_pipeline/) | Incremental sync into Qdrant with `QdrantVectorStore` / `QdrantDocumentStore` (runs offline against in-process Qdrant) |
| [Extraction & Knowledge Graphs](../notebooks/extraction/) | Mention → entity workflow end-to-end with a stub extractor (no API key) |
| [Custom Elements](../notebooks/custom_elements/) | Define custom element types, parsers, and renderers |
| [MinerU Example](../notebooks/mineru_example/) | Minimal MinerU parser usage |
