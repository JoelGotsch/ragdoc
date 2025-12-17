# Feature Comparison

How **ragdoc** compares to other document-processing / RAG libraries, organised by module.

> Legend: **Full** = built-in, production-ready | **Partial** = limited or requires workarounds | **Plugin** = via third-party integration | **--** = not supported

---

## 1. Parsing

Convert source files into a structured in-memory representation.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **HTML** | Full | Full | Full | Full | Partial |
| **DOCX / DOC** (via Pandoc) | Full | Full | Full | Full | Full |
| **XLSX** (per-sheet config) | Full | Partial | Partial | Partial | -- |
| **PDF — Azure Document Intelligence** | Full | Full | Full | Partial | -- |
| **PDF — MinerU** (open-source OCR) | Full | -- | -- | -- | -- |
| **PDF — PyMuPDF / pdfplumber / etc.** | -- | Full | Full | Full | Full |
| **RTF / ODT** (via Pandoc) | Full | Partial | Partial | Partial | -- |
| **Structured XML (RIXML, etc.)** | -- | -- | -- | -- | -- |
| **CSV / TSV** | -- | Full | Full | Full | Full |
| **Email (.eml, .msg)** | -- | Full | Full | Full | -- |
| **PowerPoint** | -- | Full | Full | Full | Full |
| **Automatic format dispatch** | Full | -- | -- | Full | Full |
| **Parser provenance tracking** (`document.parser`) | Full | -- | -- | -- | -- |
| **Visual property preservation** (CSS-in-HTML) | Full | -- | -- | Partial | Partial |
| **Bounding box / coordinates** | Full | -- | Partial | Full | Full |
| **Image extraction** (base64, with dimensions) | Full | -- | Partial | Full | Full |
| **Table structure preservation** (HTML `<table>`) | Full | Partial | Partial | Full | Full |
| **Footnote detection & linking** | Full | -- | -- | -- | -- |
| **Cross-document refs** (`ExternalRef`) | Full | -- | -- | -- | -- |
| **Inline refs** (`<ref id="..."/>` placeholders) | Full | -- | -- | -- | -- |

### What sets ragdoc apart — Parsing

- **Unified structured model.** Every parser emits the same `Document` → `BaseElement` hierarchy (Heading, Paragraph, Table, Image, DocumentList, Footnote, RawText). Other libraries typically return flat text or loosely-typed dicts.
- **CSS-in-HTML convention.** Visual properties (font-size, font-weight, text-align) are stored as inline CSS on element HTML. This lets downstream processors reason about visual hierarchy without parser-specific fields — a heading processor that reads `font-size` works identically whether the document came from MinerU, Azure DI, or Pandoc.
- **Parser provenance.** `document.parser` records which parser produced the document. Processors can branch on this (e.g. trust native heading levels from HTML/Pandoc, but recompute them for PDF parsers).
- **Inline & external references.** Footnotes, images, and tables embedded in text are tracked as `InlineRef` placeholders (`<ref id="..." rel="footnote"/>`). Cross-document parent/child/sibling relationships use `ExternalRef`. No other library in this comparison models intra- and inter-document references as first-class citizens.

---

## 2. Processing

Transform, enrich, and clean parsed documents before splitting.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Processor pipeline** (chain N processors) | Full | -- | Full | Full | Full |
| **Heading level detection** (visual heuristics) | Full | -- | -- | Partial | Partial |
| **Heading level detection** (LLM-based) | Full | -- | -- | -- | -- |
| **Title detection & extraction** | Full | -- | -- | Full | Partial |
| **Heading normalization** (gap compaction) | Full | -- | -- | -- | -- |
| **Footnote reference linking** (heuristic) | Full | -- | -- | -- | -- |
| **Footnote reference linking** (LLM-based) | Full | -- | -- | -- | -- |
| **Image summarization** (LLM vision) | Full | -- | Partial | -- | -- |
| **Image → text_representation** (table/mermaid/MathML) | Full | -- | -- | -- | -- |
| **Custom processor ABC** (async) | Full | -- | Full | Full | Full |
| **Post-split processing** (processors on sub-documents) | Full | -- | -- | -- | -- |

### What sets ragdoc apart — Processing

- **Visual-heuristic heading refinement.** `HeadingLevelProcessor` reads CSS from element HTML (font-size, bold, centered, all-caps) and computes effective heading levels 1–6 using proportional mapping. This is critical for PDF parsers that assign flat heading levels.
- **Two-phase footnote linking.** `FootnoteProcessor` first collects all candidate reference sites (phase 1), then applies `<ref>` patches right-to-left to avoid HTML offset drift (phase 2). Candidates are scored by page proximity, context quality, and structural signals. An `LLMFootnoteResolver` strategy is available for ambiguous cases.
- **Image → structured text.** `ImageSummaryProcessor` calls a vision LLM and writes structured output into `element.text_representation`: HTML tables for tabular images, Mermaid for diagrams, MathML for formulas, plain text otherwise. This is consumed by `render_for_prompt` and differs from simple captioning.
- **Post-split processing.** Because splits are still `Document` objects, processors (e.g. section summarizers) can run *after* splitting on focused sub-documents. Other libraries typically process before splitting and cannot revisit.

---

## 3. Rendering

Convert `Document` objects into formatted text strings.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dual rendering** (prompt vs. embedding) | Full | -- | -- | -- | -- |
| **Output: HTML** | Full | -- | -- | Full | Full |
| **Output: Markdown** | Full | Full | Full | Full | Full |
| **Output: GFM** (GitHub Flavored Markdown) | Full | -- | -- | -- | -- |
| **Output: RST** (reStructuredText) | Full | -- | -- | -- | -- |
| **Output: Plain text** | Full | Full | Full | Full | Full |
| **Pluggable element renderers** (singledispatch) | Full | -- | -- | -- | -- |
| **Inline reference resolution** | Full | -- | -- | -- | -- |
| **Cross-document breadcrumbs** | Full | -- | -- | -- | -- |
| **Metadata inclusion in rendered output** | Full | Partial | Full | -- | -- |
| **Rendering-aware splitting** (token measurement) | Full | -- | -- | -- | -- |

### What sets ragdoc apart — Rendering

- **Dual-content rendering is the defining feature.** Every document can be rendered through two independent pipelines:
  - `render_for_prompt` — full-fidelity content for LLM context windows and BM25 search. Never uses `element.summary` (except as image fallback).
  - `render_for_embedding` — compact semantic text for dense vector search. Always prefers `element.summary`; falls back to full content when absent.

  No other library in this comparison separates these concerns at the rendering level.

- **Singledispatch element renderers.** Each element type (Heading, Table, Image, …) has a dedicated renderer registered via `functools.singledispatch`. To customize how tables render for embeddings, register a new handler — no subclassing required.

- **Inline reference resolution.** The renderer finds `<ref id="..." rel="..."/>` placeholders in element HTML, looks up the referenced element, renders it (with `inline=True` for compact form), and substitutes. Footnotes become `[Footnote N: text]` inline; images become their text representation.

---

## 4. Splitting

Divide documents into smaller documents respecting structure and token budgets.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Heading-based split** | Full | Partial | Partial | Full | Full |
| **Hierarchical heading split** (recursive) | Full | -- | Full | -- | -- |
| **Token-budget splitting** | Full | Full | Full | Full | -- |
| **Three-tier splitting** (heading → element → text) | Full | -- | -- | -- | -- |
| **Sentence-boundary splitting** | Full | Full | Full | Full | -- |
| **HTML-structure-aware splitting** (block tags) | Full | Partial | -- | -- | -- |
| **Parent-child tracking** (ExternalRef) | Full | -- | Partial | -- | -- |
| **Splits remain `Document` objects** | Full | -- | Partial | -- | -- |
| **Pluggable sentence splitter** (Protocol) | Full | -- | -- | -- | -- |
| **Pluggable tokenizer** | Full | Full | Full | -- | -- |
| **Overlap / sliding window** | -- | Full | Full | Full | -- |
| **Recursive character splitting** | -- | Full | Full | -- | -- |
| **Semantic splitting** (embedding-based) | -- | Partial | Full | -- | -- |

### What sets ragdoc apart — Splitting

- **Three-tier splitting.** `split_document()` cascades through three tiers: (1) hierarchical heading split at the lowest level yielding ≥ 2 parts, (2) element-level grouping that respects heading + content boundaries, (3) text-level split for oversized single elements (HTML block children → sentences → token slices). This avoids the "one big table" problem where a single element exceeds the budget.
- **Splits are `Document` objects.** Every split preserves the full structured model — elements, metadata, references. This enables post-split processing (e.g. running an LLM summarizer on each section). In LangChain/LlamaIndex, splits are typically flat strings.
- **Token-budget measurement uses rendering.** The splitter internally renders each candidate split to measure its token cost. This means the budget is accurate to the *rendered* output, not the raw element content.

---

## 5. Chunking

Materialize retrieval-ready units with dual content representations.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dual content** (prompt_content + embedding_content) | Full | -- | -- | -- | -- |
| **Content-hash-based chunk IDs** (idempotent upserts) | Full | -- | Partial | -- | -- |
| **SimpleChunker** (1 chunk per document) | Full | Full | Full | Full | Full |
| **LLMChunker** (N topic chunks per document) | Full | -- | -- | -- | -- |
| **Custom chunker ABC** | Full | -- | Full | -- | -- |
| **Metadata on chunks** | Full | Full | Full | Full | -- |
| **Pre-computed embedding slot** | Full | -- | Full | -- | -- |

### What sets ragdoc apart — Chunking

- **Ragdoc separates splitting from chunking.** Most libraries conflate these into "text splitting." In ragdoc, splitting produces sub-`Document` objects (structured, re-processable); chunking is a distinct final step that materializes `prompt_content` and `embedding_content` strings from those documents via two renderers.
- **LLMChunker — multi-topic semantic chunking.** Produces N chunks from a single document, one per LLM-identified topic. All N chunks share the same `prompt_content` (the full section) but have distinct `embedding_content` (a focused topic summary). This is analogous to *multi-vector indexing* in other frameworks but modelled as explicit chunks rather than a side-channel.
- **Content-hash chunk IDs.** Default `id` is derived from a content hash, enabling idempotent vector store upserts — re-chunking unchanged content produces identical IDs.

---

## 6. Pipeline

High-level facade composing parse → process → split → chunk.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Linear pipeline facade** | Full | Full | Full | Full | Full |
| **Bounded concurrency** (parallel file processing) | Full | Partial | Partial | -- | -- |
| **Streaming / async iterator** | Full | Full | Full | -- | -- |
| **Error handling** (skip / raise per file) | Full | Partial | Partial | -- | -- |
| **Auto-parser dispatch** (by file extension) | Full | -- | -- | Full | Full |

---

## 7. Document Merging

Combine documents from different parsers to leverage complementary strengths.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Element-alignment merge** (difflib-based) | Full | -- | -- | -- | -- |
| **Inspectable patch** (`DocumentPatch`) | Full | -- | -- | -- | -- |
| **Render-merge-reparse** (HTML-level) | Full | -- | -- | -- | -- |
| **Prefer-source per element** | Full | -- | -- | -- | -- |

### What sets ragdoc apart — Merging

Document merging is a feature unique to ragdoc in this comparison. When parsing a complex PDF, Azure DI may excel at table extraction while MinerU preserves heading hierarchy better. `compute_patch()` aligns elements from both parses using text similarity, selects the richer element per pair, and produces an inspectable `DocumentPatch` before applying. A simpler `merge_documents_html()` path operates at the rendered HTML level for quick merges.

---

## 8. Core Model

The structured document representation that underpins all stages.

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Typed element hierarchy** (Heading, Table, Image, …) | Full | -- | Partial | Full | Full |
| **Pydantic models** (validated, serializable) | Full | Partial | Partial | Partial | Full |
| **Inline CSS for visual properties** | Full | -- | -- | -- | -- |
| **Inline references** (`<ref>` placeholders) | Full | -- | -- | -- | -- |
| **Cross-document references** (`ExternalRef`) | Full | -- | Partial | -- | -- |
| **Page numbers & bounding boxes** | Full | Partial | Partial | Full | Full |
| **Document merge operator** (`doc_a | doc_b`) | Full | -- | -- | -- | -- |
| **Content hash** | Full | -- | Partial | -- | -- |
| **Element summary field** (for embedding content) | Full | -- | -- | -- | -- |
| **Image with text_representation** | Full | -- | -- | -- | -- |

---

## 9. Integrations

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **LlamaIndex TextNode ↔ Chunk** | Full | -- | N/A | Plugin | -- |
| **LangChain Document ↔ Chunk** | -- | N/A | Plugin | Full | Plugin |
| **Vector store loaders** (Pinecone, Weaviate, …) | -- | Full | Full | Partial | -- |
| **OpenAI / Azure OpenAI** (for LLM processors) | Full | Full | Full | -- | -- |

---

## 10. Configuration & DX

| Capability | ragdoc | LangChain | LlamaIndex | Unstructured | Docling |
|---|:---:|:---:|:---:|:---:|:---:|
| **Async-only API** | Full | Partial | Partial | -- | -- |
| **ContextVar-based config** (async-safe) | Full | -- | -- | -- | -- |
| **CLI** (parse, chunk) | Full | -- | Partial | Full | Full |
| **Pluggable tokenizers** (tiktoken, HF, reranker) | Full | Full | Full | -- | -- |
| **Optional extras** (azure_di, pdf_mineru) | Full | Full | Full | -- | Full |
| **Strict type hints** (no `Any`, no `**kwargs`) | Full | Partial | Partial | -- | Partial |

---

## Summary: When to choose what

| If you need… | Consider |
|---|---|
| Broadest format coverage (email, PPT, CSV, …) | LangChain, LlamaIndex, Unstructured |
| Structured document model with typed elements | ragdoc, Unstructured, Docling |
| Dual prompt/embedding content per chunk | **ragdoc** (unique) |
| LLM-enriched processing (heading/footnote/image) | **ragdoc** |
| Multi-topic semantic chunking (LLMChunker) | **ragdoc** (unique) |
| Document merging from multiple parsers | **ragdoc** (unique) |
| Visual-heuristic heading detection from CSS | **ragdoc** (unique) |
| Footnote detection and inline reference linking | **ragdoc** (unique) |
| Sliding window / overlap splitting | LangChain, LlamaIndex |
| Built-in vector store integrations | LangChain, LlamaIndex |
| Minimal dependencies / serverless deployment | Unstructured, Docling |
