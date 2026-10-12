# Installation

## Requirements

- Python 3.10 – 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install with uv

The base install is deliberately lean: it parses HTML and Word `.docx` (via pandoc),
processes, splits, and chunks with no ML-sized dependencies. Everything else is
an extra:

```bash
# Core library (HTML + pandoc parsing, processing, splitting, chunking)
uv add ragdoc

# LLM features: LLMChunker, LLM processors, OpenAI embedders (openai + pillow)
uv add "ragdoc[llm]"

# Basic local PDF parsing — zero config (pymupdf)
uv add "ragdoc[pdf]"

# Excel parsing (pandas + openpyxl)
uv add "ragdoc[xlsx]"

# HuggingFace tokenizers, e.g. RerankerTokenizer (transformers)
uv add "ragdoc[tokenizers]"

# Azure Document Intelligence PDF parsing (includes pymupdf for figure extraction)
uv add "ragdoc[azure-di]"

# MinerU middle-JSON PDF parsing
uv add "ragdoc[pdf-mineru]"

# Qdrant vector store support
uv add "ragdoc[qdrant]"

# Knowledge-graph / structured extraction (edtf + numpy)
uv add "ragdoc[extraction]"

# Several extras at once
uv add "ragdoc[llm,pdf,qdrant]"
```

If a feature needs an extra you don't have, ragdoc fails loudly with the exact
install command — e.g. parsing a `.pdf` on a base install raises at parser
*resolution* time with `install 'ragdoc[pdf]' for the basic local parser`.

## PDF parsing options

| Extra | Parser | Fidelity | Setup |
|---|---|---|---|
| `pdf` | `pdf_basic` (pymupdf) | Text + font-size headings | zero config |
| `azure-di` | `azure_di` | Full layout, tables, figures | Azure credentials required |
| `pdf-mineru` | `mineru` | Full layout via MinerU `_middle.json` | run MinerU separately |

For a `.pdf` path the registry picks the highest-priority *available* parser
(azure_di 40 > pdf_basic 10); unavailable ones (missing package or credentials)
are skipped at resolve time. MinerU is not a `.pdf` parser: run MinerU yourself
and `load()` the `*_middle.json` it writes.

## Install for development

Clone the repository and install all dependencies including docs tools:

```bash
git clone <repo-url>
cd ragdoc

# Install all extras and dev/docs dependency groups
uv sync --all-extras --group docs
```

## Optional system dependencies

Some parsers require system tools:

- **Word documents** (`.docx`): requires [Pandoc](https://pandoc.org/installing.html)
  installed and on `PATH`.
- **MinerU output** (`pdf-mineru` extra): ragdoc only reads MinerU's `*_middle.json`; install and
  run MinerU separately (MinerU downloads its own models).

## Verify installation

```python
import ragdoc
from ragdoc import Document, DocumentPipeline, load
print("ragdoc installed successfully")
```
