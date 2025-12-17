# Installation

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install with uv

```bash
# Core library
uv add ragdoc

# With Azure Document Intelligence support
uv add "ragdoc[azure-di]"

# With MinerU PDF parsing support
uv add "ragdoc[pdf-mineru]"

# With Qdrant vector store support
uv add "ragdoc[qdrant]"

# All extras
uv add "ragdoc[azure-di,pdf-mineru,qdrant]"
```

## Install for development

Clone the repository and install all dependencies including docs tools:

```bash
git clone <repo-url>
cd aa-document-processing

# Install all extras and dev/docs dependency groups
uv sync --all-extras --group docs
```

## Optional system dependencies

Some parsers require system tools:

- **Word documents** (`.docx`, `.doc`): requires [Pandoc](https://pandoc.org/installing.html)
  installed and on `PATH`.
- **PDF via MinerU** (`pdf-mineru` extra): requires additional model downloads on first use.

## Verify installation

```python
import ragdoc
from ragdoc import Document, DocumentPipeline, load
print("ragdoc installed successfully")
```
