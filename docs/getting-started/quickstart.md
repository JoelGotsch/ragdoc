# Quickstart

> Run interactively: `marimo edit docs/notebooks/quickstart.py`

This guide walks through a complete parsing → splitting → chunking pipeline.

## Minimal example

```python
from pathlib import Path
from ragdoc.pipeline import DocumentPipeline, TokenSplitter

pipeline = DocumentPipeline(splitter=TokenSplitter(max_tokens=4000))
chunks = await pipeline.run(Path("report.docx"))

print(f"Produced {len(chunks)} chunks")
print(chunks[0].prompt_content[:500])
```

`DocumentPipeline` wires together parse → process → split → chunk in one call: the parser
is inferred from the file extension, `TokenSplitter` splits at heading boundaries within a
token budget, and the default `SimpleChunker` renders each split into a `Chunk`
(with `embedding_content = prompt_content`).

> **Tip:** For concurrent multi-file processing, error handling, and incremental
> vector-store sync, see the [Pipeline guide](../guide/pipeline.md).

## With LLM enrichment

Add processing steps between parsing and splitting to enrich the document:

```python
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from ragdoc.chunking import Chunk
from ragdoc.pipeline import DocumentPipeline, TokenSplitter
from ragdoc.processing import (
    HeadingLevelProcessor,
    ImageSummaryProcessor,
    TitleDetectionProcessor,
    openai_image_summarizer,
)


async def build_chunks(file_path: str) -> list[Chunk]:
    client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment

    pipeline = DocumentPipeline(
        processors=[
            HeadingLevelProcessor(),
            TitleDetectionProcessor(),
            ImageSummaryProcessor(summarize=openai_image_summarizer(client, model="gpt-4o")),
        ],
        splitter=TokenSplitter(max_tokens=4000),
    )
    return await pipeline.run(Path(file_path))


chunks = asyncio.run(build_chunks("report.docx"))
```

## Supported file formats

| File name ends with | Parser | Notes |
|-----------|--------|-------|
| `.html`, `.htm` | `html` | |
| `.docx` | `pandoc` | Converted via pandoc (legacy `.doc` is not readable — convert it first) |
| `.xlsx` | `xlsx` | Needs the `xlsx` extra |
| `.pdf` | `azure_di` or `pdf_basic` | Azure Document Intelligence if the `azure-di` extra and credentials are available; otherwise the local `pdf_basic` parser (`pdf` extra) |
| `.azure.json` | `azure_json` | A saved Azure Document Intelligence result |
| `_middle.json` | `mineru` | MinerU layout output (`pdf-mineru` extra) |
| `.ragdoc.json` | `ragdoc_json` | A `Document` previously dumped by ragdoc |

Use [`load()`](../api/parsing.md#load) to automatically select the right parser, or pass
`parser=` to force a specific one.

## Next steps

- Read the [Guide](../guide/overview.md) for a deep dive into each pipeline stage
- See [Processing API](../api/processing.md) for all available processors
- See [Rendering API](../api/rendering.md) for output format options
