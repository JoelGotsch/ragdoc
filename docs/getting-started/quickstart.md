# Quickstart

> Run interactively: `marimo edit docs/notebooks/quickstart.py`

This guide walks through a complete parsing → splitting → chunking pipeline.

## Minimal example

```python
from ragdoc.parsing import load
from ragdoc.rendering import Renderer, render_for_prompt, OutputFormat
from ragdoc.splitting import split_by_headings
from ragdoc.chunking import Chunk

# 1. Parse — infers parser from extension
document = await load("report.docx")

# 2. Split into sections at heading boundaries
sections = split_by_headings(document)

# 3. Render and chunk
renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)

chunks = [
    Chunk(
        prompt_content=renderer.render(doc),
        embedding_content=renderer.render(doc),
        filename=doc.filename,
        source_path=doc.source_path,
        metadata=doc.metadata,
    )
    for doc in sections
]

print(f"Produced {len(chunks)} chunks")
print(chunks[0].prompt_content[:500])
```

> **Tip:** For the full pipeline in one call — including concurrent processing, error
> handling, and incremental vector-store sync — use
> [`DocumentPipeline`](../guide/pipeline.md).

## With LLM enrichment

Add processing steps between parsing and splitting to enrich the document:

```python
from ragdoc.parsing import load
from ragdoc.rendering import Renderer, render_for_prompt, OutputFormat
from ragdoc.config import configure, RagdocConfig
from ragdoc.processing import (
    ProcessingPipeline,
    HeadingLevelProcessor,
    TitleDetectionProcessor,
    ImageSummaryProcessor,
)
from ragdoc.splitting import split_by_headings
from ragdoc.chunking import Chunk
from openai import AsyncOpenAI
import asyncio


async def build_chunks(file_path: str) -> list[Chunk]:
    config = RagdocConfig(
        openai_client=AsyncOpenAI(),
        default_llm_model="gpt-4o-mini",
        default_image_llm_model="gpt-4o",
    )

    with configure(config):
        # 1. Parse
        document = await load(file_path)

        # 2. Process: refine headings, detect title, summarize images
        pipeline = ProcessingPipeline()
        pipeline.add(HeadingLevelProcessor())
        pipeline.add(TitleDetectionProcessor())
        pipeline.add(ImageSummaryProcessor())

        document = await pipeline.process(document)

        # 3. Split
        sections = split_by_headings(document)

        # 4. Chunk — SimpleChunker sets embedding_content = prompt_content
        renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)

        chunks = [
            Chunk(
                prompt_content=renderer.render(doc),
                embedding_content=renderer.render(doc),
                metadata=doc.metadata,
            )
            for doc in sections
        ]

    return chunks


chunks = asyncio.run(build_chunks("report.docx"))
```

## Supported file formats

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.docx`, `.doc` | Pandoc | Requires Pandoc installed |
| `.html` | HTML parser | |
| `.xlsx` | Excel parser | |
| `.azure.json` | Azure DI | Azure Document Intelligence output |
| `.pdf` | Azure DI | Dispatches to Azure Document Intelligence |
| `_middle.json` | MinerU | MinerU layout output |

Use [`load()`](../api/parsing.md#load) to automatically select the right parser, or pass
`parser=` to force a specific one.

## Next steps

- Read the [Guide](../guide/overview.md) for a deep dive into each pipeline stage
- See [Processing API](../api/processing.md) for all available processors
- See [Rendering API](../api/rendering.md) for output format options
