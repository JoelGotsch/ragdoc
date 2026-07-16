# Rendering

> Run interactively: `marimo edit docs/notebooks/rendering.py`

Rendering converts a [`Document`](document-model.md) into a formatted string. ragdoc provides
two element renderers and multiple output formats.

## The two element renderers

| Renderer function | Intent |
|-------------------|--------|
| [`render_for_prompt`](../api/rendering.md#render_for_prompt) | Full-fidelity text for LLM context windows and BM25 search |
| [`render_raw`](../api/rendering.md#render_raw) | Full fidelity, no transformation — for debugging |

`render_for_prompt` is the primary renderer used throughout the pipeline. For `Image`
elements, it uses `text_representation` as a text fallback when the image cannot be
displayed directly.

## Using the Renderer

[`Renderer`](../api/rendering.md#renderer) combines an element renderer with a target
output format:

```python
from ragdoc.rendering import Renderer, render_for_prompt, OutputFormat

# For LLM context windows — full content as Markdown
prompt_renderer = Renderer(
    format=OutputFormat.MARKDOWN,
    element_renderer=render_for_prompt,
)

prompt_text = prompt_renderer.render(document)
```

## Output formats

[`OutputFormat`](../api/rendering.md#outputformat) controls the final text format.
Element renderers always output HTML internally; the `Renderer` handles conversion:

| Format | Description |
|--------|-------------|
| `OutputFormat.HTML` | Raw HTML output (no conversion) |
| `OutputFormat.MARKDOWN` | Standard Markdown |
| `OutputFormat.GFM` | GitHub Flavored Markdown (tables rendered) |
| `OutputFormat.RST` | reStructuredText |
| `OutputFormat.PLAIN` | Plain text (all markup stripped) |

Format conversion is done via [pypandoc](https://github.com/NicklasTegner/pypandoc).

## Metadata in output

Use `metadata_keys` to include selected metadata fields as a header in the rendered output:

```python
renderer = Renderer(
    format=OutputFormat.MARKDOWN,
    element_renderer=render_for_prompt,
    metadata_keys=["source", "date", "author"],
    include_title=True,
)
text = renderer.render(document)
# Output starts with: # Document Title\nsource: ...\ndate: ...\n\n...content...
```

## Inline reference resolution

The `Renderer` resolves `<ref id="..." rel="..."/>` tags in element HTML. When a paragraph
contains a reference to an image, the renderer replaces the `<ref>` tag with the rendered
content of the image element.

Elements that are only referenced inline (and never appear as standalone block elements)
are skipped during the main rendering pass to avoid duplication.

## Cross-document references

Pass an [`ExternalRefProvider`](../api/rendering.md#externalrefprovider) to access parent
documents and siblings during rendering. A plain `dict[str, Document]` works:

```python
all_docs = {doc.id: doc for doc in all_documents}

renderer = Renderer(
    format=OutputFormat.MARKDOWN,
    element_renderer=render_for_prompt,
    external_refs=all_docs,
)
```

## Extending with custom element renderers

Element renderers are `singledispatch` functions. You can register custom renderers for
specific element types:

```python
from ragdoc.rendering.elements import render_for_prompt
from ragdoc.document import Image

@render_for_prompt.register(Image)
def render_image_custom(element, ctx, inline=False):
    alt = element.alt or ""
    return f"<figure><img alt='{alt}'/></figure>"
```

## See Also

- [API Reference: Rendering](../api/rendering.md)
- [Chunking Guide](chunking.md) — how renderers are used to produce `Chunk`
- [Processing API](../api/processing.md) — LLM processors that enrich documents
