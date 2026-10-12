# Rendering

> Run interactively: `marimo edit docs/notebooks/rendering.py`

Rendering converts a [`Document`](document-model.md) into a formatted string. ragdoc provides
two element renderers and multiple output formats.

## The two element renderers

| Renderer function | Intent |
|-------------------|--------|
| [`render_for_prompt`](../api/rendering.md#render_for_prompt) | Full-fidelity text for LLM context windows and BM25 search |
| [`render_raw`](../api/rendering.md#render_raw) | Full fidelity, no transformation — for debugging |

`render_for_prompt` is the primary renderer used throughout the pipeline. It never emits
image data: an `Image` renders as its `text_representation`, else its `alt` text, else
`[Unprocessed Image]`.

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

MARKDOWN, GFM and RST are converted with [pypandoc](https://github.com/NicklasTegner/pypandoc);
HTML is returned as is; PLAIN is the HTML's text content (BeautifulSoup `get_text()`).

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
# Markdown output starts with a YAML front-matter block (keys sorted, title as a key):
# ---\nauthor: ...\ndate: ...\nsource: ...\ntitle: Document Title\n---\n...content...
```

## Inline reference resolution

The `Renderer` resolves `<ref id="..." rel="..."/>` tags in element HTML. When a paragraph
contains a reference to an image, the renderer replaces the `<ref>` tag with the rendered
content of the image element.

With `render_for_prompt`, an element that is referenced inline is rendered only at its
`<ref>` location, not a second time as a standalone block. `render_raw` renders it in both
places.

## Cross-document references

Pass an [`ExternalRefProvider`](../api/rendering.md#externalrefprovider) to make other
documents available to **custom** element renderers (via the render context). The built-in
renderers do not use it, so it does not change their output. A plain `dict[str, Document]`
works:

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
