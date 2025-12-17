"""
Element renderers for different rendering targets.

This module provides singledispatch-based element renderers for:
- render_for_prompt: Optimized for LLM prompt context
- render_raw: Full fidelity rendering (all content, incl images)

All renderers output HTML. Format conversion (to markdown, plain text, etc.)
is handled by the Renderer class, not by element renderers.

Architecture:
-------------
Each renderer uses singledispatch with add_unregister for extensibility.
The `inline` parameter controls whether elements are rendered for inline
substitution (compact format) or standalone display (full format).

To override rendering of an existing element type (e.g. change how Footnotes render):

    from ragdoc.document import Footnote
    from ragdoc.rendering.elements import render_for_prompt
    from ragdoc.rendering.base import RenderContext

    # Unregister the built-in handler, then register your own.
    render_for_prompt.unregister(Footnote)

    @render_for_prompt.register(Footnote)
    def _render_footnote_custom(element: Footnote, ctx: RenderContext, inline: bool = False) -> str:
        # Always expand inline — no "[N]" placeholders, full text instead.
        return f"<span class='fn'>[Note: {element.text}]</span>"

Inline vs Standalone Rendering:
-------------------------------
- inline=False (default): Element rendered as a standalone block
- inline=True: Element rendered for substitution at <ref> location
  - Footnotes: Compact "[Footnote: text]" instead of full block
  - Images: Placeholder or brief description instead of full rendering
  - Tables: May use summary instead of full table

Example with inline refs:
    ```python
    @render_for_prompt.register(Footnote)
    def render_footnote_prompt(element: Footnote, ctx: RenderContext, inline: bool = False) -> str:
        if inline:
            return f"<span>[{element.number}: {element.text}]</span>"
        return element.html
    ```
"""

from __future__ import annotations

import json
import re
from functools import singledispatch
from html import escape
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from ragdoc.utils.helpers import add_unregister

if TYPE_CHECKING:
    from ragdoc.rendering.base import RenderContext

from ragdoc.document import (
    BaseElement,
    Footnote,
    Heading,
    Image,
)
from ragdoc.metadata import MetadataValue, serialize_metadata_value as _canonical_serialize
from ragdoc.rendering.base import DocumentMetadata

# =============================================================================
# Prompt Content Renderer
# =============================================================================


@add_unregister
@singledispatch
def render_for_prompt(element: BaseElement, ctx: RenderContext, inline: bool = False) -> str:
    """
    Render element for LLM prompt context.

    This renderer prioritizes:
    - Full content (NOT summary) for complete context
    - Text representations over binary content
    - Human-readable HTML for conversion to markdown

    Note: This renderer does NOT use summaries.

    Use this for generating content to include in LLM prompts.
    The Renderer will convert HTML to markdown or other formats.

    Args:
        element: The element to render
        ctx: Render context with document access
        inline: If True, render for inline substitution (compact format)

    Returns:
        HTML string (full content, not summary)
    """
    if inline:
        return f"<span>{element.html}</span>"
    if not ctx.is_inline_referenced(element.id):
        return f"{element.html}"
    return ""  # only render once


@render_for_prompt.register(Heading)
def _render_heading_prompt(
    element: Heading,
    ctx: RenderContext,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
    inline: bool = False,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
) -> str:
    """Prompt heading renderer - strips inline CSS (e.g. text-align, font-size from MinerU).

    CSS visual properties are noise for LLMs. The heading tag and level are preserved.
    """
    soup = BeautifulSoup(element.html, "html.parser")
    tag = soup.find(re.compile(r"^h[1-6]$"))
    if tag and tag.get("style"):
        del tag["style"]
    return str(soup)


@render_for_prompt.register(Image)
def _render_image_prompt(element: Image, ctx: RenderContext, inline: bool = False) -> str:
    """
    Prompt image renderer - prioritizes text representation.

    Order of preference:
    1. text_representation (structured content or natural-language description)
    2. alt text
    3. Generic "[Unprocessed Image]" marker

    When inline=True, uses a more compact span format.
    """
    if ctx.is_inline_referenced(element.id) and not inline:
        return ""  # only render once at the inline ref location
    prompt_text = element.text_representation or element.alt
    if prompt_text:
        if inline:
            return f"<span>[Image: {prompt_text}]</span>"
        return f"<p><em>[Image: {prompt_text}]</em></p>"
    if inline:
        return "<span>[Unprocessed Image]</span>"
    return "<p><em>[Unprocessed Image]</em></p>"


@render_for_prompt.register(Footnote)
def _render_footnote_prompt(element: Footnote, ctx: RenderContext, inline: bool = False) -> str:
    """Prompt footnote renderer - compact when inline, full html for referenced standalone.

    Orphaned footnotes (no inline ref pointing to them) are wrapped in a
    blockquote so the LLM can distinguish them from properly-anchored notes.
    """
    if inline:
        return f"<span>[Footnote: {element.text}]</span>"
    if not ctx.is_inline_referenced(element.id):
        return element.html
    return ""  # only render once


# =============================================================================
# Raw/Full Fidelity Renderer
# =============================================================================


@add_unregister
@singledispatch
def render_raw(
    element: BaseElement,
    ctx: RenderContext,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
    inline: bool = False,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
) -> str:
    """
    Render element with full fidelity.

    This renderer includes:
    - Full HTML content
    - Embedded images (base64)
    - All formatting

    Use this for HTML export or full document rendering.

    Args:
        element: The element to render
        ctx: Render context with document access
        inline: If True, render for inline substitution

    Returns:
        Full HTML string with all content
    """
    return element.html


@render_raw.register(Footnote)
def _render_footnote_raw(
    element: Footnote,
    ctx: RenderContext,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
    inline: bool = False,
) -> str:
    """Raw footnote renderer - compact when inline."""
    if inline:
        return f'<a href="#footnote-{element.id}">[{element.number}]</a>'  # f"<span class=\"footnote-inline\">[{element.number}: {element.innerhtml}]</span>"
    return element.html


# =============================================================================
# DocumentMetadata Renderer
# =============================================================================


def format_metadata_value(value: MetadataValue) -> str:
    """Serialize a single metadata value to a plain string for HTML/YAML output.

    Delegates to ``ragdoc.metadata.serialize_metadata_value`` for the canonical
    JSON-primitive conversion, then formats the result as a string for display:

      str            → returned as-is
      int/float/bool → str()
      None           → "" (caller should skip empty results)
      dict/list      → json.dumps() (BaseModel instances inside are handled by canonical serializer)
      BaseModel      → json.dumps() of model_dump(mode='json')

    Limitations:
      - dict/list/BaseModel values always become a flat JSON string — they do NOT
        produce structured YAML keys or nested HTML.
    """
    result = _canonical_serialize(value)
    if result is None:
        return ""
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


def _metadata_to_html(data: DocumentMetadata) -> str:
    """Render a metadata dict as a semantic HTML5 <header> block.

    The title key (if present) becomes an <h1>; remaining keys become a
    <dl> of term/definition pairs.  The enclosing <header> carries the
    class "document-metadata" so _convert() can locate it when building
    a full HTML document for pandoc's --standalone mode.

    Values are formatted via format_metadata_value() and HTML-escaped.
    None values are skipped.
    """
    parts: list[str] = []
    if (raw_title := data.get("title")) is not None:
        parts.append(f"<h1>{escape(format_metadata_value(raw_title))}</h1>")
    other = {k: v for k, v in data.items() if k != "title" and v is not None}
    if other:
        dt_dd = "".join(f"<dt>{escape(k)}</dt><dd>{escape(format_metadata_value(v))}</dd>" for k, v in other.items())
        parts.append(f"<dl>{dt_dd}</dl>")
    return f'<header class="document-metadata">{"".join(parts)}</header>' if parts else ""


@render_for_prompt.register(DocumentMetadata)
@render_raw.register(DocumentMetadata)
def _render_metadata_block(
    element: DocumentMetadata,
    ctx: RenderContext,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
    inline: bool = False,  # pyright: ignore[reportUnusedParameter] # singledispatch interface
) -> str:
    """Render document metadata as a semantic HTML header block."""
    return _metadata_to_html(element)
