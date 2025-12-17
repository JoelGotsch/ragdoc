"""
Rendering module for ragdoc.

This module provides document rendering functionality with pluggable element
renderers and format conversion support.

Architecture:
- Renderer: Main class that renders Document to string
- RenderContext: Context passed to element renderers with document and resolver
- ExternalRefProvider: Protocol for resolving cross-document references
- Element renderers: singledispatch functions for different render targets

Example:
    ```python
    from ragdoc.rendering import Renderer, render_for_prompt, OutputFormat

    renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        metadata_keys=["source", "date"],
    )

    content = renderer.render(document)
    ```
"""

from ragdoc.rendering.base import (
    DocumentMetadata,
    ExternalRefProvider,
    MetadataValue,
    OutputFormat,
    RenderContext,
    Renderer,
)
from ragdoc.rendering.defaults import resolve_renderer
from ragdoc.rendering.elements import (
    format_metadata_value,
    render_for_prompt,
    render_raw,
)

# TODO: Implement HierarchicalRenderer using the new architecture.
#
# The old render.py had HierarchicalRenderer(StandardRenderer) which, given a document
# and an ExternalRefProvider, rendered it together with context from its parents,
# siblings, and children — contributing headings + summaries up to a token budget.
# Priority order (trimmed first when over budget): children > siblings > parents > content.
# Position in output was based on document reading order (sort_key on headings).
#
# In the new architecture this should be a standalone Renderer subclass or function that
# uses RenderContext.get_external() and RenderContext.get_siblings() to gather related
# documents, then assembles the output with the same priority/position logic.
#
# Also missing from the new architecture vs old render.py:
# - StandardRenderer.render_fragment(document) -> Chunk
#   (should become a helper combining Renderer + chunking.Chunk)
# - render_documents(documents, format, target) -> str
#   (trivial helper: "\n\n".join(renderer.render(d) for d in documents))

__all__ = [
    "DocumentMetadata",
    "ExternalRefProvider",
    "MetadataValue",
    "OutputFormat",
    "RenderContext",
    "Renderer",
    "format_metadata_value",
    "render_for_prompt",
    "render_raw",
    "resolve_renderer",
]
