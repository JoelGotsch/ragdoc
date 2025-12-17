"""Default prompt-rendering resolution shared across the library.

Every LLM-adjacent stage (chunkers, summarizers, extractors, the mention pipeline's
default splitter) renders documents with the same default strategy: full-fidelity
structured text (``render_for_prompt``) as Markdown. That default is defined once
here — call :func:`resolve_renderer` instead of constructing the fallback inline,
so a change to the library-wide default rendering strategy happens in one place.
"""

from __future__ import annotations

from ragdoc.rendering.base import OutputFormat, Renderer
from ragdoc.rendering.elements import render_for_prompt


def resolve_renderer(explicit: Renderer | None) -> Renderer:
    """Resolve a document renderer: explicit → ``Renderer(MARKDOWN, render_for_prompt)``."""
    if explicit is not None:
        return explicit
    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
