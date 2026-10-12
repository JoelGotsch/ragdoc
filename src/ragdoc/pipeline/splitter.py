"""TokenSplitter: a Splitter-Protocol-compatible wrapper around split_document().

:class:`TokenSplitter` is the recommended default splitter for
:class:`~ragdoc.pipeline.linear.DocumentPipeline`.  It wraps
:func:`~ragdoc.splitting.split_document` and exposes the most common knobs
(max tokens, overlap, renderer, tokenizer) as constructor arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragdoc.splitting.token import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS

if TYPE_CHECKING:
    from ragdoc.document import Document
    from ragdoc.rendering import Renderer
    from ragdoc.utils import Tokenizer


class TokenSplitter:
    """Wraps :func:`~ragdoc.splitting.split_document` as a :class:`~ragdoc.splitting.Splitter`-compatible callable.

    ``max_tokens`` controls retrieval chunk size, not LLM context-window fit.
    Chunks should typically be 1/50th to 1/10th of the context window — the
    right value depends on your retrieval quality requirements and should be
    tested empirically.

    Args:
        max_tokens: Token budget per split (default 7000).
        overlap_tokens: Overlap between consecutive splits (default 200).
        renderer: Renderer used for token measurement.  Defaults to
            MARKDOWN + ``render_for_prompt`` (same renderer as prompt content).
        tokenizer: Custom tokenizer.  Defaults to tiktoken ``cl100k_base``.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        renderer: Renderer | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = tokenizer
        if not renderer:
            from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

            renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
        self._renderer = renderer

    def __call__(self, document: Document) -> list[Document]:
        from ragdoc.splitting import split_document

        return split_document(
            document,
            self._renderer,
            self._tokenizer,
            self.max_tokens,
            self.overlap_tokens,
        )
