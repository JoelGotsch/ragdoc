"""LLMChunker: multi-summary, multi-embedding chunker.

Calls an LLM to produce N topic summaries for a document, then returns N
:class:`~ragdoc.chunking.chunk.Chunk` objects that all share the same
``prompt_content`` but have distinct ``embedding_content`` values — one per
topic summary.  This allows a single document to be retrieved from a vector
store via multiple distinct semantic angles.

Typical usage::

    from openai import AsyncOpenAI
    from ragdoc.chunking import LLMChunker

    chunker = LLMChunker(client=AsyncOpenAI())
    chunks = await chunker.chunk(doc)
    # chunks[0].prompt_content == chunks[1].prompt_content == ...
    # chunks[0].embedding_content != chunks[1].embedding_content

Customising the prompt::

    from functools import partial
    from ragdoc.chunking.llm import build_topic_summary_messages

    my_messages = partial(build_topic_summary_messages, system_prompt="Be terse.")
    chunker = LLMChunker(client=client, create_messages=my_messages)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragdoc.chunking.base import Chunker
from ragdoc.chunking.chunk import Chunk
from ragdoc.chunking.provenance import resolve_chunk_provenance
from ragdoc.llm import ChatClient, call_structured, resolve_openai_client

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.document import Document
    from ragdoc.rendering import Renderer
    from ragdoc.utils import Tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class DocumentTopicSummaries(BaseModel):
    """Structured LLM output: a list of per-topic summaries for one document."""

    summaries: list[str] = Field(
        description=(
            "List of 2–5 concise retrieval-optimised summaries, one per distinct topic "
            "or angle covered by the document.  Each summary should stand alone as a "
            "dense-vector search target."
        )
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LLM_CHUNKER_SYSTEM_PROMPT = """
You are an expert at analysing documents and producing concise, retrieval-optimised summaries.

Given the full text of a document, your task is to identify the **2 to 5 most distinct topics or
themes** present and write one short summary (1-100 words) per topic.  Each summary must:

1. Stand alone — it will be embedded independently and matched against user queries without any
   context from the other summaries.
2. Use natural, descriptive language that mirrors how an llm-agent might phrase a search query about
   that topic.
3. Capture the essential facts, entities, and relationships for that topic.
4. Not repeat information that is specific to another topic's summary (no duplication).

If the document covers only one clear topic, return a single summary.
""".strip()

LLM_CHUNKER_USER_MESSAGE = "Please analyse the following document and produce topic summaries as described."

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_topic_summary_messages(
    rendered_text: str,
    *,
    system_prompt: str = LLM_CHUNKER_SYSTEM_PROMPT,
    user_message: str = LLM_CHUNKER_USER_MESSAGE,
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for a topic-summary LLM call.

    Args:
        rendered_text: The document rendered to a string (e.g. via ``Renderer``).
        system_prompt: Override the default system prompt.
        user_message: Override the default user instruction.

    Returns:
        A ``messages`` list suitable for ``client.chat.completions.parse``.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "text", "text": f"*Document to analyse:*\n\n{rendered_text}"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class LLMChunker(Chunker):
    """Produces N :class:`~ragdoc.chunking.chunk.Chunk` objects from one document.

    All chunks share the same ``prompt_content`` (full document rendered for
    LLM context) but have distinct ``embedding_content`` values — one per
    topic summary returned by the LLM.

    Args:
        client: Async OpenAI-compatible client.  Falls back to
            ``get_config().openai_client`` when ``None``; raises
            :class:`~ragdoc.llm.LLMNotConfiguredError` at construction when
            neither is available (fail-loud — never a ``None`` crash at chunk time).
        model: The model to use.  ``None`` reads ``get_config().default_llm_model``
            at init time.
        prompt_renderer: Renderer for ``prompt_content``.
            Defaults to MARKDOWN with ``render_for_prompt``.
        create_messages: Function that builds the ``messages`` list from the
            rendered text.  Signature: ``(rendered_text: str) -> list``.
            Defaults to :func:`build_topic_summary_messages`.
        metadata_fn: Called with the document to build ``Chunk.metadata``.
            Defaults to using the document's metadata (copied per chunk — the
            N chunks never share one metadata dict object).
        request_timeout: Per-request LLM timeout in seconds (``None`` uses the
            client default).
        max_prompt_tokens: When set, the rendered text sent to the **LLM** is
            truncated to this many tokens (one WARNING per over-budget document).
            The emitted chunks keep the full rendered text as ``prompt_content`` —
            truncation applies to the LLM input only.
        tokenizer: Tokenizer used for the ``max_prompt_tokens`` budget.
            ``None`` uses :class:`~ragdoc.utils.GPTTokenizer`.

    Raises:
        LLMNotConfiguredError: No client passed and none configured.

    Note:
        Chunkers leave ``Chunk.id`` at its uuid4 default —
        :meth:`~ragdoc.pipeline.ChunkPipeline.run` mints
        deterministic ids via :func:`~ragdoc.chunking.provenance.mint_chunk_id`.
        Ids are stable across runs only while the model returns the same number
        of summaries for the same content.
    """

    def __init__(
        self,
        client: ChatClient | None = None,
        model: str | None = None,
        prompt_renderer: Renderer | None = None,
        create_messages: Callable[[str], list[ChatCompletionMessageParam]] = build_topic_summary_messages,
        metadata_fn: Callable[[Document], dict] | None = None,
        request_timeout: float | None = None,
        max_prompt_tokens: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        from ragdoc.config import get_config

        config = get_config()
        self._client: ChatClient = resolve_openai_client(client)
        self._model = model if model is not None else config.default_llm_model
        self._prompt_renderer = prompt_renderer
        self._create_messages = create_messages
        self._metadata_fn: Callable[[Document], dict] = metadata_fn or (lambda doc: doc.metadata)
        self._request_timeout = request_timeout
        self._max_prompt_tokens = max_prompt_tokens
        self._tokenizer = tokenizer

    def _get_prompt_renderer(self) -> Renderer:
        if self._prompt_renderer is not None:
            return self._prompt_renderer
        from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

        return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        from ragdoc.utils import GPTTokenizer

        return GPTTokenizer()

    def _budget_llm_input(self, rendered: str, doc_label: str) -> str:
        """Truncate the LLM input to ``max_prompt_tokens`` (chunks keep the full text)."""
        if self._max_prompt_tokens is None:
            return rendered
        tokenizer = self._get_tokenizer()
        n_tokens = tokenizer.count(rendered)
        if n_tokens <= self._max_prompt_tokens:
            return rendered
        logger.warning(
            f"LLMChunker: {doc_label} renders to {n_tokens} tokens, over "
            f"max_prompt_tokens={self._max_prompt_tokens}; truncating the LLM input "
            "(emitted chunks keep the full prompt_content)."
        )
        return tokenizer.truncate(rendered, self._max_prompt_tokens)

    async def chunk(self, document: Document) -> list[Chunk]:
        from pathlib import Path

        doc_label = f"{Path(document.source_path).name} ({document.id})" if document.source_path else document.id
        # Rendering is CPU-bound and may fork a pandoc subprocess (non-HTML formats).
        rendered = await asyncio.to_thread(self._get_prompt_renderer().render, document)
        logger.debug(f"LLMChunker: rendered document: {len(rendered)} chars")
        messages = self._create_messages(self._budget_llm_input(rendered, doc_label))

        # A refusal raises LLMRefusalError — a chunker with no output is an error, not a degrade.
        topic_summaries = await call_structured(
            self._client,
            model=self._model,
            messages=messages,
            response_format=DocumentTopicSummaries,
            temperature=0.0,
            timeout=self._request_timeout,
            log_prefix="LLMChunker",
        )
        logger.info(
            f"LLMChunker: generating topic summaries (model={self._model})"
            f" -> {len(topic_summaries.summaries)} summaries"
        )

        metadata = self._metadata_fn(document)
        prov = resolve_chunk_provenance(document)
        chunks = [
            Chunk(
                source_path=document.source_path or None,
                source_id=prov.source_id,
                source_hash=prov.source_hash,
                content_hash=prov.content_hash,
                prompt_content=rendered,
                embedding_content=summary,
                metadata=dict(metadata),  # type: ignore[reportArgumentType]  # fresh dict per chunk
            )
            for summary in topic_summaries.summaries
        ]
        logger.info(f"LLMChunker: {doc_label} -> {len(topic_summaries.summaries)} embedding contents")
        logger.debug(f"LLMChunker: {doc_label} topics: {[s[:80] for s in topic_summaries.summaries]}")
        return chunks
