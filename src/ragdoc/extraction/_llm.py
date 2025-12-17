"""Shared LLM plumbing for the extraction stage (private module).

The two extractors (:class:`~ragdoc.extraction.structured.StructuredExtractor`,
:class:`~ragdoc.extraction.kg.KnowledgeGraphExtractor`) used to duplicate ~100 lines of
client/model/renderer/tokenizer fallback resolution plus the structured-output retry loop.
Those live here once, as standalone functions (house convention: reusable logic = standalone
functions, not private methods).

Client resolution and the retry engine live in :mod:`ragdoc.llm`
(:func:`~ragdoc.llm.resolve_openai_client` / :func:`~ragdoc.llm.call_structured`);
:func:`parse_with_retry` is a thin extraction-flavoured adapter that maps
``ExtractionSettings`` knobs onto ``call_structured``. Do not inline retry logic anywhere else.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from ragdoc.llm import ChatClient, call_structured

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.extraction.structured import ExtractionSettings
    from ragdoc.rendering import Renderer
    from ragdoc.utils import Tokenizer

logger = logging.getLogger(__name__)

BatchT = TypeVar("BatchT", bound=BaseModel)


def resolve_model(explicit: str | None, settings: ExtractionSettings) -> str:
    """Resolve the model name: explicit → ``settings.model_name`` → ``get_config().default_llm_model``."""
    if explicit is not None:
        return explicit
    if settings.model_name:
        return settings.model_name
    from ragdoc.config import get_config

    return get_config().default_llm_model


def resolve_renderer(explicit: Renderer | None) -> Renderer:
    """Resolve the document renderer: explicit → ``Renderer(MARKDOWN, render_for_prompt)``."""
    if explicit is not None:
        return explicit
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


def resolve_tokenizer(explicit: Tokenizer | None) -> Tokenizer:
    """Resolve the tokenizer for the ``min_tokens`` gate: explicit → ``GPTTokenizer()``."""
    if explicit is not None:
        return explicit
    from ragdoc.utils import GPTTokenizer

    return GPTTokenizer()


def build_messages(text: str, *, system_prompt: str, user_message: str) -> list[ChatCompletionMessageParam]:
    """Build the two-message ``messages`` list shared by every extraction LLM call."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_message}\n\n{text}"},
    ]


async def parse_with_retry(
    client: ChatClient,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_format: type[BatchT],
    settings: ExtractionSettings,
    log_prefix: str,
) -> BatchT:
    """One structured ``.parse()`` call with the library retry policy, from extraction settings.

    Maps ``settings.temperature`` / ``settings.request_timeout`` / ``settings.max_retries`` onto
    :func:`ragdoc.llm.call_structured`: transport errors (429/connection/timeout/5xx) are retried
    with backoff, deterministic errors (other 4xx) are not, and a ``parsed`` of ``None`` raises
    :class:`~ragdoc.llm.LLMRefusalError` without retrying. When exhausted, the last underlying
    exception re-raises unchanged (the KG halving wrapper catches it above).
    """
    return await call_structured(
        client,
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        log_prefix=log_prefix,
    )
