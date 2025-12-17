"""Shared LLM plumbing for the extraction stage (private module).

The two extractors (:class:`~ragdoc.extraction.structured.StructuredExtractor`,
:class:`~ragdoc.extraction.kg.KnowledgeGraphExtractor`) used to duplicate ~100 lines of
client/model/renderer/tokenizer fallback resolution plus the structured-output retry loop.
Those live here once, as standalone functions (house convention: reusable logic = standalone
functions, not private methods).

Phase-7 note: :func:`parse_with_retry` is deliberately the *only* function whose body the shared
``ragdoc.llm`` reliability layer will replace (a ``call_structured`` call with real backoff and
error classification); its signature is final so no call site changes again. Do not inline retry
logic anywhere else.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.extraction.structured import ExtractionSettings
    from ragdoc.rendering import Renderer
    from ragdoc.utils import Tokenizer

logger = logging.getLogger(__name__)

BatchT = TypeVar("BatchT", bound=BaseModel)


def resolve_client(explicit: Any | None) -> Any:
    """Resolve the LLM client: explicit → ``get_config().openai_client`` → ``ValueError``.

    ``Any`` typing is deliberate until the Phase-7 ``ChatClient`` protocol exists.
    """
    if explicit is not None:
        return explicit
    from ragdoc.config import get_config

    client = get_config().openai_client
    if client is None:
        raise ValueError("No client provided and get_config().openai_client is None.")
    return client


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
    client: Any,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_format: type[BatchT],
    settings: ExtractionSettings,
    log_prefix: str,
) -> BatchT:
    """One structured ``.parse()`` call with the extraction retry policy.

    ``settings.max_retries`` additional attempts; ``temperature`` and ``request_timeout`` come
    from *settings* (``timeout`` is forwarded per request only when set — ``None`` defers to the
    client default). A ``parsed`` of ``None`` raises ``ValueError`` (fed back into the retry
    loop). Each failure logs an ERROR with attempt counters; when exhausted the last exception
    re-raises unchanged.
    """
    parse_kwargs: dict[str, Any] = {}
    if settings.request_timeout is not None:
        parse_kwargs["timeout"] = settings.request_timeout
    for attempt in range(settings.max_retries + 1):
        try:
            response = await client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=settings.temperature,
                response_format=response_format,
                **parse_kwargs,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("LLM returned no parsed extraction")
            return parsed
        except Exception as exc:
            logger.error(
                "%s: API call failed (attempt %d/%d): %s",
                log_prefix,
                attempt + 1,
                settings.max_retries + 1,
                exc,
            )
            if attempt == settings.max_retries:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover
