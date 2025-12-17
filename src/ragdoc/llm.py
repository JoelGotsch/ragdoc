"""Shared LLM reliability layer: client protocols, resolution, retry policy, structured calls.

Every LLM call in the library goes through this module:

- :func:`call_structured` — one structured-output call on the **non-beta**
  ``client.chat.completions.parse`` namespace, with the library-wide retry policy.
- :func:`retry_llm` — the generic retry engine (also used by
  :class:`~ragdoc.pipeline.embedders.OpenAIEmbedder` for ``embeddings.create``).
- :func:`resolve_openai_client` — the single client fallback policy: explicit →
  ``get_config().openai_client`` → :class:`LLMNotConfiguredError`. Fail-loud at construction.
- :class:`ChatClient` / :class:`EmbeddingsClient` — narrow structural protocols, so call sites
  demand only what they use (an embeddings-only gateway never has to fake a ``chat`` attribute).
  :class:`LLMClient` is their intersection — what :class:`~ragdoc.config.RagdocConfig` holds,
  since one configured ``AsyncOpenAI`` serves both.

Retry policy (transport errors only — each site keeps its own failure *product*, e.g. the heading
resolver degrades to ``[]`` while the chunker raises):

- **Retried** (full-jitter exponential backoff; a parseable ``Retry-After`` header is honored):
  ``openai.RateLimitError`` (429), ``openai.APIConnectionError`` (network — includes
  ``openai.APITimeoutError``), and any ``openai.APIStatusError`` with ``status_code >= 500``.
- **Not retried** (raised immediately): other 4xx (400/401/403/404/422 — deterministic request
  errors), :class:`LLMRefusalError` (a refusal is deterministic for a given input — retrying
  burns tokens for the same answer), ``pydantic.ValidationError``, and every non-openai error.

This module never imports ``openai`` at module level — openai's exception classes are loaded
lazily via :func:`_openai_errors` — so ``import ragdoc.llm`` works without the LLM dependency
installed (Phase 8 moves ``openai`` behind an extra).
"""

from __future__ import annotations

import logging
import random
from asyncio import sleep as _sleep
from collections.abc import Awaitable, Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    import httpx
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
ResponseT_co = TypeVar("ResponseT_co", bound=BaseModel, covariant=True)
CallT = TypeVar("CallT")


# ---------------------------------------------------------------------------
# Structural client protocols
# ---------------------------------------------------------------------------


class ParsedMessage(Protocol[ResponseT_co]):
    """The message of one parsed chat-completion choice (structural, read-only)."""

    @property
    def parsed(self) -> ResponseT_co | None: ...


class ParsedChoice(Protocol[ResponseT_co]):
    """One choice of a parsed chat-completion response (structural, read-only)."""

    @property
    def message(self) -> ParsedMessage[ResponseT_co]: ...


class ParsedResponse(Protocol[ResponseT_co]):
    """A parsed chat-completion response (structural, read-only)."""

    @property
    def choices(self) -> Sequence[ParsedChoice[ResponseT_co]]: ...


class ChatCompletionsEndpoint(Protocol):
    """The completions sub-client exposing structured-output ``parse`` (non-beta namespace)."""

    async def parse(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        response_format: type[ResponseT],
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ParsedResponse[ResponseT]: ...


class ChatEndpoint(Protocol):
    """The ``chat`` sub-client of an OpenAI-compatible client (structural)."""

    @property
    def completions(self) -> ChatCompletionsEndpoint: ...


@runtime_checkable
class ChatClient(Protocol):
    """Minimal structural view of an ``AsyncOpenAI``-compatible client: ``chat.completions.parse``."""

    @property
    def chat(self) -> ChatEndpoint: ...


class EmbeddingItem(Protocol):
    """One embedding entry from an OpenAI-compatible response (structural)."""

    @property
    def index(self) -> int: ...
    @property
    def embedding(self) -> list[float]: ...


class EmbeddingResponse(Protocol):
    """An OpenAI-compatible embeddings response (structural)."""

    @property
    def data(self) -> Sequence[EmbeddingItem]: ...


class EmbeddingsEndpoint(Protocol):
    """The ``embeddings`` sub-client of an OpenAI-compatible client (structural)."""

    async def create(
        self, *, model: str, input: list[str], dimensions: int | None = None, timeout: float | None = None
    ) -> EmbeddingResponse: ...


@runtime_checkable
class EmbeddingsClient(Protocol):
    """Minimal structural view of an ``AsyncOpenAI``-compatible client: ``embeddings.create``."""

    @property
    def embeddings(self) -> EmbeddingsEndpoint: ...


@runtime_checkable
class LLMClient(ChatClient, EmbeddingsClient, Protocol):
    """A client usable for both structured chat and embeddings (what ``RagdocConfig`` holds).

    ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` satisfy this structurally.
    """


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base for ragdoc's LLM-layer errors."""


class LLMRefusalError(LLMError):
    """The model returned no parsed payload (refusal / empty parse). Not retryable.

    A refusal is deterministic for a given input — retrying burns tokens for the same answer.
    Callers that want to degrade catch this specifically.
    """


class LLMNotConfiguredError(LLMError):
    """No client passed and ``get_config().openai_client`` is ``None``. Raised at construction time."""


# ---------------------------------------------------------------------------
# Client resolution
# ---------------------------------------------------------------------------


def resolve_openai_client(explicit: ChatClient | None = None) -> ChatClient:
    """Resolve the LLM client with the one library-wide fallback policy.

    Policy: *explicit* → ``get_config().openai_client`` → :class:`LLMNotConfiguredError`.
    Fail-loud and construction-time: call this in ``__init__``, never lazily at request time,
    so a missing client surfaces where the object is built rather than mid-pipeline.

    Args:
        explicit: An explicitly passed client, or ``None`` to fall back to the active config.

    Returns:
        The resolved client.

    Raises:
        LLMNotConfiguredError: When no explicit client is given and the active config has none.
    """
    if explicit is not None:
        return explicit
    from ragdoc.config import get_config

    client = get_config().openai_client
    if client is None:
        raise LLMNotConfiguredError(
            "No LLM client available. Pass client=... explicitly, or configure one globally: "
            "configure(RagdocConfig(openai_client=AsyncOpenAI(...)))."
        )
    return client


# ---------------------------------------------------------------------------
# Retry engine
# ---------------------------------------------------------------------------


class _OpenAIErrorTypes(NamedTuple):
    """The openai exception classes the retry classifier consults (lazily imported)."""

    rate_limit: type[Exception]
    connection: type[Exception]
    status: type[Exception]


@cache
def _openai_errors() -> _OpenAIErrorTypes:
    """Lazy, cached import of openai's exception types (keeps ``ragdoc.llm`` importable without openai)."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - openai is currently a hard dependency
        raise ImportError('LLM features require the openai package. Install it with: uv add "ragdoc[llm]"') from exc
    return _OpenAIErrorTypes(
        rate_limit=openai.RateLimitError,
        connection=openai.APIConnectionError,
        status=openai.APIStatusError,
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract a parseable ``Retry-After`` header (seconds) from an openai status error, if any."""
    response = cast("httpx.Response | None", getattr(exc, "response", None))
    if response is None:
        return None
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        value = float(header)
    except ValueError:
        return None
    return value if value >= 0 else None


def _classify(exc: Exception) -> tuple[bool, float | None]:
    """Classify *exc* under the library retry policy.

    Returns:
        ``(retryable, retry_after_seconds)`` — ``retry_after_seconds`` is only ever non-``None``
        for a rate limit carrying a parseable ``Retry-After`` header.
    """
    errors = _openai_errors()
    if isinstance(exc, errors.rate_limit):  # 429 — check first: subclass of APIStatusError
        return True, _retry_after_seconds(exc)
    if isinstance(exc, errors.connection):  # network error; includes APITimeoutError
        return True, None
    if isinstance(exc, errors.status):  # any other HTTP status: retry only 5xx
        status_code = cast(int, getattr(exc, "status_code", 0))
        return status_code >= 500, None
    return False, None  # refusals, ValidationError, and every non-openai error


async def retry_llm(
    fn: Callable[[], Awaitable[CallT]],
    *,
    max_retries: int = 2,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    log_prefix: str = "LLM",
) -> CallT:
    """Run *fn* with the library-wide LLM retry policy (also used for embeddings).

    Retryable (waits, then retries, up to *max_retries* additional attempts): rate limits
    (honoring a parseable ``Retry-After`` header), connection/timeout errors, and 5xx server
    errors. Everything else — other 4xx, :class:`LLMRefusalError`, validation errors, non-openai
    errors — raises immediately.

    Backoff schedule (full jitter): ``min(backoff_max, backoff_base * 2**attempt) *
    uniform(0.5, 1.5)``. Each retried failure logs a WARNING with attempt counters; the final
    failure logs an ERROR and re-raises the last underlying exception unchanged (callers keep
    the real openai error type — no wrapper exception).

    Args:
        fn: Zero-argument coroutine factory performing one LLM call attempt.
        max_retries: Maximum *additional* attempts after the first failure.
        backoff_base: Base delay in seconds for the exponential schedule.
        backoff_max: Cap on the pre-jitter delay in seconds.
        log_prefix: Site name used in log messages (e.g. ``"LLMChunker"``).

    Returns:
        The first successful result of *fn*.
    """
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            retryable, retry_after = _classify(exc)
            if not retryable or attempt == max_retries:
                logger.error(
                    "%s: LLM call failed (attempt %d/%d, %s): %s",
                    log_prefix,
                    attempt + 1,
                    max_retries + 1,
                    "giving up" if retryable else "not retryable",
                    exc,
                )
                raise
            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(backoff_max, backoff_base * 2**attempt) * random.uniform(0.5, 1.5)
            logger.warning(
                "%s: retryable LLM failure (attempt %d/%d), retrying in %.1fs: %s",
                log_prefix,
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            await _sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


async def call_structured(
    client: ChatClient,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_format: type[ResponseT],
    temperature: float = 0.0,
    timeout: float | None = None,
    max_retries: int = 2,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    log_prefix: str = "LLM",
) -> ResponseT:
    """One structured-output call with the library retry policy.

    Calls ``client.chat.completions.parse`` (never the deprecated ``beta`` namespace),
    forwarding ``timeout`` per request only when not ``None`` (``None`` defers to the client
    default, typically 600s). Reads ``response.choices[0].message.parsed``; ``None`` — or empty
    ``choices`` — raises :class:`LLMRefusalError`, which is **not** retried (a refusal is
    deterministic for a given input).

    Args:
        client: The chat-capable client to call.
        model: Model name.
        messages: Chat messages for the call.
        response_format: The Pydantic model to parse the response into.
        temperature: Sampling temperature.
        timeout: Per-request timeout in seconds; ``None`` uses the client default.
        max_retries: Maximum additional attempts on retryable transport errors.
        backoff_base: Base delay in seconds for the exponential backoff.
        backoff_max: Cap on the pre-jitter backoff delay in seconds.
        log_prefix: Site name used in log messages.

    Returns:
        The parsed *response_format* instance.

    Raises:
        LLMRefusalError: The model returned no parsed payload.
    """

    async def _once() -> ResponseT:
        if timeout is not None:
            response = await client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
        else:
            response = await client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
            )
        choices = response.choices
        if not choices:
            raise LLMRefusalError(f"{log_prefix}: model {model!r} returned no choices (refusal).")
        parsed = choices[0].message.parsed
        if parsed is None:
            raise LLMRefusalError(
                f"{log_prefix}: model {model!r} returned no parsed {response_format.__name__} "
                "(refusal or schema failure)."
            )
        return parsed

    return await retry_llm(
        _once, max_retries=max_retries, backoff_base=backoff_base, backoff_max=backoff_max, log_prefix=log_prefix
    )
