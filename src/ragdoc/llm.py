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
  :class:`LLMClient` is their intersection (a client usable for both, e.g. ``AsyncOpenAI``).
  :class:`~ragdoc.config.RagdocConfig` accepts either capability; :func:`resolve_openai_client`
  demands the chat one.

Retry policy (transport errors only — each site keeps its own failure *product*, e.g. the heading
resolver degrades to ``[]`` while the chunker raises):

- **Retried** (full-jitter exponential backoff): ``openai.RateLimitError`` (429),
  ``openai.APIConnectionError`` (network — includes ``openai.APITimeoutError``), and any
  ``openai.APIStatusError`` with ``status_code >= 500``. A parseable ``Retry-After`` header on a
  429 replaces the computed backoff but is **capped at** ``backoff_max`` — a server asking for a
  day-long pause never parks a pipeline for a day; the capped retries are bounded by
  ``max_retries`` and the last error then propagates.
- **Not retried** (raised immediately): other 4xx (400/401/403/404/422 — deterministic request
  errors), :class:`LLMRefusalError` (a refusal is deterministic for a given input — retrying
  burns tokens for the same answer), ``pydantic.ValidationError``, and every non-openai error.

Exactly one retry layer. ``openai.AsyncOpenAI`` retries on its own (constructor default
``max_retries=2``: 408/409/429/5xx and connection errors), which stacked under this module would
turn "``max_retries=2``" into up to 3 x 3 = 9 HTTP attempts. :func:`without_sdk_retries` therefore
derives a per-call view of a real SDK client via ``client.with_options(max_retries=0)`` (the SDK's
per-request-options mechanism: same connection pool, credentials, base URL and timeout; the
caller's client object is never mutated and no client is ever constructed on the caller's behalf).
:func:`call_structured` and :class:`~ragdoc.pipeline.embedders.OpenAIEmbedder` both apply it, so
for ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` the attempt count is exactly ``max_retries + 1`` and the
client's own ``max_retries`` setting is **ignored** for ragdoc calls. A custom structural client
(anything that is not an ``openai.AsyncOpenAI`` instance) is called as-is: if it retries
internally, those retries multiply with this policy and are the caller's to configure.

Timeouts. Every call site accepts a per-request timeout (seconds) and forwards it through this
layer. ``None`` — the default everywhere — sends no per-request timeout, so the client's own
configured timeout applies (``AsyncOpenAI`` default: 600 s, i.e. worst case
``(max_retries + 1) * 600`` s plus backoff per call); set ``AsyncOpenAI(timeout=...)`` for a
corpus-wide default or the per-site knob for one stage. A timeout surfaces as
``openai.APITimeoutError`` and is retried like any connection error.

Failure classification (shared — do not re-implement at call sites): :func:`is_size_related_failure`
(would a *smaller input* plausibly succeed? — drives the KG extractor's halving) and
:func:`is_llm_failure` (is this an LLM-call failure rather than a programming error? — drives
per-item skip-with-warning degradation).

This module never imports ``openai`` at module level — openai's exception classes are loaded
lazily via :func:`_openai_errors` — so ``import ragdoc.llm`` works without the LLM dependency
installed (Phase 8 moves ``openai`` behind an extra).
"""

from __future__ import annotations

import json
import logging
import random
from asyncio import sleep as _sleep
from collections.abc import Awaitable, Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    import httpx
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
ResponseT_co = TypeVar("ResponseT_co", bound=BaseModel, covariant=True)
CallT = TypeVar("CallT")
ClientT = TypeVar("ClientT")


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

    # ``dimensions`` is an *omittable* int, not ``int | None``: the OpenAI SDK's ``create`` rejects
    # an explicit ``None`` (its sentinel is ``Omit``), so a ``None``-accepting protocol would not be
    # satisfied by a real ``AsyncOpenAI``. Callers omit the argument when no truncation is wanted.
    async def create(
        self, *, model: str, input: list[str], dimensions: int = ..., timeout: float | None = None
    ) -> EmbeddingResponse: ...


@runtime_checkable
class EmbeddingsClient(Protocol):
    """Minimal structural view of an ``AsyncOpenAI``-compatible client: ``embeddings.create``."""

    @property
    def embeddings(self) -> EmbeddingsEndpoint: ...


@runtime_checkable
class LLMClient(ChatClient, EmbeddingsClient, Protocol):
    """A client usable for both structured chat and embeddings.

    ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` satisfy this structurally. No library call site
    requires it — each demands only :class:`ChatClient` or :class:`EmbeddingsClient` — it is
    exported as a convenience annotation for callers that pass one client to both kinds of site.
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
    """No usable client: none passed, and ``get_config().openai_client`` is ``None`` or lacks the capability.

    Raised at construction time.
    """


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
        LLMNotConfiguredError: When no explicit client is given and the active config has none,
            or the configured client is embeddings-only (no ``chat`` attribute).
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
    if not isinstance(client, ChatClient):
        raise LLMNotConfiguredError(
            f"The configured openai_client ({type(client).__name__}) has no chat capability "
            "(no `chat` attribute), but this feature needs chat.completions.parse. Pass a "
            "chat-capable client=... explicitly, or configure one globally: "
            "configure(RagdocConfig(openai_client=AsyncOpenAI(...)))."
        )
    return client


# ---------------------------------------------------------------------------
# Retry engine
# ---------------------------------------------------------------------------


class _OpenAIErrorTypes(NamedTuple):
    """The openai exception classes the retry classifier consults (lazily imported)."""

    base: type[Exception]
    rate_limit: type[Exception]
    connection: type[Exception]
    timeout: type[Exception]
    status: type[Exception]
    length_finish: type[Exception]
    async_client: type[AsyncOpenAI]


@cache
def _openai_errors() -> _OpenAIErrorTypes:
    """Lazy, cached import of openai's exception types (keeps ``ragdoc.llm`` importable without openai)."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError("LLM features require the 'llm' extra: pip install 'ragdoc[llm]'") from exc
    return _OpenAIErrorTypes(
        base=openai.OpenAIError,
        rate_limit=openai.RateLimitError,
        connection=openai.APIConnectionError,
        timeout=openai.APITimeoutError,
        status=openai.APIStatusError,
        length_finish=openai.LengthFinishReasonError,
        async_client=openai.AsyncOpenAI,
    )


def without_sdk_retries(client: ClientT) -> ClientT:
    """Return a view of *client* whose SDK-level retries are disabled (one retry layer — ours).

    For a real ``openai.AsyncOpenAI`` (incl. ``AsyncAzureOpenAI``) this is
    ``client.with_options(max_retries=0)`` — the SDK's per-request-options mechanism: a cheap
    shallow copy sharing the same HTTP connection pool, credentials, base URL and timeout. The
    caller's client is not mutated. Any other client (a custom structural client, a test double,
    or any client when ``openai`` is not installed) is returned unchanged.
    """
    try:
        errors = _openai_errors()
    except ImportError:
        return client
    if isinstance(client, errors.async_client):
        return client.with_options(max_retries=0)
    return client


# OpenAI error codes / message fragments that mean "the request did not fit the model's context".
# Message matching is the fallback for OpenAI-compatible gateways that do not set ``code``.
_CONTEXT_LENGTH_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})
_CONTEXT_LENGTH_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "reduce the length",
)


def is_size_related_failure(exc: BaseException) -> bool:
    """Whether *exc* is plausibly caused by the size of the input or of the requested output.

    ``True`` means "the same call on a *smaller* input may well succeed" — the only failures for
    which shrinking the input (e.g. :class:`~ragdoc.extraction.kg.KnowledgeGraphExtractor`'s
    halving) is a reasonable reaction:

    - **context-length errors** — HTTP 413, or HTTP 400 whose ``code`` is
      ``context_length_exceeded`` / ``string_above_max_length`` or whose message names the context
      length;
    - **truncated output** — ``openai.LengthFinishReasonError`` (``finish_reason == "length"``:
      the completion hit the output-token limit mid-JSON);
    - **malformed structured output** — ``pydantic.ValidationError`` / ``json.JSONDecodeError``
      from parsing the completion. Structured-output drift (repetition loops, half-closed JSON,
      schema violations) grows with output length, and a caller has no other lever than a smaller
      input; the cost is bounded by the caller's own depth limit;
    - **request timeout** — ``openai.APITimeoutError``, which reaches a caller only after the
      shared retry policy is exhausted: repeated timeouts on one input point at a pathologically
      long generation, not at a network blip.

    Everything else is ``False``: authentication/permission errors, rate limits, non-timeout
    connection errors, 5xx, refusals (:class:`LLMRefusalError`), and non-LLM errors do not get
    better with a smaller input. Without ``openai`` installed only the parse failures can match.
    """
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return True
    try:
        errors = _openai_errors()
    except ImportError:
        return False
    if isinstance(exc, (errors.length_finish, errors.timeout)):
        return True
    if isinstance(exc, errors.status):
        status_code = cast(int, getattr(exc, "status_code", 0))
        if status_code == 413:
            return True
        if status_code == 400:
            code = cast("str | None", getattr(exc, "code", None))
            message = str(exc).lower()
            return code in _CONTEXT_LENGTH_CODES or any(marker in message for marker in _CONTEXT_LENGTH_MARKERS)
    return False


def is_llm_failure(exc: BaseException) -> bool:
    """Whether *exc* is a failure *of an LLM call* (as opposed to a programming error).

    ``True`` for ragdoc's :class:`LLMError` family (refusals), any ``openai.OpenAIError`` (HTTP
    status errors incl. 400/401, connection/timeout errors after retries, truncated output), and
    structured-output parse failures (``pydantic.ValidationError`` / ``json.JSONDecodeError``).
    ``False`` for everything else — ``TypeError``, ``AttributeError``, ``KeyError``, … and all
    ``BaseException``-only types such as ``asyncio.CancelledError``.

    Sites with per-item degrade semantics (e.g. image summary: skip-with-warning) use this to
    contain one bad item without swallowing bugs or cancellation.
    """
    if isinstance(exc, (LLMError, ValidationError, json.JSONDecodeError)):
        return True
    try:
        errors = _openai_errors()
    except ImportError:
        return False
    return isinstance(exc, errors.base)


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
        for a rate limit carrying a parseable ``Retry-After`` header (uncapped here;
        :func:`retry_llm` caps it at ``backoff_max``).
    """
    try:
        errors = _openai_errors()
    except ImportError:
        # Base install with a custom structural ChatClient/EmbeddingsClient: there are no
        # openai exception types to classify against. Treat everything as non-retryable so
        # the ORIGINAL exception propagates instead of being masked by an ImportError.
        return False, None
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

    Retryable (waits, then retries, up to *max_retries* additional attempts): rate limits,
    connection/timeout errors, and 5xx server errors. Everything else — other 4xx,
    :class:`LLMRefusalError`, validation errors, non-openai errors — raises immediately.

    Backoff schedule (full jitter): ``min(backoff_max, backoff_base * 2**attempt) *
    uniform(0.5, 1.5)``. A rate limit carrying a parseable ``Retry-After`` header waits
    ``min(retry_after, backoff_max)`` instead (no jitter): the server's hint is honored up to the
    cap, never beyond it — ``Retry-After: 86400`` with ``backoff_max=30`` sleeps 30 s, not a day.
    Each retried failure logs a WARNING with attempt counters; the final failure logs an ERROR
    and re-raises the last underlying exception unchanged (callers keep the real openai error
    type — no wrapper exception).

    *fn* should perform exactly one HTTP attempt; wrap SDK clients with
    :func:`without_sdk_retries` so the SDK's own retries do not multiply with this loop.

    Args:
        fn: Zero-argument coroutine factory performing one LLM call attempt.
        max_retries: Maximum *additional* attempts after the first failure.
        backoff_base: Base delay in seconds for the exponential schedule.
        backoff_max: Cap in seconds on the pre-jitter exponential delay and on an honored
            ``Retry-After``.
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
                delay = min(retry_after, backoff_max)
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

    Calls ``client.chat.completions.parse`` (never the deprecated ``beta`` namespace) on a
    :func:`without_sdk_retries` view of *client* — this function's retry loop is the only one, so
    a real ``AsyncOpenAI`` makes at most ``max_retries + 1`` HTTP attempts. ``timeout`` is
    forwarded per request only when not ``None`` (``None`` defers to the client's configured
    timeout, 600 s for a default ``AsyncOpenAI``). Reads ``response.choices[0].message.parsed``; ``None`` — or empty
    ``choices`` — raises :class:`LLMRefusalError`, which is **not** retried (a refusal is
    deterministic for a given input).

    Args:
        client: The chat-capable client to call.
        model: Model name.
        messages: Chat messages for the call.
        response_format: The Pydantic model to parse the response into.
        temperature: Sampling temperature.
        timeout: Per-request timeout in seconds; ``None`` uses the client default.
        max_retries: Maximum additional attempts on retryable transport errors (the SDK
            client's own ``max_retries`` is bypassed).
        backoff_base: Base delay in seconds for the exponential backoff.
        backoff_max: Cap in seconds on the pre-jitter backoff delay and on an honored ``Retry-After``.
        log_prefix: Site name used in log messages.

    Returns:
        The parsed *response_format* instance.

    Raises:
        LLMRefusalError: The model returned no parsed payload.
    """

    completions = without_sdk_retries(client).chat.completions

    async def _once() -> ResponseT:
        if timeout is not None:
            response = await completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
        else:
            response = await completions.parse(
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
