"""Whole-document summarization into ``document.metadata["summary"]``.

A :class:`DocumentSummarizerProcessor` renders a document, summarizes it with an LLM, and stores
the result in ``document.metadata["summary"]`` — a library-written metadata key intended purely for
external consumers (no other part of the library reads or branches on it).

Long documents that exceed the model context window are folded with a single recursive function and
**one** prompt, governed by two token thresholds:

- ``min_tokens`` (floor) — text at or below it is **passed on unchanged**, no LLM call.
- ``max_input_tokens`` (ceiling) — above it the input is split and folded recursively.

::

    summarize(text):
        if tokens(text) <= min_tokens:        return text          # passed on, no LLM call
        if tokens(text) <= max_input_tokens:  return llm(text)     # one call (map OR reduce)
        else:
            pieces    = split(text)                                # each <= max_input_tokens
            summaries = [summarize(p) for p in pieces]             # recurse, concurrent, ordered
            return summarize(join(summaries))                      # recursive reduce

The first level splits the **Document** (structure-aware, via ``split_document`` on a copy); deeper
levels greedy-pack the summary **strings** up to ``max_input_tokens`` (:func:`pack_summaries`).

Typical usage::

    from openai import AsyncOpenAI
    from ragdoc.processing.summary_document import DocumentSummarizerProcessor

    processor = DocumentSummarizerProcessor(client=AsyncOpenAI())
    doc = await processor.process(doc)
    summary = doc.metadata["summary"]

Customising the prompt (programmatic)::

    from ragdoc.processing.summary_document import DocumentSummarizerSettings

    settings = DocumentSummarizerSettings(system_prompt="Summarise in one sentence.")
    processor = DocumentSummarizerProcessor(settings=settings)

Customising the prompt (environment)::

    DOCUMENT_SUMMARIZER_SYSTEM_PROMPT="..."        # inline prompt
    DOCUMENT_SUMMARIZER_SYSTEM_PROMPT_FILE=/path   # or read from a file
    DOCUMENT_SUMMARIZER_MIN_TOKENS=2000
    DOCUMENT_SUMMARIZER_MAX_INPUT_TOKENS=100000
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragdoc.processing._concurrency import _fan_out
from ragdoc.processing.base import DocumentProcessor
from ragdoc.processing.summary_base import DocumentSummary
from ragdoc.utils import Tokenizer

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.document import Document
    from ragdoc.rendering import Renderer
    from ragdoc.splitting.base import Splitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural LLM client (no ``Any``)
# ---------------------------------------------------------------------------
#
# Mirrors ``pipeline.embedders._EmbeddingsClient``: the summarizer calls exactly
# ``client.beta.chat.completions.parse(...)`` and reads ``response.choices[0].message.parsed``,
# so we type only that surface structurally. Any ``AsyncOpenAI``-compatible client satisfies it.


class _ParsedMessage(Protocol):
    """The ``message`` of a parsed chat-completion choice (structural)."""

    parsed: DocumentSummary | None


class _ParsedChoice(Protocol):
    """One choice of a parsed chat-completion response (structural)."""

    message: _ParsedMessage


class _ParsedResponse(Protocol):
    """A parsed chat-completion response (structural)."""

    choices: Sequence[_ParsedChoice]


class _ChatCompletionsParseEndpoint(Protocol):
    """The ``completions`` sub-client exposing structured-output ``parse`` (structural)."""

    async def parse(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        response_format: type[DocumentSummary],
    ) -> _ParsedResponse: ...


class _ChatEndpoint(Protocol):
    """The ``chat`` sub-client of an OpenAI-compatible beta client (structural)."""

    completions: _ChatCompletionsParseEndpoint


class _BetaEndpoint(Protocol):
    """The ``beta`` sub-client of an OpenAI-compatible client (structural)."""

    chat: _ChatEndpoint


class _SummaryClient(Protocol):
    """Minimal structural view of an ``AsyncOpenAI``-compatible client.

    Exposes only ``beta.chat.completions.parse`` — the sole call the summarizer makes.
    """

    beta: _BetaEndpoint


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """
You are an expert at writing concise, faithful summaries.

You will be given some content — either a document or a set of summaries of consecutive sections of
one document. Write a single, self-contained summary of that content. The summary must:

1. Capture the essential facts, entities, topics, and relationships.
2. Read as one coherent piece of prose, not a list of sections.
3. Avoid repetition and commentary about the text itself.
4. Stay faithful to the source — never invent information.
""".strip()

SUMMARY_USER_MESSAGE = "Summarise the following content as described."


# ---------------------------------------------------------------------------
# Settings (env-configurable)
# ---------------------------------------------------------------------------


class DocumentSummarizerSettings(BaseSettings):
    """Configuration for :class:`DocumentSummarizerProcessor`.

    Settings can be configured via environment variables with prefix
    ``DOCUMENT_SUMMARIZER_``. For example::

        DOCUMENT_SUMMARIZER_SYSTEM_PROMPT="Summarise in one sentence."
        DOCUMENT_SUMMARIZER_SYSTEM_PROMPT_FILE=/etc/prompts/summary.txt
        DOCUMENT_SUMMARIZER_MIN_TOKENS=2000
        DOCUMENT_SUMMARIZER_MAX_INPUT_TOKENS=100000
        DOCUMENT_SUMMARIZER_MODEL_NAME=gpt-4o
        DOCUMENT_SUMMARIZER_TEMPERATURE=0.0

    System-prompt resolution order (highest priority first): explicit ``system_prompt`` (constructor
    argument or ``DOCUMENT_SUMMARIZER_SYSTEM_PROMPT``) → ``system_prompt_file`` contents → built-in
    :data:`SUMMARY_SYSTEM_PROMPT`.

    Attributes:
        system_prompt: The summarization prompt, used for both the initial document and any
            recursive combination of summaries. ``None`` resolves to the file or built-in default.
        system_prompt_file: Path to read the prompt from when ``system_prompt`` is unset.
        min_tokens: Floor below which content is passed through unchanged (no LLM call).
        max_input_tokens: Ceiling for a single LLM call; larger inputs are split and folded.
            Must already account for system-prompt and completion overhead.
        model_name: Model to use. ``None`` falls back to ``get_config().default_llm_model``.
        temperature: Sampling temperature (lower = more deterministic).
        max_retries: Maximum retries on an LLM API failure.
        max_recursion_depth: Hard cap on fold depth; exceeding it raises (guards against a prompt
            that fails to compress its input).
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCUMENT_SUMMARIZER_",
        extra="ignore",
    )

    system_prompt: str | None = Field(
        default=None, description="Explicit summarization prompt; None resolves to file or built-in default."
    )
    system_prompt_file: str | None = Field(
        default=None, description="Path to read the prompt from when system_prompt is unset."
    )
    min_tokens: int = Field(
        default=2000, description="Floor below which content is passed through unchanged (no LLM call)."
    )
    max_input_tokens: int = Field(
        default=100_000, description="Ceiling for a single LLM call; larger inputs are split and folded."
    )
    model_name: str | None = Field(
        default=None, description="Model to use; None falls back to get_config().default_llm_model."
    )
    temperature: float = Field(default=0.0, description="Sampling temperature (lower = more deterministic).")
    max_retries: int = Field(default=2, description="Maximum retries on an LLM API failure.")
    max_recursion_depth: int = Field(default=5, description="Hard cap on fold depth; exceeding it raises.")

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> DocumentSummarizerSettings:
        if self.system_prompt is None:
            if self.system_prompt_file:
                self.system_prompt = Path(self.system_prompt_file).read_text(encoding="utf-8").strip()
            else:
                self.system_prompt = SUMMARY_SYSTEM_PROMPT
        if self.min_tokens >= self.max_input_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be strictly less than max_input_tokens ({self.max_input_tokens})."
            )
        return self


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_summary_messages(
    text: str,
    *,
    system_prompt: str = SUMMARY_SYSTEM_PROMPT,
    user_message: str = SUMMARY_USER_MESSAGE,
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for a summarization LLM call.

    Args:
        text: The content to summarize (a rendered document or concatenated summaries).
        system_prompt: Override the default system prompt.
        user_message: Override the default user instruction.

    Returns:
        A ``messages`` list suitable for ``client.beta.chat.completions.parse``.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_message}\n\n{text}"},
    ]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def pack_summaries(summaries: list[str], max_tokens: int, tokenizer: Tokenizer) -> list[list[str]]:
    """Greedily pack summary strings into batches each no larger than ``max_tokens``.

    Strings are kept in order. A single string that exceeds ``max_tokens`` on its own lands in its
    own batch (never dropped). Token counts are summed per string; the small join-separator overhead
    is absorbed by the ``max_input_tokens`` reserve.

    Args:
        summaries: Summary strings to pack, in reading order.
        max_tokens: Maximum combined token count per batch.
        tokenizer: Tokenizer used to count tokens.

    Returns:
        A list of batches (each a list of strings), preserving input order.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for s in summaries:
        s_tokens = tokenizer.count(s)
        if current and current_tokens + s_tokens > max_tokens:
            batches.append(current)
            current = [s]
            current_tokens = s_tokens
        else:
            current.append(s)
            current_tokens += s_tokens
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

_JOIN = "\n\n"


class DocumentSummarizerProcessor(DocumentProcessor):
    """Writes an LLM summary of the whole document to ``document.metadata[metadata_key]``.

    Folds documents larger than ``max_input_tokens`` with a single recursive function and one
    prompt (see module docstring). A document at or below ``min_tokens`` is passed through: its
    rendered text becomes the summary verbatim, with no LLM call.

    Idempotent by default: if ``metadata[metadata_key]`` is already set and ``overwrite`` is False,
    :meth:`process` returns unchanged without an LLM call.

    Args:
        client: Async OpenAI-compatible client. ``None`` falls back to ``get_config().openai_client``.
        model: Model name. ``None`` falls back to ``settings.model_name`` then
            ``get_config().default_llm_model``.
        settings: :class:`DocumentSummarizerSettings`. ``None`` reads env (``DOCUMENT_SUMMARIZER_*``).
        renderer: Renderer for document → text. ``None`` uses ``Renderer(MARKDOWN, render_for_prompt)``.
        splitter: Document splitter. ``None`` uses ``split_document`` bounded to
            ``settings.max_input_tokens``.
        tokenizer: Tokenizer for the threshold decisions. ``None`` uses ``GPTTokenizer``.
        metadata_key: Metadata key to write (default ``"summary"``).
        overwrite: When False (default), skip documents that already have the key.
        concurrency: Maximum concurrent LLM calls within a fold level. ``int`` (private semaphore)
            or a shared ``asyncio.Semaphore``. Defaults to ``1`` (sequential).
    """

    def __init__(
        self,
        client: _SummaryClient | None = None,
        model: str | None = None,
        settings: DocumentSummarizerSettings | None = None,
        renderer: Renderer | None = None,
        splitter: Splitter | None = None,
        tokenizer: Tokenizer | None = None,
        metadata_key: str = "summary",
        overwrite: bool = False,
        concurrency: int | asyncio.Semaphore = 1,
    ) -> None:
        self.settings = settings or DocumentSummarizerSettings()
        self._client = client
        self._model = model
        self._renderer = renderer
        self._splitter = splitter
        self._tokenizer = tokenizer
        self.metadata_key = metadata_key
        self.overwrite = overwrite
        self._concurrency = concurrency

    def _get_client(self) -> _SummaryClient:
        if self._client is not None:
            return self._client
        from ragdoc.config import get_config

        client = get_config().openai_client
        if client is None:
            raise ValueError("No client provided and get_config().openai_client is None.")
        return client

    def _get_model(self) -> str:
        if self._model is not None:
            return self._model
        if self.settings.model_name:
            return self.settings.model_name
        from ragdoc.config import get_config

        return get_config().default_llm_model

    def _get_renderer(self) -> Renderer:
        if self._renderer is not None:
            return self._renderer
        from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

        return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        from ragdoc.utils import GPTTokenizer

        return GPTTokenizer()

    def _get_splitter(self, renderer: Renderer, tokenizer: Tokenizer) -> Splitter:
        if self._splitter is not None:
            return self._splitter
        from ragdoc.splitting.token import split_document

        max_tokens = self.settings.max_input_tokens
        return lambda document: split_document(document, renderer, tokenizer, max_tokens=max_tokens)

    async def process(self, document: Document) -> Document:  # noqa: C901  (recursive map/reduce fold with several closures)
        if not self.overwrite and document.metadata.get(self.metadata_key):
            logger.debug("DocumentSummarizer: %s already set, skipping", self.metadata_key)
            return document

        renderer = self._get_renderer()
        tokenizer = self._get_tokenizer()
        rendered = renderer.render(document)
        if not rendered.strip():
            logger.debug("DocumentSummarizer: empty render, nothing to summarize")
            return document

        client = self._get_client()
        model = self._get_model()
        splitter = self._get_splitter(renderer, tokenizer)
        settings = self.settings

        async def call(text: str) -> str:
            """One LLM summarization call with retries."""
            messages = build_summary_messages(text, system_prompt=settings.system_prompt or SUMMARY_SYSTEM_PROMPT)
            for attempt in range(settings.max_retries + 1):
                try:
                    response = await client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        temperature=settings.temperature,
                        response_format=DocumentSummary,
                    )
                    parsed = response.choices[0].message.parsed
                    if parsed is None:
                        raise ValueError("LLM returned no parsed summary")
                    return parsed.summary
                except Exception as exc:
                    logger.error(
                        "DocumentSummarizer: API call failed (attempt %d/%d): %s",
                        attempt + 1,
                        settings.max_retries + 1,
                        exc,
                    )
                    if attempt == settings.max_retries:
                        raise
            raise RuntimeError("unreachable")  # pragma: no cover

        def _check_depth(depth: int) -> None:
            if depth > settings.max_recursion_depth:
                raise RuntimeError(
                    f"DocumentSummarizer exceeded max_recursion_depth={settings.max_recursion_depth}: "
                    "summaries are not shrinking below max_input_tokens. Check the prompt or raise the limit."
                )

        async def fan_out_summaries(coros: list[Awaitable[str]]) -> list[str]:
            """Run summary coroutines with bounded concurrency, preserving input order."""
            results: list[str] = [""] * len(coros)

            async def _run(i: int, coro: Awaitable[str]) -> None:
                results[i] = await coro

            await _fan_out([_run(i, c) for i, c in enumerate(coros)], self._concurrency)
            return results

        async def reduce(summaries: list[str], depth: int) -> str:
            _check_depth(depth)
            joined = _JOIN.join(summaries)
            n = tokenizer.count(joined)
            if n <= settings.min_tokens:
                return joined
            if n <= settings.max_input_tokens:
                return await call(joined)
            batches = pack_summaries(summaries, settings.max_input_tokens, tokenizer)
            batch_summaries = await fan_out_summaries([call(_JOIN.join(b)) for b in batches])
            return await reduce(batch_summaries, depth + 1)

        async def summarize_doc(doc: Document, depth: int, parent_tokens: int | None) -> str:
            _check_depth(depth)
            text = renderer.render(doc)
            n = tokenizer.count(text)
            if n <= settings.min_tokens:
                return text
            if n <= settings.max_input_tokens:
                return await call(text)
            # No-progress guard: if this piece is no smaller than its parent, the splitter cannot
            # reduce it further (e.g. a single oversized element, or a budget too small for the
            # heading/overlap overhead). Truncate-and-summarize rather than re-splitting forever.
            if parent_tokens is not None and n >= parent_tokens:
                logger.warning("DocumentSummarizer: splitter made no progress (%d tokens); truncating", n)
                return await call(tokenizer.truncate(text, settings.max_input_tokens))
            # Split a copy so split_document's in-place split_sequence/split_total stamp never
            # touches the real document's metadata.
            doc_copy = doc.model_copy(update={"metadata": dict(doc.metadata)})
            pieces = splitter(doc_copy)
            if len(pieces) <= 1:
                logger.warning("DocumentSummarizer: splitter could not split oversized content; truncating")
                return await call(tokenizer.truncate(text, settings.max_input_tokens))
            piece_summaries = await fan_out_summaries([summarize_doc(p, depth + 1, n) for p in pieces])
            return await reduce(piece_summaries, depth + 1)

        document.metadata[self.metadata_key] = await summarize_doc(document, 0, None)
        return document
