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
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragdoc.llm import ChatClient, call_structured, resolve_openai_client
from ragdoc.processing.base import DocumentProcessor
from ragdoc.processing.summary_base import DocumentSummary
from ragdoc.utils import Tokenizer
from ragdoc.utils.concurrency import fan_out

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.document import Document
    from ragdoc.rendering import Renderer
    from ragdoc.splitting.base import Splitter

logger = logging.getLogger(__name__)


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
    max_retries: int = Field(default=2, description="Maximum retries on retryable LLM transport failures.")
    request_timeout: float | None = Field(
        default=None, description="Per-request timeout in seconds (None uses the client default)."
    )
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
        A ``messages`` list suitable for ``client.chat.completions.parse``.
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
        client: Async OpenAI-compatible client. ``None`` falls back to ``get_config().openai_client``;
            when neither is available, :class:`~ragdoc.llm.LLMNotConfiguredError` is raised at
            construction (fail-loud, never at :meth:`process` time).
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
        client: ChatClient | None = None,
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
        # Fail-loud at construction (library client policy): explicit → config → LLMNotConfiguredError.
        self._client: ChatClient = resolve_openai_client(client)
        self._model = model
        self._renderer = renderer
        self._splitter = splitter
        self._tokenizer = tokenizer
        self.metadata_key = metadata_key
        self.overwrite = overwrite
        self._concurrency = concurrency

    def _get_model(self) -> str:
        if self._model is not None:
            return self._model
        if self.settings.model_name:
            return self.settings.model_name
        from ragdoc.config import get_config

        return get_config().default_llm_model

    def _get_renderer(self) -> Renderer:
        from ragdoc.rendering import resolve_renderer

        return resolve_renderer(self._renderer)

    def _get_tokenizer(self) -> Tokenizer:
        from ragdoc.utils import resolve_tokenizer

        return resolve_tokenizer(self._tokenizer)

    def _get_splitter(self, renderer: Renderer, tokenizer: Tokenizer) -> Splitter:
        if self._splitter is not None:
            return self._splitter
        from ragdoc.splitting.token import split_document

        max_tokens = self.settings.max_input_tokens
        return lambda document: split_document(document, renderer, tokenizer, max_tokens=max_tokens)

    async def process(self, document: Document) -> Document:
        if not self.overwrite and document.metadata.get(self.metadata_key):
            logger.debug("DocumentSummarizer: %s already set, skipping", self.metadata_key)
            return document

        renderer = self._get_renderer()
        tokenizer = self._get_tokenizer()
        # Rendering shells out to pandoc for non-HTML formats — blocking work off the event loop.
        rendered = await asyncio.to_thread(renderer.render, document)
        if not rendered.strip():
            logger.debug("DocumentSummarizer: empty render, nothing to summarize")
            return document

        client = self._client
        model = self._get_model()
        splitter = self._get_splitter(renderer, tokenizer)
        settings = self.settings

        async def call(text: str) -> str:
            """One LLM summarization call (library retry policy; exhaustion re-raises)."""
            # _resolve_and_validate always sets a non-None system_prompt before the processor runs.
            assert settings.system_prompt is not None
            messages = build_summary_messages(text, system_prompt=settings.system_prompt)
            parsed = await call_structured(
                client,
                model=model,
                messages=messages,
                response_format=DocumentSummary,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
                max_retries=settings.max_retries,
                log_prefix="DocumentSummarizer",
            )
            return parsed.summary

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

            await fan_out([_run(i, c) for i, c in enumerate(coros)], self._concurrency)
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
            text = await asyncio.to_thread(renderer.render, doc)
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
            # Splitting renders per element/group to measure token budgets — blocking work
            # (pandoc subprocesses) off the event loop, mirroring ChunkPipeline.run.
            pieces = await asyncio.to_thread(splitter, doc_copy)
            if len(pieces) <= 1:
                logger.warning("DocumentSummarizer: splitter could not split oversized content; truncating")
                return await call(tokenizer.truncate(text, settings.max_input_tokens))
            piece_summaries = await fan_out_summaries([summarize_doc(p, depth + 1, n) for p in pieces])
            return await reduce(piece_summaries, depth + 1)

        document.metadata[self.metadata_key] = await summarize_doc(document, 0, None)
        return document
