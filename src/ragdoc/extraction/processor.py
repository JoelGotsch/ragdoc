"""Structured entity extraction into ``document.metadata["mentions"]``.

A :class:`StructuredExtractionProcessor` renders a (post-split) document, asks an LLM to extract
every instance of a caller-supplied payload model (e.g. ``Event``), and writes the results as a
list of :class:`~ragdoc.extraction.mention.Mention` dicts into
``document.metadata[metadata_key]`` — the transport that
:class:`~ragdoc.extraction.pipeline.MentionStorePipeline` later harvests.

The caller's model *is* the schema: its docstring and ``Field(description=...)`` are serialised into
the JSON schema the LLM is constrained to. Extraction always returns a list, via a synthesised
``ExtractionBatch`` wrapper (this sidesteps the "one-of" ambiguity of a bare ``list[T]`` output).

Mirrors :class:`~ragdoc.processing.summary_document.DocumentSummarizerProcessor`: same
client/model fallback, env-var settings with prompt-file indirection, structured ``.parse()`` call,
idempotency guard, and single-key metadata discipline.

Customising the prompt (environment)::

    EXTRACTION_SYSTEM_PROMPT="..."        # inline prompt
    EXTRACTION_SYSTEM_PROMPT_FILE=/path   # or read from a file
    EXTRACTION_MIN_TOKENS=0
    EXTRACTION_MODEL_NAME=gpt-4o
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragdoc.extraction.mention import Mention, mint_mention_id
from ragdoc.processing.base import DocumentProcessor
from ragdoc.utils import Tokenizer

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.document import Document
    from ragdoc.rendering import Renderer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are an expert at extracting structured information from documents.

You will be given a piece of a document. Identify every distinct instance of the requested entity
type that the text describes, and return one structured record per instance. Rules:

1. Extract only what the text actually supports — never invent or infer beyond the text.
2. Return one record per distinct occurrence; if the same entity is described twice, return it twice.
3. Fill each field as completely as the text allows; leave a field null when the text does not say.
4. For dates, copy the wording verbatim into the date's original text and give the canonical form.

If the text contains no such entity, return an empty list.
""".strip()

EXTRACTION_USER_MESSAGE = "Extract all matching entities from the following content."


# ---------------------------------------------------------------------------
# Settings (env-configurable)
# ---------------------------------------------------------------------------


class ExtractionSettings(BaseSettings):
    """Configuration for :class:`StructuredExtractionProcessor` (env prefix ``EXTRACTION_``).

    System-prompt resolution order (highest priority first): explicit ``system_prompt`` →
    ``EXTRACTION_SYSTEM_PROMPT`` → ``system_prompt_file`` contents → built-in
    :data:`EXTRACTION_SYSTEM_PROMPT`.

    Attributes:
        system_prompt: The extraction instruction. ``None`` resolves to the file or built-in.
        system_prompt_file: Path to read the prompt from when ``system_prompt`` is unset.
        min_tokens: Splits whose rendered token count is below this are skipped (no LLM call,
            empty result). Default ``0`` — extract from every split.
        model_name: Model to use. ``None`` falls back to ``get_config().default_llm_model``.
        temperature: Sampling temperature (``0.0`` for the most reproducible extraction).
        max_retries: Maximum retries on an LLM API failure.
        cosine_threshold: Resolution blocking — minimum cosine similarity for two clusters to be
            considered a merge candidate.
        auto_merge_bar: Resolution — minimum LLM confidence to auto-merge a reviewed group; below
            this the group is recorded as pending instead.
        max_iterations: Resolution — hard cap on refinement rounds (backstop for non-convergence).
        top_k: Resolution blocking — nearest neighbours considered per cluster.
    """

    model_config = SettingsConfigDict(env_prefix="EXTRACTION_", extra="ignore")

    system_prompt: str | None = Field(default=None)
    system_prompt_file: str | None = Field(default=None)
    min_tokens: int = Field(default=0)
    model_name: str | None = Field(default=None)
    temperature: float = Field(default=0.0)
    max_retries: int = Field(default=2)
    request_timeout: float | None = Field(
        default=None,
        description=(
            "Per-call timeout (seconds) for the LLM ``.parse()`` request. ``None`` defers "
            "to the openai client's default (typically 600s). Set lower (e.g. ``180``) to "
            "fail-fast on pathological inputs so the caller can react — e.g. "
            "``KnowledgeGraphProcessor._extract_with_halving`` falls back to halving sooner."
        ),
    )
    # Resolution (Layer 2) knobs.
    cosine_threshold: float = Field(default=0.9)
    auto_merge_bar: float = Field(default=0.5)
    max_iterations: int = Field(default=3)
    top_k: int = Field(default=5)

    @model_validator(mode="after")
    def _resolve_prompt(self) -> ExtractionSettings:
        if self.system_prompt is None:
            if self.system_prompt_file:
                self.system_prompt = Path(self.system_prompt_file).read_text(encoding="utf-8").strip()
            else:
                self.system_prompt = EXTRACTION_SYSTEM_PROMPT
        return self


# ---------------------------------------------------------------------------
# Schema + prompt helpers
# ---------------------------------------------------------------------------


@cache
def _batch_model(payload_model: type[T]) -> type[BaseModel]:
    """Return (cached) an ``ExtractionBatch`` model wrapping ``mentions: list[payload_model]``.

    ``client.beta.chat.completions.parse`` needs a concrete ``response_format`` class, and a list
    wrapper avoids the bare-``list[T]`` "one-of" ambiguity while giving the LLM a labelled place to
    return many instances.
    """
    return create_model(
        "ExtractionBatch",
        mentions=(list[payload_model], Field(..., description="All matching entities found in the content.")),
    )


def build_extraction_messages(
    text: str,
    *,
    system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    user_message: str = EXTRACTION_USER_MESSAGE,
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for an extraction LLM call (override the prompt for `partial`)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_message}\n\n{text}"},
    ]


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class StructuredExtractionProcessor(DocumentProcessor, Generic[T]):
    """Extracts caller-typed entities into ``document.metadata[metadata_key]`` as mention dicts.

    Idempotent by default: if ``metadata[metadata_key]`` is already set (non-empty) and
    ``overwrite`` is False, :meth:`process` returns unchanged without an LLM call.

    Provenance is best-effort here (``source_id``/``source_hash``/``content_hash`` read from the
    document, falling back to its id) so standalone use works; the ``MentionStorePipeline`` restamps
    authoritative, source-uniform provenance after harvesting.

    Args:
        payload_model: The Pydantic model to extract (its docstring + ``Field`` descriptions are
            the schema), e.g. ``Event``.
        client: Async OpenAI-compatible client. ``None`` falls back to ``get_config().openai_client``.
        model: Model name. ``None`` falls back to ``settings.model_name`` then config default.
        settings: :class:`ExtractionSettings`. ``None`` reads env (``EXTRACTION_*``).
        renderer: Renderer for document → text. ``None`` uses ``Renderer(MARKDOWN, render_for_prompt)``.
        tokenizer: Tokenizer for the ``min_tokens`` decision. ``None`` uses ``GPTTokenizer``.
        metadata_key: Metadata key to write (default ``"mentions"``).
        overwrite: When False (default), skip documents that already have a non-empty key.
    """

    def __init__(
        self,
        payload_model: type[T],
        client: Any | None = None,
        model: str | None = None,
        settings: ExtractionSettings | None = None,
        renderer: Renderer | None = None,
        tokenizer: Tokenizer | None = None,
        metadata_key: str = "mentions",
        overwrite: bool = False,
    ) -> None:
        self._payload_model = payload_model
        self.settings = settings or ExtractionSettings()
        self._client = client
        self._model = model
        self._renderer = renderer
        self._tokenizer = tokenizer
        self.metadata_key = metadata_key
        self.overwrite = overwrite

    @property
    def payload_model(self) -> type[T]:
        """The payload model this processor extracts (read by ``MentionStorePipeline``)."""
        return self._payload_model

    def _get_client(self) -> Any:
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

    async def _extract(self, text: str, client: Any, model: str) -> list[T]:
        """One structured extraction call with retries; returns the list of payloads."""
        settings = self.settings
        batch_cls = _batch_model(self._payload_model)
        messages = build_extraction_messages(text, system_prompt=settings.system_prompt or EXTRACTION_SYSTEM_PROMPT)
        for attempt in range(settings.max_retries + 1):
            try:
                response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=settings.temperature,
                    response_format=batch_cls,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("LLM returned no parsed extraction")
                return list(parsed.mentions)
            except Exception as exc:
                logger.error(
                    "StructuredExtractor: API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    settings.max_retries + 1,
                    exc,
                )
                if attempt == settings.max_retries:
                    raise
        raise RuntimeError("unreachable")  # pragma: no cover

    async def process(self, document: Document) -> Document:
        existing = document.metadata.get(self.metadata_key)
        if not self.overwrite and existing:
            logger.debug("StructuredExtractor: %s already set, skipping", self.metadata_key)
            return document

        renderer = self._get_renderer()
        rendered = renderer.render(document)
        if not rendered.strip():
            document.metadata[self.metadata_key] = []
            return document
        if self.settings.min_tokens > 0 and self._get_tokenizer().count(rendered) < self.settings.min_tokens:
            logger.debug("StructuredExtractor: below min_tokens, skipping extraction")
            document.metadata[self.metadata_key] = []
            return document

        payloads = await self._extract(rendered, self._get_client(), self._get_model())

        content_hash = document.content_hash()
        source_id = document.source_id or document.source_path or document.id
        source_hash = document.source_hash or content_hash
        # Snapshot metadata BEFORE writing our key so a mention never recursively embeds the list.
        meta_snapshot = {k: v for k, v in document.metadata.items() if k != self.metadata_key}

        mentions: list[Mention[T]] = []
        for ordinal, payload in enumerate(payloads):
            mentions.append(
                Mention(
                    mention_id=mint_mention_id(source_id, content_hash, ordinal, payload),
                    source_id=source_id,
                    source_path=document.source_path or None,
                    source_hash=source_hash,
                    content_hash=content_hash,
                    ordinal=ordinal,
                    metadata=meta_snapshot,
                    payload=payload,
                )
            )

        document.metadata[self.metadata_key] = [m.model_dump(mode="json") for m in mentions]
        return document
