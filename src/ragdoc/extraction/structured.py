"""StructuredExtractor — typed entity extraction returning ``Mention`` envelopes.

A :class:`StructuredExtractor` renders a (post-split) document, asks an LLM to extract every
instance of a caller-supplied payload model (e.g. ``Event``), and **returns** the results as a
list of :class:`~ragdoc.extraction.mention.Mention` objects. It never reads or writes
``document.metadata`` keys of its own — extraction is a pure query over the document
(:class:`~ragdoc.extraction.extractor.Extractor` protocol). Callers that genuinely want mentions
serialized into a document (e.g. a DocumentStore dump) use the explicit
:func:`~ragdoc.extraction.extractor.as_processor` adapter.

The caller's model *is* the schema: its docstring and ``Field(description=...)`` are serialised into
the JSON schema the LLM is constrained to. Extraction always returns a list, via a synthesised
``ExtractionBatch`` wrapper (this sidesteps the "one-of" ambiguity of a bare ``list[T]`` output).

Customising the prompt (environment)::

    EXTRACTION_SYSTEM_PROMPT="..."        # inline prompt
    EXTRACTION_SYSTEM_PROMPT_FILE=/path   # or read from a file
    EXTRACTION_MIN_TOKENS=0
    EXTRACTION_MODEL_NAME=gpt-4o
"""

from __future__ import annotations

import asyncio
import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragdoc.extraction._llm import (
    build_messages,
    parse_with_retry,
    resolve_model,
    resolve_renderer,
    resolve_tokenizer,
)
from ragdoc.extraction.mention import Mention, PayloadT, mint_mention_id
from ragdoc.llm import ChatClient, resolve_openai_client
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
    """Configuration for the extraction stage (env prefix ``EXTRACTION_``).

    ``system_prompt`` resolution: explicit value → ``EXTRACTION_SYSTEM_PROMPT`` env →
    ``system_prompt_file`` contents → ``None``. A ``None`` prompt means "use the extractor's own
    built-in default" — :class:`StructuredExtractor` falls back to
    :data:`EXTRACTION_SYSTEM_PROMPT`, :class:`~ragdoc.extraction.kg.KnowledgeGraphExtractor` to
    its KG-specific prompt. The validator deliberately does **not** inject a built-in default, so
    each extractor can distinguish "user set a prompt" from "unset".
    """

    model_config = SettingsConfigDict(env_prefix="EXTRACTION_", extra="ignore")

    system_prompt: str | None = Field(
        default=None,
        description=(
            "The extraction instruction, honored by both extractors. ``None`` resolves to the "
            "``system_prompt_file`` contents, else each extractor's built-in default."
        ),
    )
    system_prompt_file: str | None = Field(
        default=None, description="Path to read the prompt from when ``system_prompt`` is unset."
    )
    min_tokens: int = Field(
        default=0,
        description=(
            "Splits whose rendered token count is below this are skipped (no LLM call, empty "
            "result). Default ``0`` — extract from every split."
        ),
    )
    model_name: str | None = Field(
        default=None, description="Model to use. ``None`` falls back to ``get_config().default_llm_model``."
    )
    temperature: float = Field(
        default=0.0, description="Sampling temperature (``0.0`` for the most reproducible extraction)."
    )
    max_retries: int = Field(default=2, description="Maximum retries on an LLM API failure.")
    request_timeout: float | None = Field(
        default=None,
        description=(
            "Per-call timeout (seconds) for the LLM ``.parse()`` request, honored by both "
            "extractors. ``None`` defers to the openai client's default (typically 600s). Set "
            "lower (e.g. ``180``) to fail-fast on pathological inputs so the caller can react — "
            "e.g. ``KnowledgeGraphExtractor`` falls back to halving sooner."
        ),
    )
    gleaning: bool = Field(
        default=False, description="KG: run one extra ``.parse()`` pass asking the LLM for missed entities."
    )
    max_union_size: int = Field(
        default=10,
        description=(
            "KG: maximum node/edge discriminated-union variants (was env ``KG_MAX_UNION_SIZE``). "
            "OpenAI strict-mode reliability drops with large unions; checked at "
            "``KnowledgeGraphExtractor`` construction."
        ),
    )
    halving_max_depth: int = Field(
        default=3, description="KG: maximum recursive halving depth when an LLM extraction call fails."
    )
    halving_min_chars: int = Field(
        default=1000, description="KG: below this text length, halving re-raises instead of splitting further."
    )
    on_pattern_violation: Literal["drop", "error"] = Field(
        default="drop",
        description=(
            "KG: what to do with an extracted edge whose (source_kind, edge_kind, target_kind) "
            "triple is not in the schema's patterns. ``drop`` withholds the edge (counted, one "
            "WARNING per document); ``error`` raises ValueError naming the illegal triple."
        ),
    )
    # Resolution (Layer 2) knobs.
    cosine_threshold: float = Field(
        default=0.9,
        description="Resolution blocking — minimum cosine similarity for two clusters to be merge candidates.",
    )
    auto_merge_bar: float = Field(
        default=0.5,
        description=(
            "Resolution — minimum LLM confidence to auto-merge a reviewed group; below this the "
            "group is recorded as pending instead."
        ),
    )
    max_iterations: int = Field(
        default=3, description="Resolution — hard cap on refinement rounds (backstop for non-convergence)."
    )
    top_k: int = Field(default=5, description="Resolution blocking — nearest neighbours considered per cluster.")

    @model_validator(mode="after")
    def _resolve_prompt(self) -> ExtractionSettings:
        if self.system_prompt is None and self.system_prompt_file:
            self.system_prompt = Path(self.system_prompt_file).read_text(encoding="utf-8").strip()
        return self


# ---------------------------------------------------------------------------
# Schema + prompt helpers
# ---------------------------------------------------------------------------


@cache
def _batch_model(payload_model: type[T]) -> type[BaseModel]:
    """Return (cached) an ``ExtractionBatch`` model wrapping ``mentions: list[payload_model]``.

    ``client.chat.completions.parse`` needs a concrete ``response_format`` class, and a list
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
    """Build the ``messages`` list for an extraction LLM call (documents the per-extractor defaults)."""
    return build_messages(text, system_prompt=system_prompt, user_message=user_message)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class StructuredExtractor(Generic[PayloadT]):
    """Extracts caller-typed entities from a Document as ``Mention[PayloadT]`` objects.

    Implements the :class:`~ragdoc.extraction.extractor.Extractor` protocol: :meth:`extract` is a
    pure query — it never reads or writes ``document.metadata`` keys of its own (each mention's
    locator *copies* the document metadata). Callers that need caching cache; callers that want
    the mentions dumped into a document use
    :func:`~ragdoc.extraction.extractor.as_processor`.

    Provenance is best-effort here (``source_id``/``source_hash``/``content_hash`` read from the
    document, falling back to its id) so standalone use works; the ``MentionStorePipeline``
    restamps authoritative, source-uniform provenance on the returned mentions.

    Args:
        payload_model: The Pydantic model to extract (its docstring + ``Field`` descriptions are
            the schema), e.g. ``Event``.
        client: Async OpenAI-compatible client. ``None`` falls back to ``get_config().openai_client``,
            resolved fail-loud at construction.
        model: Model name. ``None`` falls back to ``settings.model_name`` then config default.
        settings: :class:`ExtractionSettings`. ``None`` reads env (``EXTRACTION_*``).
        renderer: Renderer for document → text. ``None`` uses ``Renderer(MARKDOWN, render_for_prompt)``.
        tokenizer: Tokenizer for the ``min_tokens`` decision. ``None`` uses ``GPTTokenizer``.

    Raises:
        LLMNotConfiguredError: when no client is given and the active config has none.
    """

    def __init__(
        self,
        payload_model: type[PayloadT],
        client: ChatClient | None = None,
        model: str | None = None,
        settings: ExtractionSettings | None = None,
        renderer: Renderer | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._payload_model = payload_model
        self.settings = settings or ExtractionSettings()
        self._client: ChatClient = resolve_openai_client(client)
        self._model = model
        self._renderer = renderer
        self._tokenizer = tokenizer

    @property
    def payload_model(self) -> type[PayloadT]:
        """The payload model this extractor extracts (parametrizes ``ChangeSet[Mention[...]]`` and store queries)."""
        return self._payload_model

    async def extract(self, document: Document) -> list[Mention[PayloadT]]:
        """Extract every mention of the payload type from *document* (no metadata side effects)."""
        renderer = resolve_renderer(self._renderer)
        # Rendering is sync CPU/pandoc work — hop off the event loop (chunker precedent).
        rendered = await asyncio.to_thread(renderer.render, document)
        if not rendered.strip():
            return []
        min_tokens = self.settings.min_tokens
        if min_tokens > 0 and resolve_tokenizer(self._tokenizer).count(rendered) < min_tokens:
            logger.debug("StructuredExtractor: below min_tokens, skipping extraction")
            return []

        client = self._client
        model = resolve_model(self._model, self.settings)
        batch_cls = _batch_model(self._payload_model)
        messages = build_extraction_messages(
            rendered, system_prompt=self.settings.system_prompt or EXTRACTION_SYSTEM_PROMPT
        )
        batch = await parse_with_retry(
            client,
            model=model,
            messages=messages,
            response_format=batch_cls,
            settings=self.settings,
            log_prefix="StructuredExtractor",
        )
        # ``batch_cls`` is a Pydantic model built at runtime, so its field is invisible statically.
        payloads: list[PayloadT] = list(batch.mentions)  # pyright: ignore[reportAttributeAccessIssue]

        content_hash = document.content_hash()
        source_id = document.source_id or document.source_path or document.id
        # Honestly optional: the file-byte hash, or None — never faked from the content hash.
        source_hash = document.source_hash
        metadata_copy = dict(document.metadata)

        mentions: list[Mention[PayloadT]] = []
        for ordinal, payload in enumerate(payloads):
            mentions.append(
                Mention(
                    mention_id=mint_mention_id(source_id, content_hash, ordinal, payload),
                    source_id=source_id,
                    source_path=document.source_path or None,
                    source_hash=source_hash,
                    content_hash=content_hash,
                    ordinal=ordinal,
                    metadata=metadata_copy,
                    payload=payload,
                )
            )
        return mentions
