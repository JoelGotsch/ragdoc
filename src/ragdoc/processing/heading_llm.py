"""
LLM-based heading level processing.

This module provides an async processor that uses an LLM to determine
heading levels based on visual properties and semantic analysis.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragdoc.config import get_config
from ragdoc.document import Heading, Paragraph
from ragdoc.llm import ChatClient, call_structured
from ragdoc.processing._concurrency import _fan_out
from ragdoc.processing.base import DocumentProcessor

if TYPE_CHECKING:
    from ragdoc.document import Document

logger = logging.getLogger(__name__)


@dataclass
class HeadingInfo:
    """
    Information about a heading for LLM analysis.

    Attributes:
        index: Index of the heading in the elements list
        page_number: 1-based page number
        font_size: Font size (if available from CSS)
        is_centered: Whether the heading is centered
        text: The heading text content
    """

    index: int
    page_number: int
    font_size: float | None
    is_centered: bool
    text: str


class HeadingLevel(str, Enum):
    DOCUMENT_TITLE = "document-title"
    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    H4 = "h4"
    H5 = "h5"
    H6 = "h6"
    NOT_HEADING = "none"
    NOT_DETERMINABLE = "not_determinable"


class HeadingJudgment(BaseModel):
    """
    Model for LLM response about a single heading's level judgment.

    Attributes:
        id: The 1-based ID of the heading (corresponding to the prompt)
        level: The determined level ("document-title", "h1"-"h6", "none", or "not_determinable")
    """

    id: int = Field(
        ...,
        description="The 1-based ID of the heading as given in the prompt.",
    )
    level: HeadingLevel = Field(
        ...,
        description=(
            "The determined level: 'document-title', 'h1'–'h6', "
            "'none' (explicitly not a heading), or 'not_determinable' "
            "(only when no reasonable guess can be made — prefer any concrete level)."
        ),
    )


class HeadingResponse(BaseModel):
    """Wrapper for the structured LLM response containing heading judgments."""

    judgments: list[HeadingJudgment] = Field(
        ...,
        description=(
            "One judgment per heading. Every heading must have an entry — "
            "use 'not_determinable' when no estimate is possible."
        ),
    )


class LLMHeadingResolverSettings(BaseSettings):
    """
    Configuration settings for LLMHeadingResolver.

    Settings can be configured via environment variables with prefix
    `LLM_HEADING_RESOLVER_`. For example:
    - LLM_HEADING_RESOLVER_MODEL_NAME=gpt-4o
    - LLM_HEADING_RESOLVER_TEMPERATURE=0.1

    Attributes:
        model_name: The model to use for heading level determination
        base_url: Base URL for the API (None for default OpenAI)
        api_key: API key for authentication (None for default OpenAI)
        temperature: Temperature for responses (lower = more deterministic)
        max_retries: Maximum retries on API failure
        batch_size: Max headings per LLM call (None = all at once)
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_HEADING_RESOLVER_",
        extra="ignore",
    )

    model_name: str = Field(default="gpt-4o-mini", description="Model used for heading level determination.")
    base_url: str | None = Field(default=None, description="API base URL (None for default OpenAI).")
    api_key: SecretStr | None = Field(default=None, description="API key for authentication (None for default).")
    temperature: float = Field(default=0.0, description="Sampling temperature (lower = more deterministic).")
    max_retries: int = Field(default=2, description="Maximum retries on retryable LLM transport failures.")
    batch_size: int | None = Field(default=None, description="Max headings per LLM call (None = all at once).")
    request_timeout: float | None = Field(
        default=None, description="Per-request timeout in seconds (None uses the client default)."
    )


class LLMHeadingResolver(DocumentProcessor):
    """
    Async processor that uses an LLM to determine heading levels.

    This processor sends heading information to an LLM which analyzes
    visual properties and text content to determine appropriate levels.
    The LLM can identify:

    - document-title: The main document title
    - h1-h6: Standard heading levels
    - none: Text that looks like a heading but isn't

    This is useful for documents with inconsistent heading styles or
    complex structures that benefit from semantic analysis.

    Example:
        ```python
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        resolver = LLMHeadingResolver(client)

        doc = await resolver.process(doc)
        ```
    """

    SYSTEM_PROMPT = """You are analyzing document headings to determine their hierarchical levels.

For each heading you receive:
- A 1-based ID
- Page number (1-based)
- Font size (if available)
- Whether it is centered
- The text content

Return a judgment for each heading using the structured output schema. Set the level to one of:
- "document-title": The main document title (typically largest, centered, on pages 1–2, appears once)
- "h1" through "h6": Standard heading levels; larger/more prominent headings get lower numbers
- "none": Explicitly NOT a heading (metadata, dates, reference numbers, author names, etc.)
- "not_determinable": Reserved for cases where no reasonable guess is possible at all — use sparingly

Every heading must receive a judgment. If any signal at all suggests a level (font size, position, text pattern, context), use that — even a rough guess is better than "not_determinable".

Guidelines:
1. Document titles are typically: large font, centered, on page 1–2, descriptive of the entire document
2. Larger font sizes generally indicate higher-level headings (h1 > h2 > h3...)
3. Centered headings are often higher-level than left-aligned ones
4. Short cryptic text (dates, codes, numbers) on early pages is often metadata, not headings
5. Be consistent: similar visual properties should get similar levels
6. Consider document flow: headings should follow a logical hierarchy
7. Default to the most plausible level based on available signals; use "not_determinable" only when the heading gives no usable information whatsoever"""

    def __init__(
        self,
        client: ChatClient | None = None,
        model_name: str | None = None,
        settings: LLMHeadingResolverSettings | None = None,
        remove_title_from_elements: bool = True,
        remove_elements_before_title: bool = False,
        concurrency: int | asyncio.Semaphore = 1,
    ):
        """
        Initialize the LLM heading resolver.

        Order of settings resolution:
        1. Explicit argument
        2. From env variables (LLM_HEADING_RESOLVER_*)

        Order of client and model_name resolution (the settings-based factory is a documented
        layer above :func:`ragdoc.llm.resolve_openai_client`'s explicit → config policy):
        1. Explicit argument
        2. From settings (see above; base_url and api_key must be provided)
        3. Default client from global config (get_config().openai_client)

        Args:
            client: AsyncOpenAI client instance or compatible
            model_name: Model name to use (e.g., "gpt-4o"). Overrides settings and config.
            settings: Optional settings object. Can also be configured via environment variables.
            remove_title_from_elements: Remove the detected title element from the document.
            remove_elements_before_title: Remove all elements preceding the detected title.
            concurrency: Max concurrent LLM calls across batches (int or shared semaphore).
        """
        self.settings = settings or LLMHeadingResolverSettings()
        config = get_config()
        self.settings.model_name = model_name or self.settings.model_name or config.default_llm_model
        resolved_client = client or self._create_client_from_settings() or config.openai_client
        if resolved_client is None:
            raise ValueError("No client provided and could not create one from settings or config.")
        self.client: ChatClient = resolved_client
        self.remove_title_from_elements = remove_title_from_elements
        self.remove_elements_before_title = remove_elements_before_title
        self._concurrency = concurrency

    def _create_client_from_settings(self) -> ChatClient | None:
        """Create an AsyncOpenAI client from settings if base_url and api_key are provided."""
        if self.settings.base_url and self.settings.api_key and self.settings.api_key.get_secret_value():
            from openai import AsyncOpenAI  # lazy: openai moves behind an extra in Phase 8

            return AsyncOpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key.get_secret_value(),
            )
        return None

    async def process(self, document: Document) -> Document:
        """Analyze all headings using LLM and update their levels."""

        # Idempotency guard: a resolved title means this (or another title processor)
        # already ran. Re-running re-detects a *different* "title" and re-strips the
        # elements before it — the non-idempotency this guard prevents. Mirrors
        # TitleDetectionProcessor.
        if document.title:
            logger.debug("LLMHeadingResolver: document already has a title, skipping")
            return document

        # Collect heading information
        heading_infos = self._collect_heading_infos(document)

        if not heading_infos:
            return document

        logger.info(f"LLMHeadingResolver: resolving {len(heading_infos)} headings (model={self.settings.model_name})")

        # Process headings (possibly in batches)
        if self.settings.batch_size and len(heading_infos) > self.settings.batch_size:
            judgments = await self._process_in_batches(heading_infos)
        else:
            judgments = await self._get_heading_levels(heading_infos)

        # Apply the determined levels
        self._apply_levels(document, heading_infos, judgments)

        return document

    def _collect_heading_infos(self, document: Document) -> list[HeadingInfo]:
        """Collect information about all headings for LLM analysis."""
        import re

        from bs4 import BeautifulSoup

        heading_infos: list[HeadingInfo] = []

        for idx, element in enumerate(document.elements):
            if not isinstance(element, Heading):
                continue

            # Extract font size from CSS
            font_size = None
            soup = BeautifulSoup(element.html, "html.parser")
            for tag in soup.find_all(style=True):
                style = str(tag.get("style", ""))
                match = re.search(r"font-size:\s*([\d.]+)(pt|px)", style)
                if match:
                    font_size = float(match.group(1))
                    if match.group(2) == "px":
                        font_size = font_size * 0.75
                    break

            # Check if centered
            is_centered = "text-align: center" in element.html or "text-align:center" in element.html

            heading_infos.append(
                HeadingInfo(
                    index=idx,
                    page_number=element.page or 1,
                    font_size=font_size,
                    is_centered=is_centered,
                    text=element.text,
                )
            )

        return heading_infos

    def _build_prompt(self, heading_infos: list[HeadingInfo]) -> str:
        """Build the user prompt with heading information."""
        lines = ["Analyze these headings and determine their levels:\n"]

        for i, info in enumerate(heading_infos, 1):
            font_str = f"{info.font_size:.1f}pt" if info.font_size else "unknown"
            centered_str = "Yes" if info.is_centered else "No"
            lines.append(
                f'{i}. Page {info.page_number} | Font: {font_str} | Centered: {centered_str} | Text: "{info.text}"'
            )

        return "\n".join(lines)

    async def _get_heading_levels(self, heading_infos: list[HeadingInfo]) -> list[HeadingJudgment]:
        """Query the LLM to determine heading levels using structured output.

        Degrades to ``[]`` (headings left unchanged) on any final failure — transport retries
        with backoff happen inside :func:`ragdoc.llm.call_structured`; deterministic errors
        (4xx, refusal) are not retried.
        """
        prompt = self._build_prompt(heading_infos)

        try:
            result = await call_structured(
                self.client,
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=HeadingResponse,
                temperature=self.settings.temperature,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                log_prefix="LLMHeadingResolver",
            )
        except Exception as exc:  # noqa: BLE001 -- site degrade semantics: no judgments, headings unchanged
            logger.error(f"LLMHeadingResolver: giving up on heading batch: {exc}")
            return []
        return result.judgments

    async def _process_in_batches(self, heading_infos: list[HeadingInfo]) -> list[HeadingJudgment]:
        """Process headings in batches, with bounded concurrency across batches."""
        batch_size = self.settings.batch_size
        assert batch_size is not None  # guaranteed by the caller's `if self.settings.batch_size` check
        # Pre-compute (id_offset, batch) so offsets are stable even under parallel execution.
        batches = [(i, heading_infos[i : i + batch_size]) for i in range(0, len(heading_infos), batch_size)]

        # Pre-allocate results list: preserves order regardless of completion order.
        results: list[list[HeadingJudgment]] = [[] for _ in batches]

        async def _process_batch(batch_idx: int, id_offset: int, batch: list[HeadingInfo]) -> None:
            batch_judgments = await self._get_heading_levels(batch)
            # Re-map IDs from batch-local (1-based) to global (1-based)
            results[batch_idx] = [HeadingJudgment(id=j.id + id_offset, level=j.level) for j in batch_judgments]

        coros: list[Awaitable[None]] = [
            _process_batch(idx, offset, batch) for idx, (offset, batch) in enumerate(batches)
        ]
        await _fan_out(coros, self._concurrency)

        return [j for r in results for j in r]

    def _apply_levels(
        self,
        document: Document,
        heading_infos: list[HeadingInfo],
        judgments: list[HeadingJudgment],
    ) -> None:
        """
        Apply determined levels to heading elements.

        - Heading absent from judgments → LLM was uncertain, element left unchanged.
        - NOT_DETERMINABLE → same as absent, left unchanged.
        - NOT_HEADING → element converted to Paragraph; ``llm_heading_level = "none"`` in metadata.
        - DOCUMENT_TITLE → ``document.title`` set; metadata marks the element as the title.
        - H1-H6 → ``element.level`` updated; ``llm_heading_level`` set in metadata.
        """
        judgment_by_id = {j.id: j for j in judgments}
        title_element_idx = None

        for i, info in enumerate(heading_infos):
            element = document.elements[info.index]
            if not isinstance(element, Heading):
                continue

            judgment = judgment_by_id.get(i + 1)  # IDs are 1-based

            if judgment is None or judgment.level == HeadingLevel.NOT_DETERMINABLE:
                # LLM was uncertain — leave element unchanged, no metadata
                pass
            elif judgment.level == HeadingLevel.NOT_HEADING:
                # Explicitly not a heading — convert Heading to Paragraph in place
                font_size_css = f"font-size: {info.font_size}pt; " if info.font_size is not None else ""
                align_css = "text-align: center;" if info.is_centered else "text-align: left;"
                new_elem = Paragraph(html=f'<p style="{font_size_css}{align_css}">{element.innerhtml}</p>')
                new_elem.metadata["llm_heading_level"] = "none"
                document.elements[info.index] = new_elem
            elif judgment.level == HeadingLevel.DOCUMENT_TITLE:
                document.title = element.text
                element.metadata["llm_heading_level"] = "document-title"
                element.metadata["is_document_title"] = True
                title_element_idx = info.index
            else:
                # Standard h1-h6
                level_num = int(judgment.level.value[1])
                element.level = level_num
                element.metadata["llm_heading_level"] = judgment.level.value

        if title_element_idx is not None:
            if self.remove_title_from_elements:
                document.elements.pop(title_element_idx)
            if self.remove_elements_before_title:
                for idx in range(title_element_idx - 1, -1, -1):
                    document.elements.pop(idx)
