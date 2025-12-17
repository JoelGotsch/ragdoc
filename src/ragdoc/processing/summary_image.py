"""Image summarization: prompt building, OpenAI factory, and DocumentProcessor.

Typical usage::

    from openai import AsyncOpenAI
    from ragdoc.processing.summary_image import openai_image_summarizer, ImageSummaryProcessor

    summarize = openai_image_summarizer(AsyncOpenAI(), model="gpt-4o")
    processor = ImageSummaryProcessor(summarize=summarize)
    doc = await processor.process(doc)

Customising the prompt::

    from functools import partial
    from ragdoc.processing.summary_image import build_image_messages

    my_messages = partial(build_image_messages, system_prompt="My custom prompt")
    summarize = openai_image_summarizer(client, create_messages=my_messages)
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Literal, TypeAlias

from ragdoc.document import Image
from ragdoc.llm import ChatClient, LLMRefusalError, call_structured, resolve_openai_client
from ragdoc.processing.base import DocumentProcessor
from ragdoc.processing.summary_base import ImageSummary
from ragdoc.utils.concurrency import fan_out

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from PIL.Image import Image as PILImage

    from ragdoc.document import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ImageSummarizeFn: TypeAlias = Callable[[Image, "str | None"], Awaitable[ImageSummary]]

# ---------------------------------------------------------------------------
# PIL utilities
# ---------------------------------------------------------------------------

IMAGE_SYSTEM_PROMPT = """
You are an advanced image preprocessor designed to analyze and extract structured data from images.
Given an image, your task is to provide the following:
    - A detailed and clear textual summary of the image's content.
    - If the image contains a table, convert it into accurate HTML (including, if present, a caption) and provide it under the "text_representation" field.
    - If the image contains a graph (e.g. an organization chart or flowchart), provide the precise mermaid code representation under the "text_representation" field.
    - If the image includes a mathematical formula, convert it into exact MathML and provide it under the "text_representation" field.
    - If the image is purely decorative and carries no informational value (e.g. logos, spacers, dividers, horizontal rules, decorative borders, generic stock photos), set "decorative" to true.

Ensure that the extracted data is comprehensive and correctly formatted. Respond with complete data based on your analysis.
""".strip()

IMAGE_USER_MESSAGE = "Please analyze the provided image."


def _import_pil_module():
    """Lazy PIL import — pillow lives behind the 'llm' extra."""
    try:
        from PIL import Image as pil_module
    except ImportError as exc:
        raise ImportError(
            "Image transformations require pillow, installed with the 'llm' extra: pip install 'ragdoc[llm]'"
        ) from exc
    return pil_module


def _remove_alpha_channel(image: PILImage, background: str = "white") -> PILImage:
    """Composite image onto a solid background to remove the alpha channel."""
    if "A" not in image.mode:
        return image
    pil_module = _import_pil_module()
    bg = pil_module.new("RGBA", image.size, background)
    return pil_module.alpha_composite(bg, image)


DEFAULT_TRANSFORMATIONS: list[Callable[[PILImage], PILImage]] = [
    partial(_remove_alpha_channel, background="white"),
]


def _to_pil(image_bytes: bytes) -> PILImage:
    return _import_pil_module().open(BytesIO(image_bytes))


def _pil_to_base64(image: PILImage) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def apply_image_transformations(
    image_bytes: bytes,
    transformations: list[Callable[[PILImage], PILImage]],
) -> PILImage:
    """Open ``image_bytes`` with PIL and apply each transformation in order."""
    pil = _to_pil(image_bytes)
    for transform in transformations:
        pil = transform(pil)
    return pil


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_image_messages(
    base64_image: str,
    image_type: str,
    context: str | None,
    *,
    system_prompt: str = IMAGE_SYSTEM_PROMPT,
    user_message: str = IMAGE_USER_MESSAGE,
    image_detail: Literal["high", "low", "auto"] = "high",
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for an image summarization API call.

    Args:
        base64_image: Base64-encoded image bytes (already PIL-processed if needed).
        image_type: MIME sub-type, e.g. ``"jpeg"``, ``"png"``.
        context: Optional text injected into the prompt as extra context.
        system_prompt: Override the default system prompt.
        user_message: Override the default user instruction.
        image_detail: OpenAI vision detail level.

    Returns:
        A ``messages`` list suitable for ``client.chat.completions.parse``.
    """
    user_content: list = [{"type": "text", "text": user_message}]
    if context:
        user_content.append({"type": "text", "text": f"**Context:**\n\n{context}"})
    user_content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_type};base64,{base64_image}",
                "detail": image_detail,
            },
        }
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# Type alias for the create_messages parameter accepted by the factory.
ImageMessagesFn: TypeAlias = Callable[[str, str, "str | None"], "list[ChatCompletionMessageParam]"]

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def openai_image_summarizer(
    client: ChatClient,
    model: str | None = None,
    create_messages: ImageMessagesFn = build_image_messages,
    transformations: list[Callable[[PILImage], PILImage]] | None = None,
) -> ImageSummarizeFn:
    """Create an :data:`ImageSummarizeFn` backed by the OpenAI structured-output API.

    Args:
        client: An async OpenAI-compatible client (``AsyncOpenAI`` or ``AsyncAzureOpenAI``).
        model: The model to use for the API call.
            ``None`` reads ``get_config().default_image_llm_model`` at call time.
        create_messages: Function that builds the ``messages`` list.
            Signature: ``(base64_image, image_type, context) -> list``.
            Defaults to :func:`build_image_messages`.
        transformations: PIL transforms applied before encoding.
            ``None`` uses :data:`DEFAULT_TRANSFORMATIONS` (alpha-channel removal).
            Pass ``[]`` to skip all transformations.

    Returns:
        An async callable ``(image, context) -> ImageSummary``.
    """
    from ragdoc.config import get_config

    resolved_model = model if model is not None else get_config().default_image_llm_model
    _transforms = DEFAULT_TRANSFORMATIONS if transformations is None else transformations

    async def _summarize(image: Image, context: str | None) -> ImageSummary:
        if image.image is None:
            raise ValueError("Image has no base64 content to summarize.")
        image_bytes = base64.b64decode(image.image)
        if _transforms:
            # PIL decode/transform/encode is CPU-bound — run off the event loop.
            pil = await asyncio.to_thread(apply_image_transformations, image_bytes, _transforms)
            b64 = await asyncio.to_thread(_pil_to_base64, pil)
            img_type = "png"  # PIL always saves as PNG after transforms
        else:
            b64 = image.image  # already base64; skip PIL round-trip
            img_type = image.image_type

        messages = create_messages(b64, img_type, context)
        return await call_structured(
            client,
            model=resolved_model,
            messages=messages,
            response_format=ImageSummary,
            temperature=0.0,
            log_prefix="ImageSummary",
        )

    return _summarize


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class ImageSummaryProcessor(DocumentProcessor):
    """Summarizes all :class:`~ragdoc.document.Image` elements in a document.

    Skips images that already have a ``text_representation``. Images are processed
    with bounded concurrency (default: sequential).

    Args:
        summarize: An :data:`ImageSummarizeFn` callable ``(image, context) -> ImageSummary``.
            When ``None``, falls back to
            ``openai_image_summarizer(get_config().openai_client)`` at
            :meth:`process` time.
        context_fn: Optional ``(image, document) -> str | None`` injecting
            per-image context into the LLM prompt.
        concurrency: Maximum number of images summarized concurrently.  Accepts
            an ``int`` (private semaphore) or a shared ``asyncio.Semaphore``.
            Defaults to ``1`` (sequential).  Pass a shared semaphore to enforce
            a cross-processor LLM-call budget.

    Example::

        summarize = openai_image_summarizer(AsyncOpenAI())
        processor = ImageSummaryProcessor(summarize=summarize)

        # Bounded parallelism:
        processor = ImageSummaryProcessor(summarize=summarize, concurrency=3)

        # Shared budget across processors:
        shared = asyncio.Semaphore(5)
        image_proc = ImageSummaryProcessor(summarize=summarize, concurrency=shared)
    """

    def __init__(
        self,
        summarize: ImageSummarizeFn | None = None,
        context_fn: Callable[[Image, Document], str | None] | None = None,
        concurrency: int | asyncio.Semaphore = 1,
    ):
        self._summarize = summarize
        self._context_fn = context_fn
        self._concurrency = concurrency

    def _get_summarize(self) -> ImageSummarizeFn:
        if self._summarize is not None:
            return self._summarize
        # Fail-loud when no client is configured (LLMNotConfiguredError instead of a
        # None-attribute crash mid-document).
        return openai_image_summarizer(resolve_openai_client(None))

    async def process(self, document: Document) -> Document:
        summarize = self._get_summarize()
        all_images = [img for img in document.images if img.image is not None]
        images = [img for img in all_images if not img.text_representation]
        skipped = len(all_images) - len(images)
        if images:
            logger.info(
                f"ImageSummaryProcessor: {len(images)} images to summarize"
                + (f" ({skipped} skipped: already summarized)" if skipped else "")
            )
        coros: list[Awaitable[None]] = [self._process_one(summarize, img, document) for img in images]
        await fan_out(coros, self._concurrency)
        return document

    async def _process_one(
        self,
        summarize: ImageSummarizeFn,
        image: Image,
        document: Document,
    ) -> None:
        context = self._context_fn(image, document) if self._context_fn else None
        try:
            result = await summarize(image, context)
            if result.decorative:
                image.text_representation = None
            else:
                image.text_representation = result.text_representation or result.summary
        except OSError:
            # PIL's UnidentifiedImageError subclasses OSError, so this also covers unreadable
            # image payloads without importing PIL here (pillow lives behind the 'llm' extra).
            logger.exception(f"Failed to summarize image {image.id}: image type {image.image_type!r} not supported")
        except (LLMRefusalError, ValueError) as exc:
            # Contain per-image LLM refusals/schema failures — one bad image must not abort the
            # document; the image keeps text_representation=None (skip-with-warning).
            logger.warning(f"Failed to summarize image {image.id}: {exc}")
