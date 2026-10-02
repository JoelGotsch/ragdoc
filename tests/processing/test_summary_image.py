"""Tests for ImageSummaryProcessor and related functions (summary_image.py)."""

from __future__ import annotations

import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("PIL", reason="llm extra (pillow) not installed")
from PIL import Image as PILImage, UnidentifiedImageError

from ragdoc.document import Document, Image
from ragdoc.processing.summary_base import ImageSummary
from ragdoc.processing.summary_image import (
    ImageSummaryProcessor,
    build_image_messages,
    openai_image_summarizer,
)


def _tiny_png_base64() -> str:
    """Return a minimal 1x1 white PNG as base64."""
    img = PILImage.new("RGB", (1, 1), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_openai_client() -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


# --- TestBuildImageMessages ---


def test_build_image_messages_contains_system_and_user():
    msgs = build_image_messages("b64data", "jpeg", context=None)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]


def test_build_image_messages_image_url_uses_correct_mime():
    msgs = build_image_messages("b64data", "png", context=None)
    user_content = msgs[1]["content"]
    image_block = next(b for b in user_content if b.get("type") == "image_url")
    assert "data:image/png;base64,b64data" in image_block["image_url"]["url"]


def test_build_image_messages_context_injected_when_provided():
    msgs = build_image_messages("b64", "jpeg", context="some context")
    user_content = msgs[1]["content"]
    texts = [b["text"] for b in user_content if b.get("type") == "text"]
    assert any("some context" in t for t in texts)


def test_build_image_messages_no_context_block_when_none():
    msgs = build_image_messages("b64", "jpeg", context=None)
    user_content = msgs[1]["content"]
    texts = [b["text"] for b in user_content if b.get("type") == "text"]
    assert not any("Context" in t for t in texts)


def test_build_image_messages_custom_system_prompt():
    msgs = build_image_messages("b64", "jpeg", None, system_prompt="CUSTOM")
    assert msgs[0]["content"] == "CUSTOM"


def test_build_image_messages_image_detail_default_is_high():
    msgs = build_image_messages("b64", "jpeg", context=None)
    user_content = msgs[1]["content"]
    image_block = next(b for b in user_content if b.get("type") == "image_url")
    assert image_block["image_url"]["detail"] == "high"


# --- TestImageSummaryProcessor ---


def test_image_processor_unconfigured_raises_at_construction():
    """No summarize fn, no client anywhere → LLMNotConfiguredError in __init__, not process()."""
    from ragdoc.config import RagdocConfig, configure
    from ragdoc.llm import LLMNotConfiguredError

    with configure(RagdocConfig()), pytest.raises(LLMNotConfiguredError):
        ImageSummaryProcessor()


def test_image_processor_custom_summarize_fn_needs_no_client():
    """A custom summarize fn must not trigger client resolution."""
    from ragdoc.config import RagdocConfig, configure

    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        return ImageSummary(summary="x")

    with configure(RagdocConfig()):
        ImageSummaryProcessor(summarize=mock_summarize)  # must not raise


@pytest.mark.anyio
async def test_image_processor_sets_text_representation_from_structured():
    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        return ImageSummary(summary="A chart.", text_representation="<table/>")

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    result = await ImageSummaryProcessor(summarize=mock_summarize).process(doc)

    assert result.images[0].text_representation == "<table/>"


@pytest.mark.anyio
async def test_image_processor_sets_text_representation_from_summary_fallback():
    """When text_representation is None, summary becomes text_representation."""

    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        return ImageSummary(summary="A photo of a building.")

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    result = await ImageSummaryProcessor(summarize=mock_summarize).process(doc)

    assert result.images[0].text_representation == "A photo of a building."


@pytest.mark.anyio
async def test_image_processor_skips_already_processed():
    calls: list = []

    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        calls.append(image)
        return ImageSummary(summary="x")

    doc = Document(
        elements=[
            Image(image=_tiny_png_base64(), image_type="png", text_representation="already"),
        ]
    )
    await ImageSummaryProcessor(summarize=mock_summarize).process(doc)
    assert calls == []


@pytest.mark.anyio
async def test_image_processor_processes_multiple_images():
    summaries: list[str] = []

    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        s = f"summary_{len(summaries)}"
        summaries.append(s)
        return ImageSummary(summary=s)

    doc = Document(
        elements=[
            Image(image=_tiny_png_base64(), image_type="png"),
            Image(image=_tiny_png_base64(), image_type="png"),
        ]
    )
    await ImageSummaryProcessor(summarize=mock_summarize).process(doc)
    assert len(summaries) == 2
    assert all(img.text_representation is not None for img in doc.images)


@pytest.mark.anyio
async def test_image_processor_context_fn_passed_to_summarize():
    received: list[str | None] = []

    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        received.append(context)
        return ImageSummary(summary="x")

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    processor = ImageSummaryProcessor(
        summarize=mock_summarize,
        context_fn=lambda img, doc: "injected context",
    )
    await processor.process(doc)
    assert received == ["injected context"]


@pytest.mark.anyio
async def test_image_processor_decorative_image_skipped():
    """Decorative images get text_representation=None."""

    async def mock_summarize(image: Image, context: str | None) -> ImageSummary:
        return ImageSummary(summary="A logo.", decorative=True)

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    result = await ImageSummaryProcessor(summarize=mock_summarize).process(doc)

    assert result.images[0].text_representation is None


@pytest.mark.anyio
async def test_image_processor_logs_and_continues_on_error():
    async def bad_summarize(image: Image, context: str | None) -> ImageSummary:
        raise UnidentifiedImageError("bad image")

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    result = await ImageSummaryProcessor(summarize=bad_summarize).process(doc)
    assert result.images[0].text_representation is None


# --- TestOpenaiImageSummarizerFactory ---


@pytest.mark.anyio
async def test_openai_summarizer_calls_beta_parse_and_returns_image_summary():
    client = _make_openai_client()
    parsed = ImageSummary(summary="A graph.", text_representation="mermaid code")
    client.chat.completions.parse = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(parsed=parsed))])
    )

    summarize = openai_image_summarizer(client, model="gpt-4o")
    img = Image(image=_tiny_png_base64(), image_type="png")
    result = await summarize(img, context=None)

    assert result.summary == "A graph."
    assert result.text_representation == "mermaid code"
    client.chat.completions.parse.assert_called_once()


@pytest.mark.anyio
async def test_openai_summarizer_custom_create_messages_called():
    client = _make_openai_client()
    captured: list = []

    async def fake_parse(*args, **kwargs):
        captured.append(kwargs.get("messages", []))
        return MagicMock(choices=[MagicMock(message=MagicMock(parsed=ImageSummary(summary="x")))])

    client.chat.completions.parse = fake_parse

    def my_messages(b64: str, img_type: str, context: str | None) -> list:
        return [{"role": "user", "content": "MY CUSTOM MESSAGES"}]

    summarize = openai_image_summarizer(client, create_messages=my_messages)
    img = Image(image=_tiny_png_base64(), image_type="png")
    await summarize(img, context=None)

    assert captured[0][0]["content"] == "MY CUSTOM MESSAGES"


@pytest.mark.anyio
async def test_openai_summarizer_no_transformations_skips_pil():
    """Passing transformations=[] skips the PIL round-trip."""
    client = _make_openai_client()
    client.chat.completions.parse = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(parsed=ImageSummary(summary="x")))])
    )
    summarize = openai_image_summarizer(client, transformations=[])
    img = Image(image=_tiny_png_base64(), image_type="png")
    result = await summarize(img, context=None)
    assert result.summary == "x"


@pytest.mark.anyio
async def test_openai_summarizer_refusal_raises_refusal_error():
    """message.parsed=None (refusal) must raise LLMRefusalError inside _summarize (bug 6b)."""
    from ragdoc.llm import LLMRefusalError

    client = _make_openai_client()
    message = MagicMock()
    message.parsed = None
    client.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))
    summarize = openai_image_summarizer(client, model="gpt-test", transformations=[])
    image = Image(image=_tiny_png_base64(), image_type="png")
    with pytest.raises(LLMRefusalError, match="no parsed"):
        await summarize(image, None)
    client.chat.completions.parse.assert_awaited_once()  # refusal is deterministic: never retried


@pytest.mark.anyio
async def test_summary_image_refusal_skips_with_warning(caplog):
    """A refusal through the real factory skips the image with a WARNING; document survives."""
    import logging

    client = _make_openai_client()
    message = MagicMock()
    message.parsed = None
    client.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))
    summarize = openai_image_summarizer(client, model="gpt-test", transformations=[])

    image = Image(image=_tiny_png_base64(), image_type="png")
    doc = Document(elements=[image])
    processor = ImageSummaryProcessor(summarize=summarize)
    with caplog.at_level(logging.WARNING):
        result = await processor.process(doc)
    assert result is doc
    assert image.text_representation is None
    assert any("Failed to summarize image" in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_processor_contains_per_image_refusal(caplog):
    """One refused image must not abort the document — skip with a warning (bug 6b)."""
    import logging

    from ragdoc.llm import LLMRefusalError

    async def refusing_summarize(image, context):
        raise LLMRefusalError("LLM returned no parsed ImageSummary (refusal?)")

    image = Image(image=_tiny_png_base64(), image_type="png")
    doc = Document(elements=[image])
    processor = ImageSummaryProcessor(summarize=refusing_summarize)
    with caplog.at_level(logging.WARNING):
        result = await processor.process(doc)
    assert result is doc
    assert image.text_representation is None
    assert any("Failed to summarize image" in r.message for r in caplog.records)


# --- Degrade semantics: one failing image is skipped with a warning; bugs and cancellation propagate ---


def _bad_request_error():
    import httpx
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "https://api.test/v1"))
    return openai.BadRequestError("invalid image", response=response, body=None)


def _rate_limit_error():
    import httpx
    import openai

    response = httpx.Response(429, request=httpx.Request("POST", "https://api.test/v1"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def _parse_ok(summary: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(parsed=ImageSummary(summary=summary)))])


@pytest.mark.anyio
async def test_bad_request_for_one_image_skips_only_that_image(caplog):
    """A 400 for one bad image (non-retryable) must not abort the document."""
    import logging

    pytest.importorskip("openai", reason="llm extra not installed")
    client = _make_openai_client()
    client.chat.completions.parse = AsyncMock(side_effect=[_bad_request_error(), _parse_ok("second image")])
    summarize = openai_image_summarizer(client, model="gpt-test", transformations=[])

    bad = Image(image=_tiny_png_base64(), image_type="png")
    good = Image(image=_tiny_png_base64(), image_type="png")
    doc = Document(elements=[bad, good])
    with caplog.at_level(logging.WARNING):
        result = await ImageSummaryProcessor(summarize=summarize).process(doc)

    assert result is doc
    assert bad.text_representation is None
    assert good.text_representation == "second image"
    warnings = [r for r in caplog.records if "Failed to summarize image" in r.getMessage()]
    assert len(warnings) == 1
    assert bad.id in warnings[0].getMessage()


@pytest.mark.anyio
async def test_exhausted_retries_for_one_image_skip_only_that_image(monkeypatch, caplog):
    """Retries exhausted on a 429 skip the image (after the shared retry policy), not the document."""
    import logging

    import ragdoc.llm

    pytest.importorskip("openai", reason="llm extra not installed")

    async def _no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(ragdoc.llm, "_sleep", _no_sleep)
    client = _make_openai_client()
    client.chat.completions.parse = AsyncMock(
        side_effect=[_rate_limit_error(), _rate_limit_error(), _parse_ok("second image")]
    )
    summarize = openai_image_summarizer(client, model="gpt-test", transformations=[], max_retries=1)

    first = Image(image=_tiny_png_base64(), image_type="png")
    second = Image(image=_tiny_png_base64(), image_type="png")
    with caplog.at_level(logging.WARNING):
        await ImageSummaryProcessor(summarize=summarize).process(Document(elements=[first, second]))

    assert client.chat.completions.parse.await_count == 3  # 2 attempts for the first image, 1 for the second
    assert first.text_representation is None
    assert second.text_representation == "second image"
    assert any("Failed to summarize image" in r.getMessage() for r in caplog.records)


@pytest.mark.anyio
@pytest.mark.parametrize("bug", [TypeError("bug"), AttributeError("bug"), KeyError("bug"), RuntimeError("bug")])
async def test_programming_error_in_summarize_propagates(bug: Exception):
    """Containment is for LLM/image failures only — a bug in a summarize fn must not be swallowed."""

    async def buggy_summarize(image: Image, context: str | None) -> ImageSummary:
        raise bug

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    with pytest.raises(type(bug)):
        await ImageSummaryProcessor(summarize=buggy_summarize).process(doc)


@pytest.mark.anyio
async def test_cancellation_in_summarize_propagates():
    import asyncio

    async def cancelled_summarize(image: Image, context: str | None) -> ImageSummary:
        raise asyncio.CancelledError

    doc = Document(elements=[Image(image=_tiny_png_base64(), image_type="png")])
    with pytest.raises(asyncio.CancelledError):
        await ImageSummaryProcessor(summarize=cancelled_summarize).process(doc)


@pytest.mark.anyio
async def test_openai_summarizer_forwards_timeout_and_max_retries(monkeypatch):
    """The image call site can take a per-request timeout and retry budget through the shared layer."""
    import ragdoc.llm

    pytest.importorskip("openai", reason="llm extra not installed")

    async def _no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(ragdoc.llm, "_sleep", _no_sleep)
    client = _make_openai_client()
    client.chat.completions.parse = AsyncMock(side_effect=[_rate_limit_error()] * 4 + [_parse_ok("x")])
    summarize = openai_image_summarizer(client, transformations=[], request_timeout=7.5, max_retries=4)

    result = await summarize(Image(image=_tiny_png_base64(), image_type="png"), None)

    assert result.summary == "x"
    assert client.chat.completions.parse.await_count == 5
    assert all(call.kwargs["timeout"] == 7.5 for call in client.chat.completions.parse.await_args_list)


@pytest.mark.anyio
async def test_openai_summarizer_default_sends_no_timeout():
    client = _make_openai_client()
    client.chat.completions.parse = AsyncMock(return_value=_parse_ok("x"))
    summarize = openai_image_summarizer(client, transformations=[])
    await summarize(Image(image=_tiny_png_base64(), image_type="png"), None)
    assert "timeout" not in client.chat.completions.parse.await_args.kwargs
