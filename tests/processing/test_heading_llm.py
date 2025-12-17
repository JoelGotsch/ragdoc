"""Tests for LLMHeadingResolver."""
import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("pydantic_settings", reason="pydantic_settings not installed")
from ragdoc.processing.heading_llm import (
    HeadingJudgment,
    HeadingLevel,
    HeadingResponse,
    LLMHeadingResolver,
    LLMHeadingResolverSettings,
)

from ragdoc.document import Document, Heading


def _make_mock_client(response: HeadingResponse | None = None, *, side_effect: Exception | None = None) -> MagicMock:
    """Create a mock client whose beta.chat.completions.parse returns *response*."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = response
    mock_response.choices[0].message.refusal = None

    client = MagicMock()
    if side_effect is not None:
        client.beta.chat.completions.parse = AsyncMock(side_effect=side_effect)
    else:
        client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
    return client


# --- TestLLMHeadingResolver ---


@pytest.mark.anyio
async def test_heading_resolver_process_empty_document():
    """Processor handles empty document."""
    doc = Document(elements=[])
    client = _make_mock_client()
    resolver = LLMHeadingResolver(client)
    result = await resolver.process(doc)
    assert result is doc
    client.beta.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_heading_resolver_process_no_headings():
    """Processor handles document with no headings."""
    from ragdoc.document import Paragraph
    doc = Document(elements=[Paragraph(html="<p>Content</p>")])
    client = _make_mock_client()
    resolver = LLMHeadingResolver(client)
    result = await resolver.process(doc)
    assert result is doc
    client.beta.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_heading_resolver_process_with_headings():
    """Processor calls LLM for headings and applies judgments."""
    doc = Document(
        elements=[
            Heading(innerhtml="Title", level=1),
            Heading(innerhtml="Section", level=1),
        ]
    )

    response = HeadingResponse(judgments=[
        HeadingJudgment(id=1, level=HeadingLevel.DOCUMENT_TITLE),
        HeadingJudgment(id=2, level=HeadingLevel.H1),
    ])
    client = _make_mock_client(response)
    # Keep title in elements so we can inspect its metadata
    resolver = LLMHeadingResolver(client, remove_title_from_elements=False)
    result = await resolver.process(doc)

    client.beta.chat.completions.parse.assert_called_once()

    assert result.elements[0].metadata.get("llm_heading_level") == "document-title"
    assert result.elements[0].metadata.get("is_document_title") is True
    assert result.title == "Title"
    assert result.elements[1].level == 1
    assert result.elements[1].metadata.get("llm_heading_level") == "h1"


@pytest.mark.anyio
async def test_heading_resolver_handles_none_response():
    """Processor handles LLM marking headings as 'none' — converts them to Paragraphs."""
    from ragdoc.document import Paragraph
    doc = Document(
        elements=[
            Heading(innerhtml="12-March-2024", level=1),  # Metadata
            Heading(innerhtml="Real Title", level=1),
        ]
    )

    response = HeadingResponse(judgments=[
        HeadingJudgment(id=1, level=HeadingLevel.NOT_HEADING),
        HeadingJudgment(id=2, level=HeadingLevel.H1),
    ])
    client = _make_mock_client(response)
    resolver = LLMHeadingResolver(client)
    result = await resolver.process(doc)

    # Element 0 converted to Paragraph; metadata signals it was not a real heading.
    assert isinstance(result.elements[0], Paragraph)
    assert result.elements[0].metadata.get("llm_heading_level") == "none"
    assert result.elements[1].level == 1


@pytest.mark.anyio
async def test_heading_resolver_handles_uncertain_response():
    """Processor leaves heading level unchanged when LLM omits the heading (uncertain)."""
    doc = Document(
        elements=[
            Heading(innerhtml="Ambiguous Heading", level=2),
            Heading(innerhtml="Clear Section", level=1),
        ]
    )

    # Heading 1 omitted from judgments → uncertain; heading 2 present → assign h1
    response = HeadingResponse(judgments=[
        HeadingJudgment(id=2, level=HeadingLevel.H1),
    ])
    client = _make_mock_client(response)
    resolver = LLMHeadingResolver(client)
    result = await resolver.process(doc)

    assert result.elements[0].level == 2  # Unchanged — LLM was uncertain (absent)
    assert "llm_heading_level" not in result.elements[0].metadata
    assert result.elements[1].level == 1


@pytest.mark.anyio
async def test_heading_resolver_handles_api_failure():
    """Processor handles LLM API failures gracefully."""
    doc = Document(
        elements=[
            Heading(innerhtml="Title", level=1),
        ]
    )

    client = _make_mock_client(side_effect=Exception("API Error"))
    settings = LLMHeadingResolverSettings(max_retries=1)
    resolver = LLMHeadingResolver(client, settings=settings)
    result = await resolver.process(doc)

    assert result is doc
