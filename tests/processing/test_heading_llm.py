"""Tests for LLMHeadingResolver."""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pydantic_settings", reason="pydantic_settings not installed")
from ragdoc.document import Document, Heading
from ragdoc.processing.heading_llm import (
    HeadingJudgment,
    HeadingLevel,
    HeadingResponse,
    LLMHeadingResolver,
    LLMHeadingResolverSettings,
)


def _make_mock_client(response: HeadingResponse | None = None, *, side_effect: Exception | None = None) -> MagicMock:
    """Create a mock client whose chat.completions.parse returns *response*."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = response
    mock_response.choices[0].message.refusal = None

    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.parse = AsyncMock(side_effect=side_effect)
    else:
        client.chat.completions.parse = AsyncMock(return_value=mock_response)
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
    client.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_heading_resolver_process_no_headings():
    """Processor handles document with no headings."""
    from ragdoc.document import Paragraph

    doc = Document(elements=[Paragraph(html="<p>Content</p>")])
    client = _make_mock_client()
    resolver = LLMHeadingResolver(client)
    result = await resolver.process(doc)
    assert result is doc
    client.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_heading_resolver_process_with_headings():
    """Processor calls LLM for headings and applies judgments."""
    doc = Document(
        elements=[
            Heading(html="<h1>Title</h1>"),
            Heading(html="<h1>Section</h1>"),
        ]
    )

    response = HeadingResponse(
        judgments=[
            HeadingJudgment(id=1, level=HeadingLevel.DOCUMENT_TITLE),
            HeadingJudgment(id=2, level=HeadingLevel.H1),
        ]
    )
    client = _make_mock_client(response)
    # Keep title in elements so we can inspect its metadata
    resolver = LLMHeadingResolver(client, remove_title_from_elements=False)
    result = await resolver.process(doc)

    client.chat.completions.parse.assert_called_once()

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
            Heading(html="<h1>12-March-2024</h1>"),  # Metadata
            Heading(html="<h1>Real Title</h1>"),
        ]
    )

    response = HeadingResponse(
        judgments=[
            HeadingJudgment(id=1, level=HeadingLevel.NOT_HEADING),
            HeadingJudgment(id=2, level=HeadingLevel.H1),
        ]
    )
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
            Heading(html="<h2>Ambiguous Heading</h2>"),
            Heading(html="<h1>Clear Section</h1>"),
        ]
    )

    # Heading 1 omitted from judgments → uncertain; heading 2 present → assign h1
    response = HeadingResponse(
        judgments=[
            HeadingJudgment(id=2, level=HeadingLevel.H1),
        ]
    )
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
            Heading(html="<h1>Title</h1>"),
        ]
    )

    client = _make_mock_client(side_effect=Exception("API Error"))
    settings = LLMHeadingResolverSettings(max_retries=1)
    resolver = LLMHeadingResolver(client, settings=settings)
    result = await resolver.process(doc)

    assert result is doc


@pytest.mark.anyio
async def test_heading_resolver_resolves_levels_on_titled_document():
    """A pre-set title (e.g. HTML <title>, MinerU document_title) must not skip level resolution."""
    doc = Document(
        title="Pre-set Title",
        elements=[
            Heading(html="<h3>Section</h3>"),
            Heading(html="<h5>Subsection</h5>"),
        ],
    )
    response = HeadingResponse(
        judgments=[
            HeadingJudgment(id=1, level=HeadingLevel.H1),
            HeadingJudgment(id=2, level=HeadingLevel.H2),
        ]
    )
    client = _make_mock_client(response)
    resolver = LLMHeadingResolver(client)
    result = await resolver.process(doc)

    client.chat.completions.parse.assert_called_once()  # resolution ran despite existing title
    assert result.elements[0].level == 1
    assert result.elements[1].level == 2


@pytest.mark.anyio
async def test_heading_resolver_does_not_overwrite_existing_title():
    """DOCUMENT_TITLE judgment on a titled document: keep the title, skip element removal."""
    doc = Document(
        title="Existing Title",
        elements=[
            Heading(html="<h1>LLM-Detected Title</h1>"),
            Heading(html="<h2>Section</h2>"),
        ],
    )
    response = HeadingResponse(
        judgments=[
            HeadingJudgment(id=1, level=HeadingLevel.DOCUMENT_TITLE),
            HeadingJudgment(id=2, level=HeadingLevel.H1),
        ]
    )
    client = _make_mock_client(response)
    resolver = LLMHeadingResolver(client, remove_title_from_elements=True, remove_elements_before_title=True)
    result = await resolver.process(doc)

    assert result.title == "Existing Title"  # never overwritten
    # No destructive title-element handling on an already-titled document.
    assert len(result.elements) == 2
    assert "is_document_title" not in result.elements[0].metadata
    assert result.elements[1].level == 1  # level resolution still applied


@pytest.mark.anyio
async def test_heading_resolver_sets_title_when_absent():
    """First run on an untitled document still detects and applies the title (with removal)."""
    doc = Document(
        elements=[
            Heading(html="<h1>The Title</h1>"),
            Heading(html="<h2>Section</h2>"),
        ]
    )
    response = HeadingResponse(
        judgments=[
            HeadingJudgment(id=1, level=HeadingLevel.DOCUMENT_TITLE),
            HeadingJudgment(id=2, level=HeadingLevel.H1),
        ]
    )
    client = _make_mock_client(response)
    resolver = LLMHeadingResolver(client, remove_title_from_elements=True)
    result = await resolver.process(doc)

    assert result.title == "The Title"
    assert len(result.elements) == 1  # title element removed
    assert result.elements[0].level == 1


@pytest.mark.anyio
async def test_heading_resolver_degrades_to_empty_on_exhaustion():
    """Behavior pin: a non-retryable 400 degrades to [] (headings unchanged) after ONE call."""
    import httpx
    import openai

    doc = Document(elements=[Heading(html="<h3>Title</h3>")])
    error = openai.BadRequestError(
        "bad request",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.test/v1")),
        body=None,
    )
    client = _make_mock_client(side_effect=error)
    settings = LLMHeadingResolverSettings(max_retries=3)
    resolver = LLMHeadingResolver(client, settings=settings)

    result = await resolver.process(doc)

    assert result is doc
    assert result.elements[0].level == 3  # unchanged — degrade to no judgments
    client.chat.completions.parse.assert_called_once()  # 400 is not retried
