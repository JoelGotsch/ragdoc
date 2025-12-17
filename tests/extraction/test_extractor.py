"""Tests for StructuredExtractor — generic typed extraction + mention envelope.

The LLM client is mocked: ``.chat.completions.parse`` returns a prebuilt ExtractionBatch, so
these tests assert the extractor's envelope/provenance behaviour, not model quality. ``extract()``
is the typed channel: it RETURNS ``Mention`` objects and never touches ``document.metadata``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.structured import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionSettings,
    StructuredExtractor,
    _batch_model,
)


class Event(BaseModel):
    """An event mentioned in the document."""

    title: str = Field(description="Short event title.")
    date: FuzzyDate | None = Field(default=None, description="When the event happened, if stated.")


def make_doc(*, source_id: str | None = None, source_hash: str | None = None, **metadata) -> Document:
    doc = Document(
        title="Report",
        elements=[Heading(html="<h1>Report</h1>"), Paragraph(html="<p>Body text.</p>")],
        metadata=metadata,
    )
    doc.source_id = source_id
    doc.source_hash = source_hash
    return doc


def make_client(events: list[Event]) -> MagicMock:
    """A mock OpenAI client whose parse() returns an ExtractionBatch of *events*."""
    batch = _batch_model(Event)(mentions=events)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=response)
    return client


def extractor(client: MagicMock, **kwargs) -> StructuredExtractor[Event]:
    return StructuredExtractor(Event, client=client, model="test-model", **kwargs)


@pytest.mark.anyio
async def test_extracts_typed_mentions():
    client = make_client([Event(title="Shutdown"), Event(title="Restart")])
    doc = make_doc(source_id="s1")
    mentions = await extractor(client).extract(doc)

    assert all(isinstance(m, Mention) for m in mentions)
    assert [m.payload.title for m in mentions] == ["Shutdown", "Restart"]
    assert all(isinstance(m.payload, Event) for m in mentions)


@pytest.mark.anyio
async def test_provenance_and_metadata_copy():
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1", source_hash="filehash", split_sequence=2, split_total=5, custom="x")
    expected_content_hash = doc.content_hash()

    mentions = await extractor(client).extract(doc)
    m = mentions[0]

    assert m.source_id == "s1"
    assert m.source_hash == "filehash"
    assert m.content_hash == expected_content_hash
    # the locator + custom keys are copied verbatim from the document's metadata
    assert m.metadata == {"split_sequence": 2, "split_total": 5, "custom": "x"}
    # extract() writes nothing back
    assert "mentions" not in doc.metadata


@pytest.mark.anyio
async def test_source_hash_none_when_document_has_none():
    """source_hash is honestly optional — never faked from the content hash."""
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1")  # no source_hash on the document
    mentions = await extractor(client).extract(doc)

    assert mentions[0].source_hash is None
    assert mentions[0].content_hash == doc.content_hash()  # content_hash still real


def test_construction_without_client_fails_loud():
    """Client resolution happens in __init__ (fail-loud at construction), never at extract time."""
    from ragdoc.config import RagdocConfig, configure
    from ragdoc.llm import LLMNotConfiguredError

    with configure(RagdocConfig()), pytest.raises(LLMNotConfiguredError):
        StructuredExtractor(Event)


@pytest.mark.anyio
async def test_document_metadata_untouched_by_extract():
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1", split_sequence=1, custom={"nested": [1, 2]})
    before = copy.deepcopy(doc.metadata)
    await extractor(client).extract(doc)
    assert doc.metadata == before


@pytest.mark.anyio
async def test_ordinal_disambiguates_identical_payloads():
    client = make_client([Event(title="Same"), Event(title="Same")])
    mentions = await extractor(client).extract(make_doc(source_id="s1"))
    ids = [m.mention_id for m in mentions]
    assert ids[0] != ids[1]  # identical payloads, distinct ids via ordinal
    assert [m.ordinal for m in mentions] == [0, 1]


@pytest.mark.anyio
async def test_empty_render_returns_empty_list():
    client = make_client([Event(title="A")])
    empty_doc = Document(title=None, elements=[])
    mentions = await extractor(client).extract(empty_doc)
    client.chat.completions.parse.assert_not_called()
    assert mentions == []
    assert empty_doc.metadata == {}


@pytest.mark.anyio
async def test_below_min_tokens_returns_empty_list():
    client = make_client([Event(title="A")])
    # a huge min_tokens guarantees the short body is below threshold
    doc = make_doc(source_id="s1")
    mentions = await extractor(client, settings=ExtractionSettings(min_tokens=10_000)).extract(doc)
    client.chat.completions.parse.assert_not_called()
    assert mentions == []
    assert "mentions" not in doc.metadata


@pytest.mark.anyio
async def test_request_timeout_forwarded():
    """settings.request_timeout must reach the parse() call (the old asymmetry bug)."""
    client = make_client([Event(title="A")])
    settings = ExtractionSettings(request_timeout=5.0)
    await extractor(client, settings=settings).extract(make_doc(source_id="s1"))
    assert client.chat.completions.parse.call_args.kwargs["timeout"] == 5.0


@pytest.mark.anyio
async def test_no_timeout_kwarg_when_unset():
    client = make_client([Event(title="A")])
    await extractor(client).extract(make_doc(source_id="s1"))
    assert "timeout" not in client.chat.completions.parse.call_args.kwargs


@pytest.mark.anyio
async def test_custom_system_prompt_used():
    client = make_client([Event(title="A")])
    settings = ExtractionSettings(system_prompt="CUSTOM EXTRACTION PROMPT")
    await extractor(client, settings=settings).extract(make_doc(source_id="s1"))

    messages = client.chat.completions.parse.call_args.kwargs["messages"]
    assert messages[0]["content"] == "CUSTOM EXTRACTION PROMPT"
    assert messages[0]["content"] != EXTRACTION_SYSTEM_PROMPT


@pytest.mark.anyio
async def test_default_system_prompt_when_unset():
    client = make_client([Event(title="A")])
    await extractor(client, settings=ExtractionSettings()).extract(make_doc(source_id="s1"))
    messages = client.chat.completions.parse.call_args.kwargs["messages"]
    assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT


@pytest.mark.anyio
async def test_retry_then_reraise_on_exhaustion(monkeypatch):
    """Retryable transport errors (5xx) are retried; exhaustion re-raises the openai error."""
    import httpx
    import openai

    import ragdoc.llm

    async def _no_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(ragdoc.llm, "_sleep", _no_sleep)
    error = openai.InternalServerError(
        "api down", response=httpx.Response(500, request=httpx.Request("POST", "https://api.test/v1")), body=None
    )
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=error)
    settings = ExtractionSettings(max_retries=1)
    with pytest.raises(openai.InternalServerError, match="api down"):
        await extractor(client, settings=settings).extract(make_doc(source_id="s1"))
    assert client.chat.completions.parse.call_count == 2  # 1 attempt + 1 retry


@pytest.mark.anyio
async def test_non_retryable_error_raises_immediately():
    """Deterministic errors (non-openai / 4xx) are not retried under the shared LLM policy."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=RuntimeError("api down"))
    settings = ExtractionSettings(max_retries=3)
    with pytest.raises(RuntimeError, match="api down"):
        await extractor(client, settings=settings).extract(make_doc(source_id="s1"))
    assert client.chat.completions.parse.call_count == 1
