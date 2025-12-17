"""Tests for StructuredExtractionProcessor — generic typed extraction + mention envelope.

The LLM client is mocked: ``.beta.chat.completions.parse`` returns a prebuilt ExtractionBatch, so
these tests assert the processor's wrapping/provenance behaviour, not model quality.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.processor import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionSettings,
    StructuredExtractionProcessor,
    _batch_model,
)


class Event(BaseModel):
    """An event mentioned in the document."""

    title: str = Field(description="Short event title.")
    date: FuzzyDate | None = Field(default=None, description="When the event happened, if stated.")


def make_doc(*, source_id: str | None = None, source_hash: str | None = None, **metadata) -> Document:
    doc = Document(
        title="Report",
        elements=[Heading(innerhtml="Report", level=1), Paragraph(html_content="<p>Body text.</p>")],
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
    client.beta.chat.completions.parse = AsyncMock(return_value=response)
    return client


def processor(client: MagicMock, **kwargs) -> StructuredExtractionProcessor[Event]:
    return StructuredExtractionProcessor(Event, client=client, model="test-model", **kwargs)


@pytest.mark.anyio
async def test_extracts_typed_mentions():
    client = make_client([Event(title="Shutdown"), Event(title="Restart")])
    doc = make_doc(source_id="s1")
    out = await processor(client).process(doc)

    mentions = out.metadata["mentions"]
    assert [m["payload"]["title"] for m in mentions] == ["Shutdown", "Restart"]
    # payloads validate back into the user model
    assert [Event.model_validate(m["payload"]).title for m in mentions] == ["Shutdown", "Restart"]


@pytest.mark.anyio
async def test_provenance_and_metadata_copy():
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1", source_hash="filehash", split_sequence=2, split_total=5, custom="x")
    expected_content_hash = doc.content_hash()

    out = await processor(client).process(doc)
    m = out.metadata["mentions"][0]

    assert m["source_id"] == "s1"
    assert m["source_hash"] == "filehash"
    assert m["content_hash"] == expected_content_hash
    # the locator + custom keys are copied verbatim from the document's metadata
    assert m["metadata"] == {"split_sequence": 2, "split_total": 5, "custom": "x"}


@pytest.mark.anyio
async def test_metadata_copy_excludes_mentions_key():
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1", split_sequence=1)
    out = await processor(client).process(doc)
    assert "mentions" not in out.metadata["mentions"][0]["metadata"]


@pytest.mark.anyio
async def test_ordinal_disambiguates_identical_payloads():
    client = make_client([Event(title="Same"), Event(title="Same")])
    out = await processor(client).process(make_doc(source_id="s1"))
    ids = [m["mention_id"] for m in out.metadata["mentions"]]
    assert ids[0] != ids[1]  # identical payloads, distinct ids via ordinal
    assert [m["ordinal"] for m in out.metadata["mentions"]] == [0, 1]


@pytest.mark.anyio
async def test_idempotency_guard_skips_when_already_set():
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1")
    doc.metadata["mentions"] = [{"already": "here"}]
    out = await processor(client).process(doc)
    client.beta.chat.completions.parse.assert_not_called()
    assert out.metadata["mentions"] == [{"already": "here"}]


@pytest.mark.anyio
async def test_overwrite_reextracts():
    client = make_client([Event(title="Fresh")])
    doc = make_doc(source_id="s1")
    doc.metadata["mentions"] = [{"stale": "data"}]
    out = await processor(client, overwrite=True).process(doc)
    client.beta.chat.completions.parse.assert_called_once()
    assert out.metadata["mentions"][0]["payload"]["title"] == "Fresh"


@pytest.mark.anyio
async def test_metadata_discipline_preserves_other_keys():
    client = make_client([Event(title="A")])
    doc = make_doc(source_id="s1", existing="keep-me")
    out = await processor(client).process(doc)
    assert out.metadata["existing"] == "keep-me"
    assert set(out.metadata) == {"existing", "mentions"}


@pytest.mark.anyio
async def test_prompt_override_flows_into_request():
    client = make_client([Event(title="A")])
    settings = ExtractionSettings(system_prompt="CUSTOM EXTRACTION PROMPT")
    await processor(client, settings=settings).process(make_doc(source_id="s1"))

    messages = client.beta.chat.completions.parse.call_args.kwargs["messages"]
    assert messages[0]["content"] == "CUSTOM EXTRACTION PROMPT"
    assert messages[0]["content"] != EXTRACTION_SYSTEM_PROMPT


@pytest.mark.anyio
async def test_min_tokens_skips_short_split():
    client = make_client([Event(title="A")])
    # a huge min_tokens guarantees the short body is below threshold
    out = await processor(client, settings=ExtractionSettings(min_tokens=10_000)).process(make_doc(source_id="s1"))
    client.beta.chat.completions.parse.assert_not_called()
    assert out.metadata["mentions"] == []
