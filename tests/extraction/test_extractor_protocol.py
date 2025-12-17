"""Tests for the Extractor protocol and the as_processor adapter (Phase 4, D3-A).

The protocol is the typed extraction channel: ``extract()`` returns ``Mention`` objects and never
touches document metadata. ``as_processor`` is the explicit, opt-in escape hatch that dumps
mentions into ``document.metadata`` for DocumentStore-style consumers.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.extractor import Extractor, as_processor
from ragdoc.extraction.kg import KnowledgeGraphExtractor
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.extraction.schema import EdgeRef, GraphSchema
from ragdoc.extraction.structured import StructuredExtractor
from ragdoc.pipeline.linear import DocumentPipeline
from ragdoc.processing.base import DocumentProcessor

from .conftest import Event, MemoryMentionStore, make_event_client, make_extractor


def make_doc(**metadata) -> Document:
    return Document(
        title="Report",
        elements=[Heading(html="<h1>Report</h1>"), Paragraph(html="<p>Body text.</p>")],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 1-3. Protocol conformance
# ---------------------------------------------------------------------------


def test_structured_extractor_satisfies_protocol():
    extractor = StructuredExtractor(Event, client=MagicMock(), model="m")
    assert isinstance(extractor, Extractor)


def test_kg_extractor_satisfies_protocol():
    class Person(BaseModel):
        kind: Literal["Person"] = "Person"
        full_name: str

    class Company(BaseModel):
        kind: Literal["Company"] = "Company"
        name: str

    class Employment(BaseModel):
        kind: Literal["Employment"] = "Employment"
        refs: EdgeRef

    schema = GraphSchema(
        node_types=(Person, Company),
        edge_types=(Employment,),
        patterns=((Person, Employment, Company),),
    )
    extractor = KnowledgeGraphExtractor(schema, client=MagicMock(), model="m")
    assert isinstance(extractor, Extractor)


def test_pipeline_rejects_non_extractor():
    with pytest.raises(TypeError, match="Extractor protocol"):
        MentionStorePipeline(
            pipeline=DocumentPipeline(),
            extractor=object(),  # type: ignore[arg-type]
            mention_store=MemoryMentionStore(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 4-7. as_processor adapter
# ---------------------------------------------------------------------------


def test_as_processor_returns_document_processor():
    adapter = as_processor(make_extractor([Event(title="A")]))
    assert isinstance(adapter, DocumentProcessor)


@pytest.mark.anyio
async def test_as_processor_writes_metadata_key():
    adapter = as_processor(make_extractor([Event(title="Shutdown"), Event(title="Restart")]))
    doc = make_doc()
    out = await adapter.process(doc)
    assert out is not None
    titles = [m["payload"]["title"] for m in out.metadata["mentions"]]
    assert titles == ["Shutdown", "Restart"]


@pytest.mark.anyio
async def test_as_processor_custom_metadata_key():
    adapter = as_processor(make_extractor([Event(title="A")]), metadata_key="my_mentions")
    out = await adapter.process(make_doc())
    assert out is not None
    assert [m["payload"]["title"] for m in out.metadata["my_mentions"]] == ["A"]
    assert "mentions" not in out.metadata


@pytest.mark.anyio
async def test_as_processor_idempotency_guard():
    client = make_event_client([[Event(title="Fresh")]])
    adapter = as_processor(StructuredExtractor(Event, client=client, model="m"))
    doc = make_doc()
    doc.metadata["mentions"] = [{"already": "here"}]
    out = await adapter.process(doc)
    assert out is not None
    client.beta.chat.completions.parse.assert_not_called()
    assert out.metadata["mentions"] == [{"already": "here"}]


@pytest.mark.anyio
async def test_as_processor_overwrite_strips_stale_key_before_extract():
    """With overwrite=True, a mention's metadata locator must never embed the prior list."""
    client = make_event_client([[Event(title="Fresh")]])
    adapter = as_processor(StructuredExtractor(Event, client=client, model="m"), overwrite=True)
    doc = make_doc(custom="keep")
    doc.metadata["mentions"] = [{"stale": "data"}]
    out = await adapter.process(doc)
    assert out is not None
    client.beta.chat.completions.parse.assert_called_once()
    mentions = out.metadata["mentions"]
    assert [m["payload"]["title"] for m in mentions] == ["Fresh"]
    for m in mentions:
        assert "mentions" not in m["metadata"]
        assert m["metadata"]["custom"] == "keep"


@pytest.mark.anyio
async def test_as_processor_writes_empty_list_when_no_mentions():
    """The key is always present after processing — the dump consumers' contract."""
    adapter = as_processor(make_extractor([]))
    out = await adapter.process(make_doc())
    assert out is not None
    assert out.metadata["mentions"] == []


# ---------------------------------------------------------------------------
# 34. Dedup guard — no duplicated LLM plumbing left in the extractor modules
# ---------------------------------------------------------------------------


def test_no_duplicate_llm_plumbing():
    """The resolve_*/retry plumbing lives once in extraction/_llm.py, not per extractor."""
    from pathlib import Path

    import ragdoc.extraction.kg
    import ragdoc.extraction.structured

    for module in (ragdoc.extraction.structured, ragdoc.extraction.kg):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for needle in ("def _get_client", "def _get_model", "def _get_renderer", "def _get_tokenizer"):
            assert needle not in source, f"{module.__name__} still defines {needle}"
        # The call syntax (not docstring mentions) must not be inlined outside _llm.py.
        assert "completions.parse(" not in source, f"{module.__name__} inlines the LLM call"
