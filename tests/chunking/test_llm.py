"""Unit tests for LLMChunker: failure handling, provenance, and per-chunk isolation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragdoc.chunking.llm import DocumentTopicSummaries, LLMChunker
from ragdoc.document import Document, Paragraph


def _mock_client(summaries: list[str] | None) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.parsed = DocumentTopicSummaries(summaries=summaries) if summaries is not None else None
    client.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))
    return client


@pytest.mark.anyio
async def test_llmchunker_refusal_raises():
    """A refusal (message.parsed is None) must raise LLMRefusalError, not AttributeError."""
    from ragdoc.llm import LLMRefusalError

    client = _mock_client(None)
    chunker = LLMChunker(client=client, model="test")
    doc = Document(elements=[Paragraph(html="<p>hi</p>")])
    with pytest.raises(LLMRefusalError, match="no parsed"):
        await chunker.chunk(doc)
    client.chat.completions.parse.assert_awaited_once()  # refusals are never retried


def test_llmchunker_init_fails_without_client():
    """No explicit client and no configured client -> fail-loud at __init__, not chunk time."""
    from ragdoc.config import RagdocConfig, configure
    from ragdoc.llm import LLMNotConfiguredError

    with configure(RagdocConfig()), pytest.raises(LLMNotConfiguredError):
        LLMChunker(model="test")


@pytest.mark.anyio
async def test_llmchunker_max_prompt_tokens_truncates_with_warning(caplog):
    """Over-budget documents: the LLM sees <= max_prompt_tokens tokens, chunks keep full text."""
    import logging

    from ragdoc.utils import GPTTokenizer

    client = _mock_client(["topic a"])
    tokenizer = GPTTokenizer()
    chunker = LLMChunker(client=client, model="test", max_prompt_tokens=5, tokenizer=tokenizer)
    long_text = "many different words appear in this deliberately overlong paragraph body"
    doc = Document(elements=[Paragraph(html=f"<p>{long_text}</p>")])

    with caplog.at_level(logging.WARNING):
        chunks = await chunker.chunk(doc)

    # The LLM input was truncated to the budget...
    messages = client.chat.completions.parse.call_args.kwargs["messages"]
    llm_text = messages[1]["content"][1]["text"]
    full_rendered = chunks[0].prompt_content
    assert llm_text != f"*Document to analyse:*\n\n{full_rendered}"
    sent_document_part = llm_text.removeprefix("*Document to analyse:*\n\n")
    assert tokenizer.count(sent_document_part) <= 5
    # ...but the emitted chunks keep the full rendered text.
    assert long_text in full_rendered
    assert any("max_prompt_tokens" in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_llmchunker_under_budget_not_truncated(caplog):
    """A document within max_prompt_tokens passes through untouched, no warning."""
    import logging

    client = _mock_client(["topic a"])
    chunker = LLMChunker(client=client, model="test", max_prompt_tokens=10_000)
    doc = Document(elements=[Paragraph(html="<p>short</p>")])

    with caplog.at_level(logging.WARNING):
        chunks = await chunker.chunk(doc)

    messages = client.chat.completions.parse.call_args.kwargs["messages"]
    llm_text = messages[1]["content"][1]["text"]
    assert llm_text == f"*Document to analyse:*\n\n{chunks[0].prompt_content}"
    assert not any("max_prompt_tokens" in r.message for r in caplog.records)


@pytest.mark.anyio
async def test_llmchunker_source_hash_none_when_document_has_none():
    """No file hash -> source_hash None on every chunk (no content-hash masquerade)."""
    chunker = LLMChunker(client=_mock_client(["topic a", "topic b"]), model="test")
    doc = Document(elements=[Paragraph(html="<p>hi</p>")])
    chunks = await chunker.chunk(doc)
    assert len(chunks) == 2
    assert all(c.source_hash is None for c in chunks)
    assert all(c.content_hash == doc.content_hash() for c in chunks)


@pytest.mark.anyio
async def test_llmchunker_chunks_do_not_share_metadata_dict():
    chunker = LLMChunker(client=_mock_client(["topic a", "topic b"]), model="test")
    doc = Document(elements=[Paragraph(html="<p>hi</p>")], metadata={"k": "v"})
    chunks = await chunker.chunk(doc)
    chunks[0].metadata["only-here"] = True
    assert "only-here" not in chunks[1].metadata
    assert "only-here" not in doc.metadata


@pytest.mark.anyio
async def test_llm_chunker_multi_chunk_ordinals_distinct():
    """Through chunk_document, an N-summary response yields N distinct pipeline-minted ids."""
    from ragdoc.pipeline import DocumentPipeline

    chunker = LLMChunker(client=_mock_client(["t1", "t2", "t3"]), model="test")
    doc = Document(elements=[Paragraph(html="<p>hi</p>")])
    doc.source_id = "src"
    chunks = await DocumentPipeline(chunker=chunker).chunk_document(doc)
    assert len(chunks) == 3
    assert len({c.id for c in chunks}) == 3
    assert {c.source_id for c in chunks} == {"src"}
