"""Unit tests for LLMChunker: failure handling, provenance, and per-chunk isolation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragdoc.chunking.llm import DocumentTopicSummaries, LLMChunker
from ragdoc.document import Document, Paragraph


def _mock_client(summaries: list[str] | None) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.parsed = DocumentTopicSummaries(summaries=summaries) if summaries is not None else None
    client.beta.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))
    return client


@pytest.mark.anyio
async def test_llmchunker_refusal_raises_value_error():
    """A refusal (message.parsed is None) must raise a clear ValueError, not AttributeError."""
    chunker = LLMChunker(client=_mock_client(None), model="test")
    doc = Document(elements=[Paragraph(html="<p>hi</p>")])
    with pytest.raises(ValueError, match="no parsed"):
        await chunker.chunk(doc)


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
