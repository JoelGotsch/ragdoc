"""Unit tests for LLMChunker failure handling (fable-review Phase 0, bug 6a)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragdoc.chunking.llm import LLMChunker
from ragdoc.document import Document, Paragraph


@pytest.mark.anyio
async def test_llmchunker_refusal_raises_value_error():
    """A refusal (message.parsed is None) must raise a clear ValueError, not AttributeError."""
    client = MagicMock()
    message = MagicMock()
    message.parsed = None
    client.beta.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))
    chunker = LLMChunker(client=client, model="test")
    doc = Document(elements=[Paragraph(html_content="<p>hi</p>")])
    with pytest.raises(ValueError, match="no parsed"):
        await chunker.chunk(doc)
