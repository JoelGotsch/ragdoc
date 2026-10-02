"""Static-typing fixture (not a test module): real OpenAI SDK clients vs ragdoc's protocols.

``tests/test_llm.py::test_real_async_openai_type_checks_against_protocols`` runs basedpyright on
this file. Every assignment below must type-check; a failure means a protocol in ``ragdoc.llm``
has drifted from the SDK's real signatures (e.g. a parameter typed ``int | None`` where the SDK
only accepts an omittable ``int``).
"""

from __future__ import annotations

from openai import AsyncAzureOpenAI, AsyncOpenAI

from ragdoc.chunking import LLMChunker
from ragdoc.config import RagdocConfig
from ragdoc.llm import ChatClient, EmbeddingsClient, LLMClient, without_sdk_retries
from ragdoc.pipeline.embedders import OpenAIEmbedder


def check_openai(client: AsyncOpenAI) -> tuple[ChatClient, EmbeddingsClient, LLMClient]:
    chat: ChatClient = client
    embeddings: EmbeddingsClient = client
    both: LLMClient = client
    return chat, embeddings, both


def check_azure(client: AsyncAzureOpenAI) -> tuple[ChatClient, EmbeddingsClient, LLMClient]:
    chat: ChatClient = client
    embeddings: EmbeddingsClient = client
    both: LLMClient = client
    return chat, embeddings, both


def check_retry_free_view(client: AsyncOpenAI) -> AsyncOpenAI:
    return without_sdk_retries(client)


def check_consumers(client: AsyncOpenAI) -> tuple[RagdocConfig, LLMChunker, OpenAIEmbedder]:
    return RagdocConfig(openai_client=client), LLMChunker(client=client), OpenAIEmbedder(client)
