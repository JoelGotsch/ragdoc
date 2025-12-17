"""Embedder protocol and helpers for VectorStorePipeline.

Usage::

    from ragdoc.pipeline import VectorStorePipeline, EmbedderConfig, prompt_content_text

    vs = VectorStorePipeline(
        pipeline=doc_pipeline,
        vector_store=qdrant_store,
        embedders={
            # Dense: embed the focused semantic summary
            "dense": EmbedderConfig(my_dense_embedder),
            # Sparse: embed the full document text for keyword recall
            "sparse": EmbedderConfig(my_sparse_embedder, text_fn=prompt_content_text),
        },
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ragdoc.llm import EmbeddingResponse, EmbeddingsClient, retry_llm

if TYPE_CHECKING:
    from ragdoc.chunking.chunk import Chunk


@runtime_checkable
class Embedder(Protocol):
    """Protocol for async embedding models."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* and return one vector per input, in order."""
        ...


class OpenAIEmbedder:
    """Concrete :class:`Embedder` backed by an OpenAI-compatible embeddings endpoint.

    Requests run under the library-wide LLM retry policy (:func:`ragdoc.llm.retry_llm`):
    rate limits, connection/timeout errors, and 5xx are retried with backoff; other errors
    raise immediately.

    Args:
        client: An ``AsyncOpenAI``-compatible client — anything satisfying
            :class:`ragdoc.llm.EmbeddingsClient` (OpenAI, Azure OpenAI, or a compatible
            gateway). The caller owns its lifecycle.
        model: Embedding model name (e.g. ``"text-embedding-3-small"``).
        dimensions: Optional output dimensionality for models that support truncation
            (``text-embedding-3-*``). ``None`` uses the model's default size.
        timeout: Per-request timeout in seconds. ``None`` uses the client default.
        max_retries: Maximum additional attempts on retryable transport failures.

    Example:
        ```python
        from openai import AsyncOpenAI
        from ragdoc.pipeline import EmbedderConfig, OpenAIEmbedder

        embedder = OpenAIEmbedder(AsyncOpenAI(), model="text-embedding-3-small")
        config = EmbedderConfig(embedder)   # use in VectorStorePipeline(embedders={...})
        ```
    """

    def __init__(
        self,
        client: EmbeddingsClient,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout
        self._max_retries = max_retries

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in one request; returns one vector per input, in input order."""
        if not texts:
            return []

        async def _create() -> EmbeddingResponse:
            # dimensions/timeout are forwarded only when set — the OpenAI SDK treats an
            # explicit None differently from an omitted parameter.
            if self._dimensions is not None and self._timeout is not None:
                return await self._client.embeddings.create(
                    model=self._model, input=texts, dimensions=self._dimensions, timeout=self._timeout
                )
            if self._dimensions is not None:
                return await self._client.embeddings.create(model=self._model, input=texts, dimensions=self._dimensions)
            if self._timeout is not None:
                return await self._client.embeddings.create(model=self._model, input=texts, timeout=self._timeout)
            return await self._client.embeddings.create(model=self._model, input=texts)

        response = await retry_llm(_create, max_retries=self._max_retries, log_prefix="OpenAIEmbedder")
        # OpenAI returns embeddings keyed by input index — sort to guarantee input order.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


@dataclass
class EmbedderConfig:
    """Pairs an :class:`Embedder` with the text-extraction strategy for a chunk.

    Args:
        embedder: The embedding model to use.
        text_fn:  Callable that extracts the text to embed from a :class:`~ragdoc.chunking.Chunk`.
                  Defaults to :func:`embedding_content_text` (the focused semantic content).
    """

    embedder: Embedder
    text_fn: Callable[[Chunk], str] = field(default=lambda chunk: chunk.embedding_content)


def embedding_content_text(chunk: Chunk) -> str:
    """Extract ``embedding_content`` from a chunk.

    This is the default text for dense embedders.  ``embedding_content`` is produced
    by the chunker specifically as a focused semantic representation:

    * :class:`~ragdoc.chunking.LLMChunker` — a topic summary generated by an LLM.
    * :class:`~ragdoc.chunking.SimpleChunker` — the rendered document text.

    Dense models benefit from this focused signal.
    """
    return chunk.embedding_content


def prompt_content_text(chunk: Chunk) -> str:
    """Extract ``prompt_content`` from a chunk.

    Use this as ``text_fn`` for sparse or keyword-based embedders (e.g. BM25, SPLADE).
    ``prompt_content`` is the full, uncompressed document text.  Sparse/keyword models
    rely on exact term overlap and benefit from the broader vocabulary in the full text,
    whereas dense models are better served by the focused ``embedding_content``.
    """
    return chunk.prompt_content
