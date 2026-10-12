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

from ragdoc.llm import EmbeddingResponse, EmbeddingsClient, retry_llm, without_sdk_retries

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

    Texts are sent in consecutive requests of at most ``batch_size`` inputs (the embeddings
    API caps inputs per request — OpenAI: 2048 inputs and 300,000 tokens summed over the
    request), one request at a time, and the vectors are returned in input order. Each request
    runs under the library-wide LLM retry policy (:func:`ragdoc.llm.retry_llm`): rate limits,
    connection/timeout errors, and 5xx are retried with backoff; other errors raise immediately.
    A real ``AsyncOpenAI`` is called with its SDK-level retries disabled
    (:func:`ragdoc.llm.without_sdk_retries`), so ``max_retries`` here is the only retry layer.
    A failing batch fails the whole :meth:`embed` call — no partial result is returned.

    Args:
        client: An ``AsyncOpenAI``-compatible client — anything satisfying
            :class:`ragdoc.llm.EmbeddingsClient` (OpenAI, Azure OpenAI, or a compatible
            gateway). The caller owns its lifecycle.
        model: Embedding model name (e.g. ``"text-embedding-3-small"``).
        dimensions: Optional output dimensionality for models that support truncation
            (``text-embedding-3-*``). ``None`` uses the model's default size.
        timeout: Per-request timeout in seconds. ``None`` uses the client default.
        max_retries: Maximum additional attempts per request on retryable transport failures.
        batch_size: Maximum number of texts per request (default ``256``). The batch is counted
            in *inputs*, not tokens: lower it when individual texts are long enough that
            ``batch_size`` of them could exceed the provider's per-request token limit.

    Raises:
        ValueError: ``batch_size`` is not positive; or (from :meth:`embed`) the endpoint returned
            a different number of vectors than inputs.

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
        batch_size: int = 256,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches of ``batch_size``; returns one vector per input, in input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One embeddings request (with retries) for a batch that fits the per-request limit."""
        embeddings = without_sdk_retries(self._client).embeddings

        async def _create() -> EmbeddingResponse:
            # dimensions/timeout are forwarded only when set — the OpenAI SDK treats an
            # explicit None differently from an omitted parameter.
            if self._dimensions is not None and self._timeout is not None:
                return await embeddings.create(
                    model=self._model, input=texts, dimensions=self._dimensions, timeout=self._timeout
                )
            if self._dimensions is not None:
                return await embeddings.create(model=self._model, input=texts, dimensions=self._dimensions)
            if self._timeout is not None:
                return await embeddings.create(model=self._model, input=texts, timeout=self._timeout)
            return await embeddings.create(model=self._model, input=texts)

        response = await retry_llm(_create, max_retries=self._max_retries, log_prefix="OpenAIEmbedder")
        if len(response.data) != len(texts):
            raise ValueError(
                f"OpenAIEmbedder: endpoint returned {len(response.data)} embeddings for {len(texts)} inputs "
                f"(model {self._model!r}); refusing to return misaligned vectors."
            )
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
