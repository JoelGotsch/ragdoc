"""Unit tests for embedder support in VectorStorePipeline.

Tests cover:
- EmbedderConfig.text_fn defaults and custom overrides
- named_embeddings populated correctly after embedding
- Multiple embedders run concurrently
- No embedders → named_embeddings stays empty
- Empty chunk list → embed not called
- Vector order preserved across chunks
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragdoc.chunking.chunk import Chunk
from ragdoc.pipeline.embedders import Embedder, EmbedderConfig, embedding_content_text, prompt_content_text
from ragdoc.pipeline.vectorstore import VectorStorePipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embedder(vectors: list[list[float]]) -> Embedder:
    m = MagicMock(spec=Embedder)
    m.embed = AsyncMock(return_value=vectors)
    return m


def make_chunk(**overrides: object) -> Chunk:
    defaults: dict = dict(prompt_content="prompt text", embedding_content="embed text")
    return Chunk(**(defaults | overrides))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EmbedderConfig text_fn helpers
# ---------------------------------------------------------------------------


def test_embedding_content_text_returns_embedding_content() -> None:
    chunk = make_chunk(prompt_content="the prompt", embedding_content="the embed")
    assert embedding_content_text(chunk) == "the embed"


def test_prompt_content_text_returns_prompt_content() -> None:
    chunk = make_chunk(prompt_content="the prompt", embedding_content="the embed")
    assert prompt_content_text(chunk) == "the prompt"


def test_embedder_config_default_text_fn_is_embedding_content() -> None:
    embedder = make_embedder([[0.1, 0.2]])
    config = EmbedderConfig(embedder=embedder)
    chunk = make_chunk(prompt_content="the prompt", embedding_content="the embed")
    assert config.text_fn(chunk) == "the embed"


def test_embedder_config_custom_text_fn() -> None:
    embedder = make_embedder([[0.1, 0.2]])
    config = EmbedderConfig(embedder=embedder, text_fn=prompt_content_text)
    chunk = make_chunk(prompt_content="the prompt", embedding_content="the embed")
    assert config.text_fn(chunk) == "the prompt"


# ---------------------------------------------------------------------------
# _embed_chunks unit tests (called directly on a minimal pipeline instance)
# ---------------------------------------------------------------------------


def _make_pipeline_with_embedders(embedders: dict | None) -> VectorStorePipeline:
    """Build a VectorStorePipeline with dummy pipeline/store for _embed_chunks tests."""
    pipeline = MagicMock()
    store = MagicMock()
    return VectorStorePipeline(
        pipeline=pipeline,
        vector_store=store,
        embedders=embedders,
    )


@pytest.mark.anyio
async def test_embed_called_with_embedding_content() -> None:
    """embedder.embed receives chunk.embedding_content by default."""
    embedder = make_embedder([[0.1, 0.2, 0.3]])
    config = EmbedderConfig(embedder=embedder)
    vs = _make_pipeline_with_embedders({"dense": config})
    chunk = make_chunk(embedding_content="embed me")

    await vs._embed_chunks([chunk])

    embedder.embed.assert_called_once_with(["embed me"])


@pytest.mark.anyio
async def test_embed_with_prompt_content_text_fn() -> None:
    """embedder.embed receives chunk.prompt_content when text_fn=prompt_content_text."""
    embedder = make_embedder([[0.5, 0.6]])
    config = EmbedderConfig(embedder=embedder, text_fn=prompt_content_text)
    vs = _make_pipeline_with_embedders({"sparse": config})
    chunk = make_chunk(prompt_content="full doc text")

    await vs._embed_chunks([chunk])

    embedder.embed.assert_called_once_with(["full doc text"])


@pytest.mark.anyio
async def test_named_embeddings_populated() -> None:
    """chunk.named_embeddings['dense'] equals the vector returned by the embedder."""
    vec = [0.1, 0.2, 0.3]
    embedder = make_embedder([vec])
    config = EmbedderConfig(embedder=embedder)
    vs = _make_pipeline_with_embedders({"dense": config})
    chunk = make_chunk()

    await vs._embed_chunks([chunk])

    assert chunk.named_embeddings["dense"] == vec


@pytest.mark.anyio
async def test_multiple_embedders_all_run() -> None:
    """Both embedders run and both keys are present in named_embeddings."""
    dense_vec = [0.1, 0.2]
    sparse_vec = [0.9, 0.8]
    dense_embedder = make_embedder([dense_vec])
    sparse_embedder = make_embedder([sparse_vec])

    vs = _make_pipeline_with_embedders(
        {
            "dense": EmbedderConfig(embedder=dense_embedder),
            "sparse": EmbedderConfig(embedder=sparse_embedder, text_fn=prompt_content_text),
        }
    )
    chunk = make_chunk(prompt_content="prompt", embedding_content="embed")

    await vs._embed_chunks([chunk])

    assert chunk.named_embeddings["dense"] == dense_vec
    assert chunk.named_embeddings["sparse"] == sparse_vec
    dense_embedder.embed.assert_called_once()
    sparse_embedder.embed.assert_called_once()


@pytest.mark.anyio
async def test_embedders_run_concurrently() -> None:
    """Both embedder tasks fire within a single asyncio.gather call."""
    call_order: list[str] = []

    async def slow_embed_a(texts: list[str]) -> list[list[float]]:
        call_order.append("a_start")
        await asyncio.sleep(0)  # yield control
        call_order.append("a_end")
        return [[1.0] * len(texts)]

    async def slow_embed_b(texts: list[str]) -> list[list[float]]:
        call_order.append("b_start")
        await asyncio.sleep(0)
        call_order.append("b_end")
        return [[2.0] * len(texts)]

    embedder_a = MagicMock()
    embedder_a.embed = slow_embed_a
    embedder_b = MagicMock()
    embedder_b.embed = slow_embed_b

    vs = _make_pipeline_with_embedders(
        {
            "a": EmbedderConfig(embedder=embedder_a),
            "b": EmbedderConfig(embedder=embedder_b),
        }
    )
    chunk = make_chunk()
    await vs._embed_chunks([chunk])

    # Both started before either ended (interleaved = concurrent)
    assert call_order.index("a_start") < call_order.index("b_end")
    assert call_order.index("b_start") < call_order.index("a_end")


@pytest.mark.anyio
async def test_no_embedders_leaves_named_embeddings_empty() -> None:
    """Pipeline with embedders=None: named_embeddings stays {}."""
    vs = _make_pipeline_with_embedders(None)
    chunk = make_chunk()

    await vs._embed_chunks([chunk])

    assert chunk.named_embeddings == {}


@pytest.mark.anyio
async def test_empty_chunks_no_embed_call() -> None:
    """embedder.embed is not called when chunk list is empty."""
    embedder = make_embedder([])
    config = EmbedderConfig(embedder=embedder)
    vs = _make_pipeline_with_embedders({"dense": config})

    await vs._embed_chunks([])

    embedder.embed.assert_not_called()


@pytest.mark.anyio
async def test_embed_order_preserved() -> None:
    """Vectors are assigned to the correct chunk by position."""
    vec_a = [0.1, 0.2]
    vec_b = [0.3, 0.4]
    embedder = make_embedder([vec_a, vec_b])
    config = EmbedderConfig(embedder=embedder)
    vs = _make_pipeline_with_embedders({"dense": config})

    chunk_a = make_chunk(embedding_content="first")
    chunk_b = make_chunk(embedding_content="second")

    await vs._embed_chunks([chunk_a, chunk_b])

    assert chunk_a.named_embeddings["dense"] == vec_a
    assert chunk_b.named_embeddings["dense"] == vec_b
