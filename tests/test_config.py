"""Tests for ragdoc.config: RagdocConfig, configure(), get_config()."""

import asyncio

import pytest

from ragdoc.config import RagdocConfig, configure, get_config

# =============================================================================
# get_config() defaults
# =============================================================================


def test_default_config():
    config = get_config()
    assert config.azure_key is None
    assert config.azure_endpoint is None
    assert config.download_images is True
    assert config.openai_client is None


# =============================================================================
# configure() as a sync context manager
# =============================================================================


def test_configure_sync_sets_values():
    custom = RagdocConfig(azure_key="k", azure_endpoint="https://example.com")
    with configure(custom):
        cfg = get_config()
        assert cfg.azure_key == "k"
        assert cfg.azure_endpoint == "https://example.com"


def test_configure_sync_restores_after_exit():
    original = get_config()
    with configure(RagdocConfig(azure_key="temporary")):
        assert get_config().azure_key == "temporary"
    assert get_config().azure_key == original.azure_key


def test_configure_sync_restores_on_exception():
    try:
        with configure(RagdocConfig(azure_key="boom")):
            raise ValueError("intentional")
    except ValueError:
        pass
    assert get_config().azure_key is None


def test_configure_sync_nested():
    with configure(RagdocConfig(azure_key="outer")):
        assert get_config().azure_key == "outer"
        with configure(RagdocConfig(azure_key="inner")):
            assert get_config().azure_key == "inner"
        assert get_config().azure_key == "outer"
    assert get_config().azure_key is None


def test_configure_download_images_false():
    with configure(RagdocConfig(download_images=False)):
        assert get_config().download_images is False
    assert get_config().download_images is True


def test_configure_model_copy_update():
    """model_copy(update=...) lets callers override a single field."""
    base = get_config()
    with configure(base.model_copy(update={"download_images": False})):
        cfg = get_config()
        assert cfg.download_images is False
        assert cfg.azure_key == base.azure_key


# =============================================================================
# configure() as an async context manager
# =============================================================================


@pytest.mark.anyio
async def test_configure_async_sets_values():
    async with configure(RagdocConfig(azure_key="async-key")):
        assert get_config().azure_key == "async-key"


@pytest.mark.anyio
async def test_configure_async_restores_after_exit():
    async with configure(RagdocConfig(azure_key="temp")):
        assert get_config().azure_key == "temp"
    assert get_config().azure_key is None


@pytest.mark.anyio
async def test_configure_async_restores_on_exception():
    try:
        async with configure(RagdocConfig(azure_key="boom")):
            raise ValueError("intentional")
    except ValueError:
        pass
    assert get_config().azure_key is None


@pytest.mark.anyio
async def test_configure_async_isolation_across_tasks():
    """ContextVar is propagated into child tasks but changes don't leak back."""

    async def child() -> str | None:
        return get_config().azure_key

    async with configure(RagdocConfig(azure_key="parent-key")):
        # A task spawned inside the context inherits the value
        result = await asyncio.ensure_future(child())
        assert result == "parent-key"

    assert get_config().azure_key is None


# =============================================================================
# LLM client field (Phase 7: typed protocol instead of Any)
# =============================================================================


def test_openai_client_accepts_protocol_stub():
    """Any object with chat + embeddings attributes satisfies the LLMClient protocol field."""
    from unittest.mock import MagicMock

    class _Stub:
        chat = MagicMock()
        embeddings = MagicMock()

    config = RagdocConfig(openai_client=_Stub())
    assert config.openai_client is not None


def test_openai_client_rejects_non_client():
    """An object with neither chat nor embeddings fails LLMClient validation."""
    with pytest.raises(Exception, match=r"LLMClient|openai_client"):
        RagdocConfig(openai_client=object())  # pyright: ignore[reportArgumentType]


def test_default_models_bumped():
    """Pin the current model defaults so the next drift is a conscious decision (gpt-4.x only:
    every call site pins temperature=0.0, which reasoning-family models reject)."""
    config = RagdocConfig()
    assert config.default_llm_model == "gpt-4.1"
    assert config.default_image_llm_model == "gpt-4.1"
