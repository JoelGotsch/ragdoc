"""
Library-wide configuration for ragdoc.

Provides a ContextVar-based configuration system that is safe for both
sync and async usage. Use the ``configure()`` context manager to set the
active configuration for a block of code.

Example::

    from openai import AsyncOpenAI
    from ragdoc.config import configure, RagdocConfig

    async with configure(RagdocConfig(azure_key="...", azure_endpoint="...")):
        doc = await load(path)

    # Override a single setting:
    current = get_config()
    with configure(current.model_copy(update={"download_images": False})):
        doc = generate_document(html)
"""

from __future__ import annotations

from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict, Field

from ragdoc.llm import LLMClient


class RagdocConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    azure_key: str | None = Field(default=None, description="Azure DI API key")
    azure_endpoint: str | None = Field(default=None, description="Azure DI endpoint URL")
    download_images: bool = Field(default=True, description="Download remote images in HTML parser")
    openai_client: LLMClient | None = Field(
        default=None,
        description="Default AsyncOpenAI-compatible client (chat + embeddings) for LLM features.",
    )
    default_llm_model: str = Field(
        default="gpt-4.1",
        description="Default model name for LLM processors if not overridden",
    )
    default_image_llm_model: str = Field(
        default="gpt-4.1",
        description="Default model name for image LLM processors if not overridden. Must be a model that supports image inputs.",
    )
    parser_preferences: dict[str, str] = Field(
        default_factory=dict,
        description="Map of file suffix pattern to parser name. Overrides priority-based selection.",
    )


_DEFAULT_CONFIG = RagdocConfig()
_config: ContextVar[RagdocConfig | None] = ContextVar("ragdoc_config", default=None)


class _ConfigContext:
    """Context manager returned by configure(). Supports both `with` and `async with`."""

    def __init__(self, config: RagdocConfig) -> None:
        self._config = config
        self._token = None

    def __enter__(self) -> RagdocConfig:
        self._token = _config.set(self._config)
        return self._config

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            _config.reset(self._token)

    async def __aenter__(self) -> RagdocConfig:
        self._token = _config.set(self._config)
        return self._config

    async def __aexit__(self, *args: object) -> None:
        if self._token is not None:
            _config.reset(self._token)


def configure(config: RagdocConfig) -> _ConfigContext:
    """
    Set the active RagdocConfig for the current (async) task context.

    Works as both a sync ``with`` block and an ``async with`` block.
    The previous config is restored when the block exits, even on exceptions.

    Args:
        config: The RagdocConfig to activate.

    Example::

        async with configure(RagdocConfig(azure_key="k", azure_endpoint="https://...")):
            doc = await load(path)
    """
    return _ConfigContext(config)


def get_config() -> RagdocConfig:
    """Return the currently active RagdocConfig (or the library default)."""
    active = _config.get()
    return active if active is not None else _DEFAULT_CONFIG
