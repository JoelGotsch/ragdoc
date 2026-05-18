"""Async Parser base class for the ragdoc parsing system.

All parsers inherit from :class:`Parser`, which is a Pydantic ``BaseModel`` combined
with ``ABC``.  Metadata fields (``name``, ``patterns``, ``priority``, ``description``)
are declared at class level and used by the parser registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from ragdoc.document import Document


class Parser(BaseModel, ABC):
    """Base class for all ragdoc parsers.

    Declare metadata as class-level field defaults::

        class HTMLParser(Parser):
            name: str = "html"
            patterns: list[str] = [".html", ".htm"]
            description: str = "HTML files"

            async def __call__(self, path: Path) -> Document:
                ...

    Instantiate with no arguments: ``HTMLParser()``.
    """

    name: str = Field(description="Unique registry key, e.g. 'html'")
    patterns: list[str] = Field(description="Suffix patterns this parser handles, e.g. ['.html', '.htm']")
    priority: int = Field(default=0, description="Higher value = higher precedence when multiple parsers match a path")
    description: str = Field(default="", description="Human-readable description for registry display")

    @abstractmethod
    async def __call__(self, path: Path) -> Document: ...
