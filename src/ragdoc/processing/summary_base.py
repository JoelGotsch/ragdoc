"""Shared result models for LLM-based summary processors.

This module contains only the Pydantic output models. All prompt-building logic,
factory functions, and processors live in the per-element-type modules:

- ``summary_image``     — image summarization
- ``summary_document``  — whole-document summarization
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageSummary(BaseModel):
    """Structured result returned by an image summarization call."""

    summary: str = Field(description="Rich description of the image content.")
    text_representation: str | None = Field(
        default=None,
        description=(
            "Structured representation of the main content: "
            "HTML for tables, Mermaid for graphs/flowcharts, MathML for formulas."
        ),
    )
    decorative: bool = Field(
        default=False,
        description=(
            "True if the image is decorative (logos, spacers, dividers, stock photos) "
            "and should be excluded from document content."
        ),
    )


class DocumentSummary(BaseModel):
    """Structured result returned by a document summarization call."""

    summary: str = Field(description="Concise, retrieval-optimised summary of the supplied content.")
