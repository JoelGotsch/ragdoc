"""Unit tests for pure-logic functions: Document.content_hash(), SimpleChunker ID
generation, and TokenSplitter splitting behavior.

These tests exercise deterministic behavior of core primitives without running
the full DocumentPipeline.
"""

from __future__ import annotations

import pytest

from ragdoc.chunking import SimpleChunker
from ragdoc.document import Document, Heading, Paragraph
from ragdoc.pipeline import TokenSplitter

from .conftest import make_document

# --- TestContentHash ---


def test_content_hash_returns_64_char_hex():
    doc = make_document()
    h = doc.content_hash()
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_same_content_same_hash():
    doc1 = make_document(title="A", body="Hello world.")
    doc2 = make_document(title="A", body="Hello world.")
    assert doc1.content_hash() == doc2.content_hash()


def test_content_hash_different_content_different_hash():
    doc1 = make_document(title="A", body="Hello world.")
    doc2 = make_document(title="A", body="Goodbye world.")
    assert doc1.content_hash() != doc2.content_hash()


def test_content_hash_different_title_different_hash():
    doc1 = make_document(title="Alpha")
    doc2 = make_document(title="Beta")
    assert doc1.content_hash() != doc2.content_hash()


def test_content_hash_empty_document_is_stable():
    doc = Document()
    h1 = doc.content_hash()
    h2 = doc.content_hash()
    assert h1 == h2


def test_content_hash_custom_renderer_accepted():
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

    renderer = Renderer(format=OutputFormat.PLAIN, element_renderer=render_for_prompt)
    doc = make_document()
    h = doc.content_hash(renderer=renderer)
    assert len(h) == 64


def test_content_hash_custom_renderer_can_differ_from_default():
    """Different renderers may produce the same or different hashes -- just
    ensure neither raises and both return valid hex strings."""
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

    doc = make_document(title="Test", body="Some text here.")
    default_hash = doc.content_hash()
    plain_renderer = Renderer(format=OutputFormat.PLAIN, element_renderer=render_for_prompt)
    plain_hash = doc.content_hash(renderer=plain_renderer)
    assert len(default_hash) == 64
    assert len(plain_hash) == 64


# --- TestSimpleChunkerContentHash ---


@pytest.mark.anyio
async def test_simple_chunker_default_id_is_content_hash():
    doc = make_document(title="Stable", body="Same content.")
    chunker = SimpleChunker()
    chunks1 = await chunker.chunk(doc)
    chunks2 = await chunker.chunk(doc)
    assert chunks1[0].id == chunks2[0].id


@pytest.mark.anyio
async def test_simple_chunker_default_id_matches_content_hash():
    doc = make_document(title="Stable", body="Same content.")
    chunker = SimpleChunker()
    chunks = await chunker.chunk(doc)
    assert chunks[0].id == doc.content_hash()


@pytest.mark.anyio
async def test_simple_chunker_different_docs_different_chunk_ids():
    doc1 = make_document(title="A")
    doc2 = make_document(title="B")
    chunker = SimpleChunker()
    id1 = (await chunker.chunk(doc1))[0].id
    id2 = (await chunker.chunk(doc2))[0].id
    assert id1 != id2


@pytest.mark.anyio
async def test_simple_chunker_custom_id_fn_overrides_default():
    doc = make_document()
    chunker = SimpleChunker(id_fn=lambda _: "fixed-id")
    assert (await chunker.chunk(doc))[0].id == "fixed-id"


# --- TestTokenSplitter ---


def test_token_splitter_small_document_is_not_split():
    doc = make_document(title="Short", body="A brief paragraph.")
    splitter = TokenSplitter(max_tokens=7000)
    splits = splitter(doc)
    assert len(splits) == 1
    assert splits[0] is doc


def test_token_splitter_large_document_is_split():
    body = "Word " * 200  # well over 200 tokens
    doc = Document(
        title="Long doc",
        elements=[
            Heading(innerhtml="Long doc", level=1),
            Paragraph(html_content=f"<p>{body}</p>"),
        ],
    )
    splitter = TokenSplitter(max_tokens=200, overlap_tokens=20)
    splits = splitter(doc)
    assert len(splits) >= 2


def test_token_splitter_respects_max_tokens_param():
    body = "Word " * 300
    doc = Document(
        title="T",
        elements=[
            Heading(innerhtml="T", level=1),
            Paragraph(html_content=f"<p>{body}</p>"),
        ],
    )
    splitter_tight = TokenSplitter(max_tokens=150, overlap_tokens=20)
    splitter_loose = TokenSplitter(max_tokens=1000, overlap_tokens=20)
    assert len(splitter_tight(doc)) >= len(splitter_loose(doc))
