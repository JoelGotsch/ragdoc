"""Mutation-isolation tests: every split owns its metadata dict (copied at construction).

Element sharing across splits remains by-reference by design (pinned by
``test_oversized_heading_context_same_objects``); only metadata is isolated.
"""

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
from ragdoc.splitting.base import split_by_headings
from ragdoc.splitting.hierarchical import split_hierarchical
from ragdoc.splitting.token import split_by_elements, split_document


class CharTokenizer:
    """Stub tokenizer where 1 UTF-8 byte = 1 token."""

    def __call__(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(t & 0xFF for t in tokens).decode("utf-8", errors="replace")

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


def rdr() -> Renderer:
    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


def _doc() -> Document:
    return Document(
        elements=[
            Heading(html="<h1>Section A</h1>"),
            Paragraph(html=f"<p>{'A' * 80}</p>"),
            Heading(html="<h1>Section B</h1>"),
            Paragraph(html=f"<p>{'B' * 80}</p>"),
        ],
        metadata={"origin": "input"},
    )


def test_splits_have_independent_metadata_dicts():
    """After split_document, mutating one split's metadata touches neither siblings nor input."""
    doc = _doc()
    splits = split_document(doc, rdr(), CharTokenizer(), max_tokens=120, overlap_tokens=5)
    assert len(splits) > 1
    splits[0].metadata["poison"] = True
    assert all("poison" not in s.metadata for s in splits[1:])
    assert "poison" not in doc.metadata
    assert doc.metadata == {"origin": "input"}  # sequence stamping never leaks to input


def test_split_by_headings_metadata_isolated():
    doc = _doc()
    splits = split_by_headings(doc)
    assert len(splits) > 1
    splits[0].metadata["poison"] = True
    assert all("poison" not in s.metadata for s in splits[1:])
    assert "poison" not in doc.metadata


def test_split_by_elements_metadata_isolated():
    doc = _doc()
    splits = split_by_elements(doc, rdr(), CharTokenizer(), max_tokens=120, overlap_tokens=5)
    assert len(splits) > 1
    splits[0].metadata["poison"] = True
    assert all("poison" not in s.metadata for s in splits[1:])
    assert "poison" not in doc.metadata


def test_split_hierarchical_metadata_isolated():
    doc = _doc()
    splits = split_hierarchical(doc)
    assert len(splits) > 1
    splits[0].metadata["poison"] = True
    assert all("poison" not in s.metadata for s in splits[1:])
    assert "poison" not in doc.metadata


def test_no_split_path_does_not_mutate_input():
    """The fits-path returns a copy carrying split_sequence; the input document stays clean."""
    doc = _doc()
    result = split_document(doc, rdr(), CharTokenizer(), max_tokens=10_000, overlap_tokens=5)
    assert len(result) == 1
    assert "split_sequence" not in doc.metadata
    assert result[0].metadata["split_sequence"] == 1
    assert result[0].metadata["split_total"] == 1
    # elements stay shared (by design)
    assert result[0].elements[0] is doc.elements[0]
