"""Tests for split_sequence/split_total metadata assignment by split_document."""

import random

import pytest

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
from ragdoc.splitting.base import split_by_headings
from ragdoc.splitting.hierarchical import split_hierarchical
from ragdoc.splitting.token import split_by_elements, split_document

# ---------------------------------------------------------------------------
# Helpers (same shape as test_token.py)
# ---------------------------------------------------------------------------


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


_OVERLAP = 5


def rdr() -> Renderer:
    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


def tok() -> CharTokenizer:
    return CharTokenizer()


def h(level: int, text: str = "") -> Heading:
    inner = text or f"H{level}"
    return Heading(html=f"<h{level}>{inner}</h{level}>")


def p(text: str) -> Paragraph:
    return Paragraph(html=f"<p>{text}</p>")


def make_doc(*elements) -> Document:
    return Document(elements=list(elements))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_no_split_sets_sequence_1():
    """Document that fits the budget is returned (as a copy) with split_sequence=1, split_total=1."""
    doc = make_doc(h(1, "Title"), p("Short"))
    result = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=10_000)
    assert len(result) == 1
    assert result[0].elements == doc.elements
    assert result[0].metadata["split_sequence"] == 1
    assert result[0].metadata["split_total"] == 1
    assert "split_sequence" not in doc.metadata  # input untouched


def test_single_level_split_assigns_contiguous_sequence():
    """3-heading doc split by token budget gets split_sequence [1, 2, 3]."""
    # Each section is ~30 chars; budget of 40 forces a split after each heading
    doc = make_doc(
        h(1, "Section One"),
        p("Content of section one here."),
        h(1, "Section Two"),
        p("Content of section two here."),
        h(1, "Section Three"),
        p("Content of section three here."),
    )
    splits = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=40, overlap_tokens=_OVERLAP)
    assert len(splits) >= 2
    sequences = [d.metadata["split_sequence"] for d in splits]
    assert sequences == list(range(1, len(splits) + 1))


def test_split_total_equals_len_splits():
    """Every split carries split_total equal to the total number of splits."""
    doc = make_doc(
        h(1, "A"),
        p("paragraph a " * 5),
        h(1, "B"),
        p("paragraph b " * 5),
        h(1, "C"),
        p("paragraph c " * 5),
    )
    splits = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=150, overlap_tokens=_OVERLAP)
    assert len(splits) > 1
    assert all(d.metadata["split_total"] == len(splits) for d in splits)


def test_sequence_equals_list_position():
    """split_sequence is 1-based and equals the list index + 1."""
    doc = make_doc(
        h(1, "A"),
        p("paragraph a " * 5),
        h(1, "B"),
        p("paragraph b " * 5),
        h(1, "C"),
        p("paragraph c " * 5),
    )
    splits = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=150, overlap_tokens=_OVERLAP)
    for i, d in enumerate(splits):
        assert d.metadata["split_sequence"] == i + 1


def test_sorting_by_split_sequence_matches_list_order():
    """Shuffling then sorting by split_sequence restores original order."""
    doc = make_doc(
        h(1, "A"),
        p("paragraph a " * 5),
        h(1, "B"),
        p("paragraph b " * 5),
        h(1, "C"),
        p("paragraph c " * 5),
    )
    splits = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=150, overlap_tokens=_OVERLAP)
    assert len(splits) >= 2

    shuffled = splits[:]
    random.shuffle(shuffled)
    restored = sorted(shuffled, key=lambda d: d.metadata["split_sequence"])
    assert [d.id for d in restored] == [d.id for d in splits]


def test_recursive_split_sequences_are_globally_ordered():
    """After recursive hierarchical+element split, sequences have no gaps or collisions."""
    # A: small, B: large (forces element-level split), C: small
    long_content = "x " * 200
    doc = make_doc(
        h(1, "A"),
        p("short a"),
        h(1, "B"),
        p(long_content),
        h(1, "C"),
        p("short c"),
    )
    splits = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=150, overlap_tokens=_OVERLAP)
    assert len(splits) >= 3
    sequences = [d.metadata["split_sequence"] for d in splits]
    assert sequences == list(range(1, len(splits) + 1))
    assert all(d.metadata["split_total"] == len(splits) for d in splits)


@pytest.mark.parametrize(
    "splitter,kwargs",
    [
        (split_by_headings, {}),
        (split_hierarchical, {}),
        (split_by_elements, {"renderer": rdr(), "tokenizer": tok(), "max_tokens": 150, "overlap_tokens": _OVERLAP}),
    ],
)
def test_primitive_splitters_do_not_set_split_metadata(splitter, kwargs):
    """Primitive splitters never write split_sequence or split_total."""
    long_content = "word " * 50
    doc = make_doc(
        h(1, "A"),
        p(long_content),
        h(1, "B"),
        p(long_content),
        h(1, "C"),
        p(long_content),
    )
    result = splitter(doc, **kwargs)
    assert len(result) > 1
    assert all("split_sequence" not in d.metadata for d in result)
    assert all("split_total" not in d.metadata for d in result)


# ---------------------------------------------------------------------------
# Integration: split_sequence survives a processing pipeline
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_split_sequence_propagates_through_processing_pipeline():
    """Processors must not clear split_sequence/split_total; values survive a full pipeline pass."""
    from ragdoc.document import Document as Doc
    from ragdoc.processing.base import DocumentProcessor, ProcessingPipeline

    class IdentityProcessor(DocumentProcessor):
        async def process(self, document: Doc) -> Doc:
            return document

    long_content = "word " * 50
    doc = make_doc(
        h(1, "A"),
        p(long_content),
        h(1, "B"),
        p(long_content),
        h(1, "C"),
        p(long_content),
    )
    splits = split_document(doc, renderer=rdr(), tokenizer=tok(), max_tokens=150, overlap_tokens=_OVERLAP)
    n = len(splits)
    assert n > 1

    pipeline = ProcessingPipeline([IdentityProcessor()])
    processed = [await pipeline.process(d) for d in splits]
    assert [d.metadata["split_sequence"] for d in processed] == list(range(1, n + 1))
    assert all(d.metadata["split_total"] == n for d in processed)
