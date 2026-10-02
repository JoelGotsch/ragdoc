"""Tests for MultiSourceParser."""

from pathlib import Path

import pytest

from ragdoc.document import Document, Heading
from ragdoc.parsing.multi_source import MultiSourceParser
from ragdoc.parsing.parser import Parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(parser_name: str, title: str = "doc") -> Document:
    return Document(
        elements=[Heading(html=f"<h1>{title}</h1>")],
        parser=parser_name,
    )


def _make_parser(parser_name: str, title: str = "doc") -> Parser:
    """Create a trivial Parser instance for testing."""
    _name = parser_name
    _title = title

    class _P(Parser):
        name: str = _name
        patterns: list[str] = []

        async def __call__(self, path: Path) -> Document:
            return _make_doc(_name, _title)

    return _P()


# ---------------------------------------------------------------------------
# Sibling-file resolver routes secondary to resolved path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resolver_routes_secondary_to_resolved_path():
    """secondary_resolver maps primary path → secondary path for the secondary parser."""
    received_paths: dict[str, Path] = {}

    class PrimaryParser(Parser):
        name: str = "primary"
        patterns: list[str] = []

        async def __call__(self, path: Path) -> Document:
            received_paths["primary"] = path
            return _make_doc("primary")

    class SecondaryParser(Parser):
        name: str = "secondary"
        patterns: list[str] = []

        async def __call__(self, path: Path) -> Document:
            received_paths["secondary"] = path
            return _make_doc("secondary")

    def fake_merge(a: Document, b: Document) -> Document:
        return Document(elements=a.elements + b.elements, parser="merged")

    multi = MultiSourceParser(
        primary=PrimaryParser(),
        secondary=SecondaryParser(),
        secondary_resolver=lambda p: p.with_suffix(".pdf"),
        merge=fake_merge,
    )
    result = await multi(Path("report.docx"))

    assert received_paths["primary"] == Path("report.docx")
    assert received_paths["secondary"] == Path("report.pdf")
    assert result.parser == "merged"


# ---------------------------------------------------------------------------
# Resolver returns None → graceful passthrough
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resolver_returns_none_skips_secondary():
    """When resolver returns None, secondary is not called; primary doc is returned as-is."""
    secondary_called = [False]

    class TrackingSecondary(Parser):
        name: str = "secondary"
        patterns: list[str] = []

        async def __call__(self, path: Path) -> Document:
            secondary_called[0] = True
            return _make_doc("secondary")

    multi = MultiSourceParser(
        primary=_make_parser("primary", "Only primary"),
        secondary=TrackingSecondary(),
        secondary_resolver=lambda _: None,
    )
    result = await multi(Path("report.docx"))

    assert not secondary_called[0]
    assert result.parser == "primary"


# ---------------------------------------------------------------------------
# Default merge produces a valid Document
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_default_merge_produces_valid_document():
    """With no custom merge fn, merge_documents is used and doesn't crash."""
    multi = MultiSourceParser(
        primary=_make_parser("a", "Title A"),
        secondary=_make_parser("b", "Title B"),
    )
    result = await multi(Path("file.pdf"))
    assert isinstance(result, Document)
    assert len(result.elements) >= 1


# ---------------------------------------------------------------------------
# description is auto-derived from primary + secondary
# ---------------------------------------------------------------------------


def test_description_auto_derived():
    class ParserA(Parser):
        name: str = "a"
        patterns: list[str] = []
        description: str = "Parser A"

        async def __call__(self, path: Path) -> Document:
            return _make_doc("a")

    class ParserB(Parser):
        name: str = "b"
        patterns: list[str] = []
        description: str = "Parser B"

        async def __call__(self, path: Path) -> Document:
            return _make_doc("b")

    multi = MultiSourceParser(primary=ParserA(), secondary=ParserB())
    assert multi.description == "Merge of Parser A + Parser B"


def test_description_explicit_overrides_derived():
    multi = MultiSourceParser(
        primary=_make_parser("a"),
        secondary=_make_parser("b"),
        description="Custom description",
    )
    assert multi.description == "Custom description"
