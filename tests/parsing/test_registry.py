"""Tests for the parser registry (resolve, priority, longest-match, config override)."""

from pathlib import Path

import pytest

from ragdoc.config import RagdocConfig, configure
from ragdoc.document import Document
from ragdoc.parsing.parser import Parser
from ragdoc.parsing.registry import (
    _clear_registry,
    _resolve_parser,
    get_parser,
    get_registered_parsers,
    register_parser,
    unregister_parser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser(name: str, patterns: list[str] | None = None, priority: int = 0) -> Parser:
    """Create a trivial Parser instance for testing."""
    _name = name
    _patterns = patterns or []
    _priority = priority

    class _P(Parser):
        name: str = _name
        patterns: list[str] = _patterns
        priority: int = _priority

        async def __call__(self, path: Path) -> Document:
            return Document(elements=[], parser=_name, metadata={"path": str(path)})

    return _P()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean registry and restores it afterward."""
    from ragdoc.parsing.registry import _registry

    snapshot = list(_registry)
    _clear_registry()
    yield
    _clear_registry()
    _registry.extend(snapshot)


# ---------------------------------------------------------------------------
# get_parser / unregister_parser
# ---------------------------------------------------------------------------


def test_get_parser_by_name():
    p = _make_parser("my_parser", [".xyz"])
    register_parser(p)
    assert get_parser("my_parser") is p


def test_get_parser_not_found():
    with pytest.raises(ValueError, match="No parser registered with name"):
        get_parser("nonexistent")


def test_unregister():
    p = _make_parser("gone", [".txt"])
    register_parser(p)
    unregister_parser(".txt", "gone")
    assert get_registered_parsers() == []


def test_unregister_not_found():
    with pytest.raises(ValueError, match="No parser registration found"):
        unregister_parser(".nope", "nope")


# ---------------------------------------------------------------------------
# _resolve_parser — suffix matching
# ---------------------------------------------------------------------------


def test_resolve_simple_suffix():
    p = _make_parser("html", [".html"])
    register_parser(p)
    assert _resolve_parser(Path("report.html")) is p


def test_resolve_case_insensitive():
    p = _make_parser("html", [".html"])
    register_parser(p)
    assert _resolve_parser(Path("REPORT.HTML")) is p


def test_resolve_no_match():
    register_parser(_make_parser("html", [".html"]))
    with pytest.raises(ValueError, match="No parser registered"):
        _resolve_parser(Path("file.unknown"))


# ---------------------------------------------------------------------------
# _resolve_parser — longest match wins
# ---------------------------------------------------------------------------


def test_resolve_longest_match_wins():
    p_json = _make_parser("generic_json", [".json"])
    p_azure = _make_parser("azure_json", [".azure.json"])
    register_parser(p_json)
    register_parser(p_azure)

    assert _resolve_parser(Path("report.azure.json")) is p_azure
    assert _resolve_parser(Path("data.json")) is p_json


def test_resolve_underscore_suffix():
    p = _make_parser("mineru", ["_middle.json"])
    register_parser(p)
    assert _resolve_parser(Path("report_middle.json")) is p


# ---------------------------------------------------------------------------
# _resolve_parser — priority tiebreaker
# ---------------------------------------------------------------------------


def test_resolve_priority_tiebreaker():
    p_low = _make_parser("low", [".pdf"], priority=10)
    p_high = _make_parser("high", [".pdf"], priority=50)
    register_parser(p_low)
    register_parser(p_high)

    assert _resolve_parser(Path("file.pdf")) is p_high


# ---------------------------------------------------------------------------
# _resolve_parser — explicit name overrides pattern matching
# ---------------------------------------------------------------------------


def test_resolve_explicit_name():
    p = _make_parser("explicit", [".html"])
    register_parser(p)
    assert _resolve_parser(Path("file.html"), parser_name="explicit") is p


# ---------------------------------------------------------------------------
# _resolve_parser — config preferences override priority
# ---------------------------------------------------------------------------


def test_resolve_config_preference_overrides_priority():
    p_low = _make_parser("low_prio", [".pdf"], priority=10)
    p_high = _make_parser("high_prio", [".pdf"], priority=50)
    register_parser(p_low)
    register_parser(p_high)

    with configure(RagdocConfig(parser_preferences={".pdf": "low_prio"})):
        assert _resolve_parser(Path("file.pdf")) is p_low


# ---------------------------------------------------------------------------
# Built-in registrations smoke test
# ---------------------------------------------------------------------------


def test_builtin_parsers_registered():
    """After calling _register_builtin_parsers, core parsers are in the registry."""
    from ragdoc.parsing import _register_builtin_parsers
    from ragdoc.parsing.registry import _registry as live_registry

    _register_builtin_parsers()
    names = {r.name for r in live_registry}
    assert "html" in names
    assert "pandoc" in names
    assert "xlsx" in names
