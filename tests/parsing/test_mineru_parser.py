"""Tests for the refactored MinerU parser public API.

Verifies that:
- MinerUParser is a proper Parser subclass (module-level class)
- MinerUExtractor is the renamed internal extraction pipeline (not a Parser)
- parse_mineru_file is importable as a module-level function
- The registry entry is correct after _register()
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

from ragdoc.document import Document
from ragdoc.parsing.mineru import MinerUExtractor, MinerUParser
from ragdoc.parsing.parser import Parser


def test_mineru_parser_is_parser_subclass() -> None:
    assert issubclass(MinerUParser, Parser)


def test_mineru_extractor_is_not_parser_subclass() -> None:
    assert not issubclass(MinerUExtractor, Parser)


def test_mineru_extractor_importable() -> None:
    """MinerUExtractor must be importable from ragdoc.parsing.mineru."""
    from ragdoc.parsing.mineru import MinerUExtractor as E

    assert E is MinerUExtractor


def test_parse_mineru_file_module_level() -> None:
    """parse_mineru_file must be importable directly from ragdoc.parsing.mineru."""
    from ragdoc.parsing.mineru import parse_mineru_file as f

    assert callable(f)


def test_mineru_parser_metadata() -> None:
    parser = MinerUParser()
    assert parser.name == "mineru"
    assert "_middle.json" in parser.patterns
    assert parser.priority == 50


@pytest.mark.anyio
async def test_mineru_parser_call_accepts_path() -> None:
    """MinerUParser.__call__ accepts a Path and returns a Document."""
    fake_doc = Document(elements=[], parser="mineru")
    with patch(
        "ragdoc.parsing.mineru.parse_mineru_file",
        new=AsyncMock(return_value=fake_doc),
    ):
        parser = MinerUParser()
        result = await parser(Path("report_middle.json"))
    assert isinstance(result, Document)


@pytest.mark.anyio
async def test_custom_extractor_injected() -> None:
    """MinerUParser(extractor=...) uses the injected MinerUExtractor."""
    fake_doc = Document(elements=[], parser="mineru")
    custom_extractor = MagicMock(spec=MinerUExtractor)
    custom_extractor.parse = AsyncMock(return_value=fake_doc)

    with patch("ragdoc.parsing.mineru.parse_middle_json_file", return_value=MagicMock()):
        parser = MinerUParser(extractor=custom_extractor)
        result = await parser(Path("report_middle.json"))

    custom_extractor.parse.assert_called_once()
    assert isinstance(result, Document)


def test_register_registers_correctly() -> None:
    """After _register(), get_registered_parsers() contains the mineru entry."""
    from ragdoc.parsing.mineru import _register
    from ragdoc.parsing.registry import get_registered_parsers

    # Ensure it's registered (may already be from import)
    _register()

    registrations = get_registered_parsers()
    mineru_regs = [r for r in registrations if r.name == "mineru"]
    assert len(mineru_regs) >= 1
    reg = mineru_regs[0]
    assert reg.pattern == "_middle.json"
    assert reg.priority == 50
