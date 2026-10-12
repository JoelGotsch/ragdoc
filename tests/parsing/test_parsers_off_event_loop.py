"""Parsers must not run blocking work (pandoc subprocesses, file/network I/O) on the event loop.

Each test records the thread in which a parser's blocking primitive executes and asserts it
is *not* the thread running the event loop.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

import ragdoc.parsing.azure_di.load as azure_load_module
import ragdoc.parsing.html.load as html_load_module
import ragdoc.parsing.mineru as mineru_module
import ragdoc.parsing.mineru.handlers as mineru_handlers_module
from ragdoc.config import RagdocConfig, configure
from ragdoc.document import Image, Table
from ragdoc.parsing.azure_di import AzureDIParser, AzureJSONParser
from ragdoc.parsing.html import HTMLParser
from ragdoc.parsing.mineru import MinerUExtractor, MinerUParser
from ragdoc.parsing.mineru.base import (
    ChartBlock,
    ChartBodyBlock,
    ChartSpan,
    Line,
    MinerUMiddleDocument,
    PageInfo,
)


class ThreadRecorder:
    """Wraps a callable and records the thread ident of every call."""

    def __init__(self, wrapped: Callable[..., object]) -> None:
        self.wrapped = wrapped
        self.thread_ids: list[int] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.thread_ids.append(threading.get_ident())
        return self.wrapped(*args, **kwargs)

    def assert_ran_off_loop(self) -> None:
        assert self.thread_ids, "the blocking call was never reached"
        assert threading.get_ident() not in self.thread_ids, "blocking call ran on the event-loop thread"


def _chart_document() -> MinerUMiddleDocument:
    bbox = [0.0, 0.0, 100.0, 50.0]
    span = ChartSpan(bbox=bbox, content="| a | b |\n|---|---|\n| 1 | 2 |\n")
    body = ChartBodyBlock(bbox=bbox, index=0, lines=[Line(bbox=bbox, spans=[span])])
    chart = ChartBlock(bbox=bbox, index=0, blocks=[body])
    return MinerUMiddleDocument(pdf_info=[PageInfo(para_blocks=[chart], discarded_blocks=[], page_idx=0)])


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_html_parser_parses_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "page.html"
    path.write_text("<html><body><h1>T</h1><p>body</p></body></html>", encoding="utf-8")
    recorder = ThreadRecorder(html_load_module.generate_document)
    monkeypatch.setattr("ragdoc.parsing.html.generate_html_document", recorder)

    doc = await HTMLParser()(path)

    assert [el.text for el in doc.elements] == ["T", "body"]
    recorder.assert_ran_off_loop()


@pytest.mark.anyio
async def test_html_parser_downloads_remote_images_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The per-image ``httpx.get`` is synchronous; it must run in the worker thread.

    Also proves the active ``configure(...)`` context (``download_images``) reaches that thread.
    """
    path = tmp_path / "page.html"
    path.write_text('<html><body><p>pic <img src="https://example.invalid/a.png"/></p></body></html>', encoding="utf-8")

    def fake_get(url: str, **_: object) -> httpx.Response:
        return httpx.Response(200, content=b"png-bytes", request=httpx.Request("GET", url))

    recorder = ThreadRecorder(fake_get)
    monkeypatch.setattr(html_load_module.httpx, "get", recorder)

    with configure(RagdocConfig(download_images=True)):
        doc = await HTMLParser()(path)

    assert len([el for el in doc.elements if isinstance(el, Image)]) == 1
    recorder.assert_ran_off_loop()


# ---------------------------------------------------------------------------
# Azure DI
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_azure_json_parser_runs_pandoc_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "report.azure.json"
    path.write_text(json.dumps({"content": "# Title\n\nSome body text.\n"}), encoding="utf-8")
    recorder = ThreadRecorder(azure_load_module.convert_text)
    monkeypatch.setattr(azure_load_module, "convert_text", recorder)

    doc = await AzureJSONParser()(path)

    assert doc.parser == "azure_di"
    assert [el.text for el in doc.elements] == ["Title", "Some body text."]
    recorder.assert_ran_off_loop()


@pytest.mark.anyio
async def test_azure_di_parser_calls_the_service_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("azure.ai.documentintelligence", reason="azure-di extra not installed")
    from ragdoc.parsing.azure_di import client as azure_client_module

    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    recorder = ThreadRecorder(lambda _path: {"content": "# Title\n\nSome body text.\n"})
    monkeypatch.setattr(azure_client_module, "get_analyze_result", recorder)

    doc = await AzureDIParser()(path)

    assert doc.parser == "azure_di"
    assert doc.source_path == str(path)
    recorder.assert_ran_off_loop()


# ---------------------------------------------------------------------------
# MinerU
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mineru_chart_handler_runs_pandoc_off_the_event_loop(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")
    recorder = ThreadRecorder(mineru_handlers_module._pandoc_convert)
    monkeypatch.setattr(mineru_handlers_module, "_pandoc_convert", recorder)

    doc = await MinerUExtractor().parse(_chart_document())

    assert len([el for el in doc.elements if isinstance(el, Table)]) == 1
    recorder.assert_ran_off_loop()


@pytest.mark.anyio
async def test_mineru_parser_reads_the_middle_json_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")
    path = tmp_path / "report_middle.json"
    path.write_text(_chart_document().model_dump_json(), encoding="utf-8")
    recorder = ThreadRecorder(mineru_module.parse_middle_json_file)
    monkeypatch.setattr(mineru_module, "parse_middle_json_file", recorder)

    default_doc = await MinerUParser()(path)
    custom_doc = await MinerUParser(extractor=MinerUExtractor())(path)

    assert default_doc.parser == custom_doc.parser == "mineru"
    assert len(recorder.thread_ids) == 2
    recorder.assert_ran_off_loop()
