"""Tests for BaseException handling in the pipeline (Issue 3).

Verifies:
- asyncio.CancelledError and SystemExit propagate out of run() rather than
  being silently swallowed by the per-file except Exception handler.
- RuntimeError and other Exception subclasses are still collected in
  result.errors (no regression).
- "Run failed" is logged before a fatal exception escapes.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline

from .conftest import MemoryVectorStore, make_document


# ---------------------------------------------------------------------------
# Parser factories
# ---------------------------------------------------------------------------


def _parser_raises(make_exc):
    """Return a parser that raises make_exc() for every path."""

    async def _parse(path: Path) -> Document:
        raise make_exc()

    return _parse


def _parser_raises_for(failing_name: str, make_exc):
    """Return a parser that raises make_exc() for failing_name, succeeds otherwise."""

    async def _parse(path: Path) -> Document:
        if path.name == failing_name:
            raise make_exc()
        return make_document(title=path.stem, body="ok")

    return _parse


# ---------------------------------------------------------------------------
# VectorStorePipeline — BaseException propagation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_vs_cancelled_error_propagates(tmp_path: Path):
    """CancelledError must propagate out of VectorStorePipeline.run()."""
    f = tmp_path / "a.txt"
    f.write_text("content", encoding="utf-8")

    vs = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parser_raises(asyncio.CancelledError)),
        vector_store=MemoryVectorStore(),
    )
    with pytest.raises(asyncio.CancelledError):
        await vs.run([f])


@pytest.mark.anyio
async def test_vs_unknown_base_exception_collected(tmp_path: Path):
    """An arbitrary BaseException subclass (not CancelledError) is collected rather
    than silently dropped, verifying the except BaseException clause.
    """

    class _WeirdBaseError(BaseException):
        pass

    f = tmp_path / "a.txt"
    f.write_text("content", encoding="utf-8")

    vs = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parser_raises(_WeirdBaseError)),
        vector_store=MemoryVectorStore(),
    )
    result = await vs.run([f])

    assert len(result.errors) == 1
    assert isinstance(result.errors[0][1], _WeirdBaseError)
    assert result.processed == []


@pytest.mark.anyio
async def test_vs_runtime_error_collected(tmp_path: Path):
    """RuntimeError is collected in result.errors; the other file completes."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("aaa", encoding="utf-8")
    b.write_text("bbb", encoding="utf-8")

    vs = VectorStorePipeline(
        pipeline=DocumentPipeline(
            parser=_parser_raises_for("a.txt", lambda: RuntimeError("boom"))
        ),
        vector_store=MemoryVectorStore(),
    )
    result = await vs.run([a, b])

    assert len(result.errors) == 1
    assert result.errors[0][0] == a
    assert isinstance(result.errors[0][1], RuntimeError)
    assert b in result.processed
    assert len(result.processed) == 1


@pytest.mark.anyio
async def test_vs_memory_error_collected(tmp_path: Path):
    """MemoryError (Exception subclass) is collected, not treated as fatal."""
    f = tmp_path / "a.txt"
    f.write_text("content", encoding="utf-8")

    vs = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parser_raises(MemoryError)),
        vector_store=MemoryVectorStore(),
    )
    result = await vs.run([f])

    assert len(result.errors) == 1
    assert isinstance(result.errors[0][1], MemoryError)
    assert result.processed == []


@pytest.mark.anyio
async def test_vs_run_failed_logged_on_propagated_exception(tmp_path: Path, caplog):
    """'Run failed' is logged before the fatal exception escapes."""
    f = tmp_path / "a.txt"
    f.write_text("content", encoding="utf-8")

    vs = VectorStorePipeline(
        pipeline=DocumentPipeline(parser=_parser_raises(asyncio.CancelledError)),
        vector_store=MemoryVectorStore(),
    )
    with caplog.at_level(logging.ERROR, logger="ragdoc.pipeline.vectorstore"):
        with pytest.raises(asyncio.CancelledError):
            await vs.run([f])

    assert any("Run failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# DocumentPipeline (on_error="skip") — BaseException propagation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_linear_cancelled_error_propagates():
    """CancelledError propagates from run_many() even with on_error='skip'."""
    pipeline = DocumentPipeline(
        parser=_parser_raises(asyncio.CancelledError),
        on_error="skip",
    )
    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_many([Path("doc.html")])


@pytest.mark.anyio
async def test_linear_unknown_base_exception_collected():
    """An arbitrary BaseException subclass (not CancelledError) is collected
    with on_error='skip', verifying the except BaseException clause.
    """

    class _WeirdBaseError(BaseException):
        pass

    pipeline = DocumentPipeline(
        parser=_parser_raises(_WeirdBaseError),
        on_error="skip",
    )
    result = await pipeline.run_many([Path("doc.html")])

    assert len(result.errors) == 1
    assert isinstance(result.errors[0][1], _WeirdBaseError)
    assert result.chunks == []


@pytest.mark.anyio
async def test_linear_runtime_error_collected():
    """RuntimeError is collected in result.errors with on_error='skip'."""
    pipeline = DocumentPipeline(
        parser=_parser_raises_for("bad.html", lambda: RuntimeError("boom")),
        on_error="skip",
    )
    result = await pipeline.run_many([Path("bad.html"), Path("good.html")])

    assert len(result.errors) == 1
    assert result.errors[0][0] == Path("bad.html")
    assert isinstance(result.errors[0][1], RuntimeError)
    assert len(result.chunks) > 0
