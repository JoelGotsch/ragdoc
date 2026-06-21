"""Tests for DocumentPipeline (Scenario A).

Covers:
- DocumentPipeline.run() — single file, error propagation, ID stability
- DocumentPipeline.run_many() — multiple files, concurrency, on_error
- DocumentPipeline.stream() — async iteration
- AutoParser — unsupported extension raises

See test_linear_integration.py for full parse → process → chunk coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.pipeline import DocumentPipeline, PipelineResult

from .conftest import make_document

# --- run() ---


@pytest.mark.anyio
async def test_pipeline_run_chunk_id_is_deterministic(simple_parser):
    pipeline = DocumentPipeline(parser=simple_parser)
    chunks1 = await pipeline.run(Path("report.html"))
    chunks2 = await pipeline.run(Path("report.html"))
    assert chunks1[0].id == chunks2[0].id


@pytest.mark.anyio
async def test_pipeline_run_propagates_parser_error(error_parser):
    pipeline = DocumentPipeline(parser=error_parser)
    with pytest.raises(ValueError, match="Cannot parse"):
        await pipeline.run(Path("bad.html"))


# --- run_many() ---


@pytest.mark.anyio
async def test_pipeline_run_many_all_processed(simple_parser, fake_paths):
    pipeline = DocumentPipeline(parser=simple_parser)
    result = await pipeline.run_many(fake_paths)
    assert isinstance(result, PipelineResult)
    assert len(result.chunks) == len(fake_paths)
    assert result.errors == []


@pytest.mark.anyio
async def test_pipeline_run_many_empty_sources(simple_parser):
    pipeline = DocumentPipeline(parser=simple_parser)
    result = await pipeline.run_many([])
    assert result.chunks == []
    assert result.errors == []


@pytest.mark.anyio
async def test_pipeline_run_many_on_error_raise(error_parser, fake_paths):
    pipeline = DocumentPipeline(parser=error_parser, on_error="raise")
    with pytest.raises(ValueError, match="Cannot parse"):
        await pipeline.run_many(fake_paths)


@pytest.mark.anyio
async def test_pipeline_run_many_on_error_skip(error_parser, fake_paths):
    pipeline = DocumentPipeline(parser=error_parser, on_error="skip")
    result = await pipeline.run_many(fake_paths)
    assert result.chunks == []
    assert len(result.errors) == len(fake_paths)
    for path, exc in result.errors:
        assert isinstance(path, Path)
        assert isinstance(exc, ValueError)


@pytest.mark.anyio
async def test_pipeline_run_many_on_error_skip_partial(fake_paths):
    """Only the second path raises; others succeed."""
    call_count = 0

    async def _parse(path: Path) -> Document:
        nonlocal call_count
        call_count += 1
        if path == fake_paths[1]:
            raise ValueError("bad")
        return make_document(title=path.stem)

    pipeline = DocumentPipeline(parser=_parse, on_error="skip")
    result = await pipeline.run_many(fake_paths)
    assert len(result.chunks) == 2
    assert len(result.errors) == 1
    assert result.errors[0][0] == fake_paths[1]


@pytest.mark.anyio
async def test_pipeline_run_many_all_files_processed_once(fake_paths):
    """Each file is processed exactly once."""
    calls: list[Path] = []

    async def _parse(path: Path) -> Document:
        calls.append(path)
        return make_document(title=path.stem)

    pipeline = DocumentPipeline(parser=_parse)
    await pipeline.run_many(fake_paths)
    assert sorted(calls) == sorted(fake_paths)


# --- stream() ---


@pytest.mark.anyio
async def test_pipeline_stream_yields_one_batch_per_file(simple_parser, fake_paths):
    pipeline = DocumentPipeline(parser=simple_parser)
    batches: list[list] = []
    async for batch in pipeline.stream(fake_paths):
        batches.append(batch)
    assert len(batches) == len(fake_paths)
    for batch in batches:
        assert len(batch) >= 1


@pytest.mark.anyio
async def test_pipeline_stream_total_chunks_match_run_many(simple_parser, fake_paths):
    pipeline = DocumentPipeline(parser=simple_parser)
    streamed: list = []
    async for batch in pipeline.stream(fake_paths):
        streamed.extend(batch)

    result = await pipeline.run_many(fake_paths)
    assert len(streamed) == len(result.chunks)


@pytest.mark.anyio
async def test_pipeline_stream_empty_sources(simple_parser):
    pipeline = DocumentPipeline(parser=simple_parser)
    batches = [b async for b in pipeline.stream([])]
    assert batches == []


# --- AutoParser ---


@pytest.mark.anyio
async def test_auto_parser_unsupported_extension_raises():
    from ragdoc.pipeline.parser import AutoParser

    parser = AutoParser()
    with pytest.raises(ValueError, match="No parser registered"):
        await parser(Path("document.pdf2"))
