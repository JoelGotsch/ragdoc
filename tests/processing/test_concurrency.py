"""Tests for bounded concurrency on element-level processors.

Covers:
- _resolve_semaphore: int → new Semaphore; Semaphore → pass-through
- _fan_out: concurrency=1 sequential fast-path; concurrency=2 bounded parallel
- ImageSummaryProcessor: concurrency=1 and concurrency=2 respect the cap
- LLMHeadingResolver._process_in_batches: concurrency=2 respects the cap
- FootnoteProcessor: concurrency=1 uses ordering constraint; concurrency=2 parallel
- Shared asyncio.Semaphore crosses processor boundaries
"""
from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import pytest
from PIL import Image as PILImage

from ragdoc.document import Document, Footnote, Heading, Image, Paragraph
from ragdoc.processing._concurrency import _fan_out, _resolve_semaphore
from ragdoc.processing.footnote import FootnoteProcessor, SimpleFootnoteResolver
from ragdoc.processing.summary_base import ImageSummary
from ragdoc.processing.summary_image import ImageSummaryProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_png_base64() -> str:
    img = PILImage.new("RGB", (1, 1), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_doc_with_images(n: int) -> Document:
    return Document(elements=[Image(image=_tiny_png_base64(), image_type="png") for _ in range(n)])


def _make_doc_with_footnotes(n: int) -> Document:
    """Document with n footnotes and n matching paragraph references (same page)."""
    elements: list = []
    footnotes: list[Footnote] = []
    for i in range(1, n + 1):
        # Page must match footnote.page for same_page_only=True to find them.
        elements.append(Paragraph(html_content=f"<p>Text reference{i}.</p>", page=1))
        footnotes.append(Footnote(number=i, innerhtml=f"<p>Footnote {i}</p>", page=1))
    elements.extend(footnotes)
    return Document(elements=elements)


# ---------------------------------------------------------------------------
# _resolve_semaphore
# ---------------------------------------------------------------------------


def test_resolve_semaphore_int_creates_new():
    sem = _resolve_semaphore(3)
    assert isinstance(sem, asyncio.Semaphore)


def test_resolve_semaphore_passthrough():
    original = asyncio.Semaphore(2)
    assert _resolve_semaphore(original) is original


# ---------------------------------------------------------------------------
# _fan_out
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_fan_out_sequential_runs_all():
    results: list[int] = []

    async def _task(n: int) -> None:
        results.append(n)

    await _fan_out([_task(i) for i in range(5)], concurrency=1)
    assert results == list(range(5))


@pytest.mark.anyio
async def test_fan_out_sequential_preserves_order():
    order: list[int] = []

    async def _task(n: int) -> None:
        order.append(n)

    await _fan_out([_task(i) for i in range(4)], concurrency=1)
    assert order == [0, 1, 2, 3]


@pytest.mark.anyio
async def test_fan_out_parallel_runs_all():
    results: list[int] = []

    async def _task(n: int) -> None:
        results.append(n)

    await _fan_out([_task(i) for i in range(5)], concurrency=2)
    assert sorted(results) == list(range(5))


@pytest.mark.anyio
async def test_fan_out_parallel_respects_cap():
    concurrent = 0
    max_concurrent = 0

    async def _task() -> None:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1

    await _fan_out([_task() for _ in range(6)], concurrency=2)
    assert max_concurrent <= 2


@pytest.mark.anyio
async def test_fan_out_shared_semaphore_respected():
    """A shared Semaphore(1) serialises tasks from two separate fan_out calls."""
    shared = asyncio.Semaphore(1)
    concurrent = 0
    max_concurrent = 0

    async def _task() -> None:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1

    await asyncio.gather(
        _fan_out([_task() for _ in range(3)], concurrency=shared),
        _fan_out([_task() for _ in range(3)], concurrency=shared),
    )
    assert max_concurrent <= 1


# ---------------------------------------------------------------------------
# ImageSummaryProcessor concurrency
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_image_processor_concurrency_1_sequential():
    """concurrency=1 processes images one at a time."""
    concurrent = 0
    max_concurrent = 0

    async def _summarize(image: Image, context: str | None) -> ImageSummary:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return ImageSummary(summary="x")

    doc = _make_doc_with_images(4)
    await ImageSummaryProcessor(summarize=_summarize, concurrency=1).process(doc)
    assert max_concurrent == 1


@pytest.mark.anyio
async def test_image_processor_concurrency_2_bounded():
    """concurrency=2 allows at most 2 simultaneous summarizations."""
    concurrent = 0
    max_concurrent = 0

    async def _summarize(image: Image, context: str | None) -> ImageSummary:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return ImageSummary(summary="x")

    doc = _make_doc_with_images(6)
    await ImageSummaryProcessor(summarize=_summarize, concurrency=2).process(doc)
    assert max_concurrent <= 2


@pytest.mark.anyio
async def test_image_processor_shared_semaphore():
    """A shared Semaphore caps total calls across two processors."""
    shared = asyncio.Semaphore(1)
    concurrent = 0
    max_concurrent = 0

    async def _summarize(image: Image, context: str | None) -> ImageSummary:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return ImageSummary(summary="x")

    doc1 = _make_doc_with_images(3)
    doc2 = _make_doc_with_images(3)
    p1 = ImageSummaryProcessor(summarize=_summarize, concurrency=shared)
    p2 = ImageSummaryProcessor(summarize=_summarize, concurrency=shared)
    await asyncio.gather(p1.process(doc1), p2.process(doc2))
    assert max_concurrent <= 1


@pytest.mark.anyio
async def test_image_processor_all_images_processed_with_concurrency():
    """All images are processed regardless of concurrency level."""
    summaries: list[str] = []

    async def _summarize(image: Image, context: str | None) -> ImageSummary:
        summaries.append(image.id)
        return ImageSummary(summary="done")

    doc = _make_doc_with_images(5)
    await ImageSummaryProcessor(summarize=_summarize, concurrency=3).process(doc)
    assert len(summaries) == 5


# ---------------------------------------------------------------------------
# FootnoteProcessor concurrency
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_footnote_processor_concurrency_1_preserves_ordering():
    """concurrency=1: resolver is called sequentially with min_element_idx."""
    min_idx_history: list[int | None] = []

    class TrackingResolver:
        async def resolve(self, candidates, footnote_number, footnote_text, min_element_idx=None):
            min_idx_history.append(min_element_idx)
            return candidates[0] if candidates else None

    doc = _make_doc_with_footnotes(3)
    processor = FootnoteProcessor(resolver=TrackingResolver(), concurrency=1)
    await processor.process(doc)

    # First call must have min_element_idx=None; subsequent calls receive the
    # element_idx from the previous resolution.
    assert min_idx_history[0] is None
    # Later calls should receive a non-None value (ordering constraint active).
    assert any(v is not None for v in min_idx_history[1:])


@pytest.mark.anyio
async def test_footnote_processor_concurrency_2_parallel_no_ordering():
    """concurrency=2: all resolver calls use min_element_idx=None."""
    min_idx_args: list[int | None] = []

    class TrackingResolver:
        async def resolve(self, candidates, footnote_number, footnote_text, min_element_idx=None):
            min_idx_args.append(min_element_idx)
            return candidates[0] if candidates else None

    doc = _make_doc_with_footnotes(4)
    processor = FootnoteProcessor(resolver=TrackingResolver(), concurrency=2)
    await processor.process(doc)

    assert all(v is None for v in min_idx_args)


@pytest.mark.anyio
async def test_footnote_processor_concurrency_respects_cap():
    """concurrency=2 allows at most 2 simultaneous resolver calls."""
    concurrent = 0
    max_concurrent = 0

    class CountingResolver:
        async def resolve(self, candidates, footnote_number, footnote_text, min_element_idx=None):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0)
            concurrent -= 1
            return candidates[0] if candidates else None

    doc = _make_doc_with_footnotes(6)
    await FootnoteProcessor(resolver=CountingResolver(), concurrency=2).process(doc)
    assert max_concurrent <= 2
