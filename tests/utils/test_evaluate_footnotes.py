"""Tests for the evaluate_footnotes repo script (scripts/, not shipped in the wheel)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from scripts.evaluate_footnotes import evaluate_footnotes

from ragdoc.document import Document, Footnote, Paragraph
from ragdoc.processing import DocumentProcessor

# =============================================================================
# Helpers
# =============================================================================


def _doc_all_resolved() -> Document:
    return Document(
        elements=[
            Paragraph(html='<p>See <ref id="fn-1" rel="footnote"/>.</p>'),
            Footnote(id="fn-1", number=1, innerhtml="Referenced footnote."),
        ]
    )


def _doc_with_orphans() -> Document:
    return Document(
        elements=[
            Paragraph(html='<p>See <ref id="fn-1" rel="footnote"/>.</p>'),
            Footnote(id="fn-1", number=1, innerhtml="Referenced footnote."),
            Footnote(id="fn-2", number=2, innerhtml="Orphaned footnote."),
            Footnote(id="fn-3", number=3, innerhtml="Also orphaned."),
        ]
    )


# --- TestEvaluateFootnotes ---


@pytest.mark.anyio
async def test_evaluate_footnotes_empty_paths_returns_empty_list():
    assert await evaluate_footnotes([]) == []


@pytest.mark.anyio
async def test_evaluate_footnotes_all_resolved_gives_zero_orphans():
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = _doc_all_resolved()
        results = await evaluate_footnotes([Path("file.docx")], processors=[])

    assert len(results) == 1
    r = results[0]
    assert r.orphan_count == 0
    assert r.total_footnotes == 1
    assert r.orphan_numbers == []
    assert r.error is None


@pytest.mark.anyio
async def test_evaluate_footnotes_orphaned_counted_correctly():
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = _doc_with_orphans()
        results = await evaluate_footnotes([Path("file.docx")], processors=[])

    r = results[0]
    assert r.total_footnotes == 3
    assert r.orphan_count == 2
    assert r.orphan_numbers == [2, 3]


@pytest.mark.anyio
async def test_evaluate_footnotes_file_error_captured_in_result():
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.side_effect = ValueError("unsupported file type")
        results = await evaluate_footnotes([Path("file.xyz")], processors=[])

    r = results[0]
    assert r.error is not None
    assert "unsupported file type" in r.error
    assert r.total_footnotes == 0
    assert r.orphan_count == 0
    assert r.orphan_numbers == []


@pytest.mark.anyio
async def test_evaluate_footnotes_multiple_paths_one_result_each():
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.side_effect = [_doc_all_resolved(), _doc_with_orphans()]
        results = await evaluate_footnotes(
            [Path("a.docx"), Path("b.docx")],
            processors=[],
        )

    assert len(results) == 2
    assert results[0].orphan_count == 0
    assert results[1].orphan_count == 2


@pytest.mark.anyio
async def test_evaluate_footnotes_path_preserved_in_result():
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = Document(elements=[])
        p = Path("some/path/doc.docx")
        results = await evaluate_footnotes([p], processors=[])

    assert results[0].path == p


@pytest.mark.anyio
async def test_evaluate_footnotes_string_paths_accepted():
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = Document(elements=[])
        results = await evaluate_footnotes(["file.docx"], processors=[])

    assert len(results) == 1
    assert results[0].path == Path("file.docx")


@pytest.mark.anyio
async def test_evaluate_footnotes_custom_processor_is_called():
    class _Recorder(DocumentProcessor):
        def __init__(self):
            self.called = False

        async def process(self, document: Document) -> Document:
            self.called = True
            return document

    recorder = _Recorder()
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = _doc_all_resolved()
        await evaluate_footnotes([Path("file.docx")], processors=[recorder])

    assert recorder.called


@pytest.mark.anyio
async def test_evaluate_footnotes_no_processors_argument_uses_defaults():
    """Passing processors=None (default) must not raise and must return a result."""
    with patch("scripts.evaluate_footnotes.load", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = Document(elements=[])
        results = await evaluate_footnotes([Path("file.docx")])

    assert len(results) == 1
    assert results[0].error is None
