"""
Integration tests for the MinerU CoreExtractionMiddleware pipeline.

These tests build minimal MinerUMiddleDocument fixtures in-memory and verify
end-to-end pipeline behaviour: block dispatch order, caption placement,
metadata propagation, and image loading.
"""

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

import asyncio

from ragdoc.document import Heading, Image, Paragraph, RawText, Table
from ragdoc.parsing.mineru.base import (
    CodeBlock,
    CodeBodyBlock,
    CodeCaptionBlock,
    DiscardedBlock,
    DiscardedBlockType,
    ImageBlock,
    ImageBodyBlock,
    ImageCaptionBlock,
    ImageSpan,
    Line,
    MinerUMiddleDocument,
    PageInfo,
    TableBlock,
    TableBodyBlock,
    TableCaptionBlock,
    TableFootnoteBlock,
    TableSpan,
    TextBlock,
    TextSpan,
    TitleBlock,
)
from ragdoc.parsing.mineru.handlers import (
    ExtractionConfig,
    handle_discarded_as_raw_text,
)
from ragdoc.parsing.mineru.parser import CoreExtractionMiddleware, MinerUExtractor

# ---------------------------------------------------------------------------
# Fixture factories (shared with test_handlers.py but duplicated to keep files independent)
# ---------------------------------------------------------------------------


def _line(text: str, y1: float = 0.0, y2: float = 12.0) -> Line:
    bbox: list[float] = [0.0, y1, 100.0, y2]
    return Line(bbox=bbox, spans=[TextSpan(bbox=bbox, content=text)])


def _table_span(html: str) -> TableSpan:
    return TableSpan(bbox=[0.0, 0.0, 100.0, 50.0], html=html)


# Module-level asyncio.run() — acceptable: builds rich fixtures for parametrize at import time.
def _parse_doc(mineru_doc: MinerUMiddleDocument, source_path=None):
    extractor = MinerUExtractor()
    return asyncio.run(extractor.parse(mineru_doc, source_path=source_path))


def _single_page_doc(*para_blocks, discarded=None) -> MinerUMiddleDocument:
    page = PageInfo(
        para_blocks=list(para_blocks),
        discarded_blocks=discarded or [],
        page_idx=0,
    )
    return MinerUMiddleDocument(pdf_info=[page])


# ---------------------------------------------------------------------------
# Rich fixture: document with title, table (with caption + footnote), code (with caption)
# ---------------------------------------------------------------------------

_TABLE_HTML = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"

_table_body_block = TableBodyBlock(
    bbox=[0.0, 20.0, 200.0, 80.0],
    index=1,
    lines=[Line(bbox=[0.0, 20.0, 200.0, 80.0], spans=[_table_span(_TABLE_HTML)])],
)
_RICH_DOC = _single_page_doc(
    TitleBlock(
        bbox=[0.0, 0.0, 200.0, 18.0],
        index=0,
        lines=[_line("Results", y1=0.0, y2=18.0)],
    ),
    TableBlock(
        bbox=[0.0, 18.0, 200.0, 100.0],
        index=1,
        blocks=[
            TableCaptionBlock(bbox=[0.0, 18.0, 200.0, 30.0], index=0, lines=[_line("Table 1: Summary")]),
            _table_body_block,
            TableFootnoteBlock(bbox=[0.0, 80.0, 200.0, 90.0], index=2, lines=[_line("* statistically significant")]),
        ],
    ),
    CodeBlock(
        bbox=[0.0, 100.0, 200.0, 150.0],
        index=2,
        blocks=[
            CodeCaptionBlock(bbox=[0.0, 100.0, 200.0, 112.0], index=0, lines=[_line("Algorithm 1: Pseudocode")]),
            CodeBodyBlock(bbox=[0.0, 112.0, 200.0, 150.0], index=1, lines=[_line("for i in range(n): pass")]),
        ],
    ),
)

_RICH_PARSED = _parse_doc(_RICH_DOC)


# ---------------------------------------------------------------------------
# Tests against the rich fixture
# ---------------------------------------------------------------------------


def test_rich_fixture_has_heading():
    headings = [el for el in _RICH_PARSED.elements if isinstance(el, Heading)]
    assert any("Results" in h.text for h in headings)


def test_table_caption_precedes_table():
    """Caption Paragraph must immediately precede its Table in document order."""
    elements = _RICH_PARSED.elements
    for i, el in enumerate(elements):
        if isinstance(el, Table):
            assert i > 0, "Table has no preceding element"
            preceding = elements[i - 1]
            assert isinstance(preceding, Paragraph), f"Expected Paragraph before Table, got {type(preceding).__name__}"
            assert "Table 1" in preceding.html


def test_table_footnote_follows_table():
    """Footnote RawText must immediately follow its Table in document order."""
    elements = _RICH_PARSED.elements
    for i, el in enumerate(elements):
        if isinstance(el, Table):
            assert i < len(elements) - 1, "Table has no following element"
            following = elements[i + 1]
            assert isinstance(following, RawText), f"Expected RawText after Table, got {type(following).__name__}"
            assert "statistically significant" in following.innerhtml


def test_code_caption_precedes_code_body():
    """Code caption Paragraph must immediately precede the code RawText."""
    elements = _RICH_PARSED.elements
    raw_texts = [(i, el) for i, el in enumerate(elements) if isinstance(el, RawText)]
    # Find the code body (contains "range(n)")
    code_idx = next(
        (i for i, el in raw_texts if "range" in el.innerhtml),
        None,
    )
    assert code_idx is not None, "Code body RawText not found"
    preceding = elements[code_idx - 1]
    assert isinstance(preceding, Paragraph)
    assert "Algorithm 1" in preceding.html


def test_table_html_content_preserved():
    tables = [el for el in _RICH_PARSED.elements if isinstance(el, Table)]
    assert len(tables) == 1
    assert "<th>A</th>" in tables[0].html_content


# ---------------------------------------------------------------------------
# Header metadata pipeline test
# ---------------------------------------------------------------------------

_HEADER_DOC = _single_page_doc(
    TextBlock(
        bbox=[0.0, 0.0, 200.0, 12.0],
        index=0,
        lines=[_line("Body paragraph")],
    ),
    discarded=[
        DiscardedBlock(
            bbox=[0.0, 0.0, 200.0, 10.0],
            index=0,
            type=DiscardedBlockType.HEADER,
            lines=[_line("Confidential — Draft v1")],
        )
    ],
)


def test_header_metadata_not_propagated_to_document_by_default():
    """The default HEADER handler (handle_discarded_as_metadata) accumulates into
    context.metadata only — see its unit tests in test_handlers.py. The parser
    deliberately does NOT inject discarded content into Document.metadata, so
    document metadata stays clean."""
    doc = _parse_doc(_HEADER_DOC)
    assert "headers" not in doc.metadata


def test_header_discarded_block_not_in_elements_by_default():
    """By default (handle_discarded_as_metadata), header text must NOT appear as a document element."""
    doc = _parse_doc(_HEADER_DOC)
    raw_texts = [el for el in doc.elements if isinstance(el, RawText)]
    paragraphs = [el for el in doc.elements if isinstance(el, Paragraph)]
    combined = "".join(el.innerhtml for el in raw_texts) + "".join(el.html for el in paragraphs)
    assert "Confidential" not in combined


def test_header_handler_swapped_to_raw_text():
    """Swapping HEADER handler to handle_discarded_as_raw_text emits a RawText element."""
    config = ExtractionConfig()
    config.discarded_handlers[DiscardedBlockType.HEADER] = handle_discarded_as_raw_text
    extractor = MinerUExtractor(use_default_middlewares=False)
    extractor.use(CoreExtractionMiddleware(config=config))
    doc = asyncio.run(extractor.parse(_HEADER_DOC))
    raw_texts = [el for el in doc.elements if isinstance(el, RawText)]
    assert any("Confidential" in el.innerhtml for el in raw_texts)


# ---------------------------------------------------------------------------
# Image loading integration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_image_loaded_through_pipeline(tmp_path):
    pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image as PILImage

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    PILImage.new("RGB", (20, 15), color=(0, 128, 0)).save(str(img_dir / "chart.png"))

    # MinerU stores the bare filename in image_path; the handler prepends "images/".
    img_span = ImageSpan(bbox=[0.0, 0.0, 200.0, 100.0], image_path="chart.png")
    span_line = Line(bbox=[0.0, 0.0, 200.0, 100.0], spans=[img_span])
    mineru_doc = _single_page_doc(
        ImageBlock(
            bbox=[0.0, 0.0, 200.0, 115.0],
            index=0,
            blocks=[
                ImageBodyBlock(bbox=[0.0, 0.0, 200.0, 100.0], index=0, lines=[span_line]),
                ImageCaptionBlock(bbox=[0.0, 100.0, 200.0, 115.0], index=1, lines=[_line("Fig 1")]),
            ],
        )
    )
    doc = await MinerUExtractor().parse(mineru_doc, source_path=tmp_path / "doc_middle.json")

    images = [el for el in doc.elements if isinstance(el, Image)]
    assert len(images) == 1
    img = images[0]
    assert img.width == 20
    assert img.height == 15
    assert img.image  # non-empty base64

    # Caption follows image
    elements = doc.elements
    img_idx = next(i for i, el in enumerate(elements) if isinstance(el, Image))
    assert img_idx < len(elements) - 1
    assert isinstance(elements[img_idx + 1], Paragraph)
    assert "Fig 1" in elements[img_idx + 1].html


# ---------------------------------------------------------------------------
# parser field and source provenance
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parser_field_set_to_mineru(tmp_path):
    doc = await MinerUExtractor().parse(MinerUMiddleDocument(pdf_info=[]), source_path=tmp_path / "report_middle.json")
    assert doc.parser == "mineru"


@pytest.mark.anyio
async def test_source_path_and_filename_set(tmp_path):
    source = tmp_path / "report_middle.json"
    doc = await MinerUExtractor().parse(MinerUMiddleDocument(pdf_info=[]), source_path=source)
    assert doc.metadata["filename"] == "report_middle.json"
    assert doc.source_path == str(source)
