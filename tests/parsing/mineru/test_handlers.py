"""
Unit tests for MinerU handler functions.

Each handler converts a single MinerU block into zero or more ParsedElement
objects.  Tests use minimal in-memory fixtures — no I/O, no real JSON files.
"""
import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

from pathlib import Path

from ragdoc.document import DocumentList, Footnote, Heading, Image, Paragraph, RawText, Table
from ragdoc.parsing.mineru.base import (
    ChartBlock,
    ChartBodyBlock,
    ChartCaptionBlock,
    ChartFootnoteBlock,
    ChartSpan,
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
    ListBlock,
    ListItemBlock,
    MinerUMiddleDocument,
    PageInfo,
    ParseContext,
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
    handle_chart_block,
    handle_code_block,
    handle_discarded_as_footnote,
    handle_discarded_as_metadata,
    handle_discarded_as_raw_text,
    handle_discarded_drop,
    handle_image_block,
    handle_list_block,
    handle_table_block,
    handle_text_block,
    handle_title_block,
)


# ---------------------------------------------------------------------------
# Minimal fixture factories
# ---------------------------------------------------------------------------


def _line(text: str, y1: float = 0.0, y2: float = 12.0) -> Line:
    bbox: list[float] = [0.0, y1, 100.0, y2]
    return Line(bbox=bbox, spans=[TextSpan(bbox=bbox, content=text)])


def _page(page_idx: int = 0, page_bbox: list[float] | None = None) -> PageInfo:
    """Return a minimal PageInfo.  Pass page_bbox to define page extents via a dummy block."""
    if page_bbox is not None:
        dummy = DiscardedBlock(
            bbox=page_bbox,
            index=99,
            type=DiscardedBlockType.PAGE_NUMBER,
            lines=[],
        )
        return PageInfo(para_blocks=[], discarded_blocks=[dummy], page_idx=page_idx)
    return PageInfo(para_blocks=[], discarded_blocks=[], page_idx=page_idx)


def _context(**metadata) -> ParseContext:
    ctx = ParseContext(source=MinerUMiddleDocument(pdf_info=[]))
    ctx.metadata.update(metadata)
    return ctx


def _discarded(
    text: str, block_type: DiscardedBlockType = DiscardedBlockType.PAGE_FOOTNOTE
) -> DiscardedBlock:
    return DiscardedBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, type=block_type, lines=[_line(text)])


# ---------------------------------------------------------------------------
# handle_title_block
# ---------------------------------------------------------------------------


def test_handle_title_block_empty_text_returns_empty():
    block = TitleBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("")])
    assert handle_title_block(block, _page(), _context()) == []


def test_handle_title_block_produces_heading():
    block = TitleBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("Introduction")])
    result = handle_title_block(block, _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, Heading)
    assert "Introduction" in result[0].element.html


def test_handle_title_block_default_heading_level_is_1():
    block = TitleBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("Title")])
    result = handle_title_block(block, _page(), _context())
    assert "<h1" in result[0].element.html


def test_handle_title_block_heading_level_from_context():
    block = TitleBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("Chapter")])
    result = handle_title_block(block, _page(), _context(_default_heading_level=3))
    assert "<h3" in result[0].element.html


def test_handle_title_block_css_font_size_from_line_height():
    # Line spans from y=0 to y=18 → average_line_height = 18.0
    block = TitleBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("Big", y1=0.0, y2=18.0)])
    result = handle_title_block(block, _page(), _context())
    assert "font-size: 18.0pt" in result[0].element.html


def test_handle_title_block_css_text_align_center_when_centred():
    # Page spans [0,0,200,300]; block center x = (40+160)/2 = 100 = page center
    page = _page(page_bbox=[0.0, 0.0, 200.0, 300.0])
    block = TitleBlock(bbox=[40.0, 100.0, 160.0, 120.0], index=0, lines=[_line("Centred")])
    result = handle_title_block(block, page, _context())
    assert "text-align: center" in result[0].element.html


def test_handle_title_block_no_center_css_when_not_centred():
    page = _page(page_bbox=[0.0, 0.0, 200.0, 300.0])
    # Block far left: center x = 20, page center = 100
    block = TitleBlock(bbox=[0.0, 100.0, 40.0, 120.0], index=0, lines=[_line("Left")])
    result = handle_title_block(block, page, _context())
    assert "text-align: center" not in result[0].element.html


def test_handle_title_block_page_number_is_one_based():
    block = TitleBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("Title")])
    result = handle_title_block(block, _page(page_idx=4), _context())
    assert result[0].element.page == 5


# ---------------------------------------------------------------------------
# handle_text_block
# ---------------------------------------------------------------------------


def test_handle_text_block_empty_returns_empty():
    block = TextBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("")])
    assert handle_text_block(block, _page(), _context()) == []


def test_handle_text_block_produces_paragraph():
    block = TextBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("Hello world")])
    result = handle_text_block(block, _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, Paragraph)
    assert "Hello world" in result[0].element.html


def test_handle_text_block_paragraph_in_p_tags():
    block = TextBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("content")])
    html = handle_text_block(block, _page(), _context())[0].element.html
    assert html.startswith("<p") and html.endswith("</p>")


# ---------------------------------------------------------------------------
# handle_text_block — CSS (Phase 2)
# ---------------------------------------------------------------------------


def test_handle_text_block_css_font_size_from_line_height():
    block = TextBlock(bbox=[0.0, 0.0, 100.0, 20.0], index=0, lines=[_line("text", y1=0.0, y2=14.0)])
    html = handle_text_block(block, _page(), _context())[0].element.html
    assert "font-size: 14.0pt" in html


def test_handle_text_block_css_text_align_center_when_centred():
    page = _page(page_bbox=[0.0, 0.0, 200.0, 300.0])
    block = TextBlock(bbox=[40.0, 50.0, 160.0, 65.0], index=0, lines=[_line("centred")])
    html = handle_text_block(block, page, _context())[0].element.html
    assert "text-align: center" in html


def test_handle_text_block_no_center_css_when_not_centred():
    page = _page(page_bbox=[0.0, 0.0, 200.0, 300.0])
    block = TextBlock(bbox=[0.0, 50.0, 40.0, 65.0], index=0, lines=[_line("left")])
    html = handle_text_block(block, page, _context())[0].element.html
    assert "text-align: center" not in html


# ---------------------------------------------------------------------------
# handle_list_block
# ---------------------------------------------------------------------------


def _list_block(items: list[str]) -> ListBlock:
    item_blocks = [
        ListItemBlock(bbox=[0.0, i * 15.0, 100.0, (i + 1) * 15.0], index=i, lines=[_line(t)])
        for i, t in enumerate(items)
    ]
    return ListBlock(bbox=[0.0, 0.0, 100.0, len(items) * 15.0], index=0, blocks=item_blocks)


def test_handle_list_block_produces_document_list():
    result = handle_list_block(_list_block(["apple", "banana"]), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, DocumentList)


def test_handle_list_block_items_in_ul_li():
    html = handle_list_block(_list_block(["apple", "banana"]), _page(), _context())[0].element.html
    assert "<ul" in html
    assert "apple" in html
    assert "banana" in html
    assert "<li>" in html


# ---------------------------------------------------------------------------
# handle_list_block — CSS (Phase 2)
# ---------------------------------------------------------------------------


def test_handle_list_block_css_font_size_from_item_line_height():
    block = _list_block(["item"])
    # Item has line from y=0 to y=16
    block.blocks[0].lines = [_line("item", y1=0.0, y2=16.0)]
    html = handle_list_block(block, _page(), _context())[0].element.html
    assert "font-size:" in html


def test_handle_list_block_css_text_align_center_when_centred():
    page = _page(page_bbox=[0.0, 0.0, 200.0, 300.0])
    item_blocks = [
        ListItemBlock(bbox=[40.0, 0.0, 160.0, 15.0], index=0, lines=[_line("centred item")])
    ]
    block = ListBlock(bbox=[40.0, 0.0, 160.0, 15.0], index=0, blocks=item_blocks)
    html = handle_list_block(block, page, _context())[0].element.html
    assert "text-align: center" in html


# ---------------------------------------------------------------------------
# handle_code_block
# ---------------------------------------------------------------------------


def _code_block(caption: str | None = None, body: str | None = None) -> CodeBlock:
    blocks = []
    if caption is not None:
        blocks.append(
            CodeCaptionBlock(bbox=[0.0, 0.0, 100.0, 15.0], index=0, lines=[_line(caption)])
        )
    if body is not None:
        blocks.append(
            CodeBodyBlock(bbox=[0.0, 15.0, 100.0, 50.0], index=1, lines=[_line(body)])
        )
    return CodeBlock(bbox=[0.0, 0.0, 100.0, 50.0], index=0, blocks=blocks)


def test_handle_code_block_body_only_produces_raw_text():
    result = handle_code_block(_code_block(body="x = 1"), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, RawText)


def test_handle_code_block_caption_only_produces_paragraph():
    result = handle_code_block(_code_block(caption="Algorithm 1"), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, Paragraph)
    assert "Algorithm 1" in result[0].element.html


def test_handle_code_block_caption_before_body():
    result = handle_code_block(_code_block(caption="Alg 1", body="x=1"), _page(), _context())
    assert len(result) == 2
    assert isinstance(result[0].element, Paragraph)
    assert isinstance(result[1].element, RawText)


def test_handle_code_block_empty_body_skipped():
    result = handle_code_block(_code_block(caption="Alg", body=""), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, Paragraph)


def test_handle_code_block_empty_caption_skipped():
    result = handle_code_block(_code_block(caption="", body="x=1"), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, RawText)


def test_handle_code_block_no_blocks_returns_empty():
    assert handle_code_block(_code_block(), _page(), _context()) == []


# ---------------------------------------------------------------------------
# handle_table_block
# ---------------------------------------------------------------------------


def _table_block(
    caption: str | None = None,
    body_html: str | None = None,
    footnote: str | None = None,
) -> TableBlock:
    blocks = []
    if caption is not None:
        blocks.append(
            TableCaptionBlock(bbox=[0.0, 0.0, 100.0, 15.0], index=0, lines=[_line(caption)])
        )
    if body_html is not None:
        span = TableSpan(bbox=[0.0, 15.0, 100.0, 50.0], html=body_html)
        span_line = Line(bbox=[0.0, 15.0, 100.0, 50.0], spans=[span])
        blocks.append(TableBodyBlock(bbox=[0.0, 15.0, 100.0, 50.0], index=1, lines=[span_line]))
    if footnote is not None:
        blocks.append(
            TableFootnoteBlock(bbox=[0.0, 50.0, 100.0, 60.0], index=2, lines=[_line(footnote)])
        )
    return TableBlock(bbox=[0.0, 0.0, 100.0, 60.0], index=0, blocks=blocks)


def test_handle_table_block_caption_produces_paragraph():
    result = handle_table_block(_table_block(caption="Table 1: Results"), _page(), _context())
    assert any(isinstance(r.element, Paragraph) and "Table 1" in r.element.html for r in result)


def test_handle_table_block_body_produces_table():
    result = handle_table_block(
        _table_block(body_html="<table><tr><td>A</td></tr></table>"), _page(), _context()
    )
    assert any(isinstance(r.element, Table) for r in result)


def test_handle_table_block_footnote_produces_raw_text():
    result = handle_table_block(
        _table_block(body_html="<table/>", footnote="* p < 0.05"), _page(), _context()
    )
    assert any(isinstance(r.element, RawText) and "p < 0.05" in r.element.innerhtml for r in result)


def test_handle_table_block_order_caption_table_footnote():
    result = handle_table_block(
        _table_block(caption="Table 1", body_html="<table/>", footnote="Note"),
        _page(),
        _context(),
    )
    assert len(result) == 3
    assert isinstance(result[0].element, Paragraph)
    assert isinstance(result[1].element, Table)
    assert isinstance(result[2].element, RawText)


def test_handle_table_block_empty_body_html_produces_no_table():
    result = handle_table_block(_table_block(body_html="  "), _page(), _context())
    assert not any(isinstance(r.element, Table) for r in result)


def test_handle_table_block_no_components_returns_empty():
    assert handle_table_block(_table_block(), _page(), _context()) == []


# ---------------------------------------------------------------------------
# handle_chart_block
#
# Regression coverage for charts being silently dropped: a ChartBlock carries
# ChartBodyBlock/ChartSpan (markdown in ``.content``), not TableBodyBlock/
# TableSpan (HTML in ``.html``).  Routing charts through handle_table_block
# matched nothing and returned [], so every chart vanished without error.
# ---------------------------------------------------------------------------


def _chart_block(
    caption: str | None = None,
    body_markdown: str | None = None,
    footnote: str | None = None,
) -> ChartBlock:
    blocks = []
    if caption is not None:
        blocks.append(
            ChartCaptionBlock(bbox=[0.0, 0.0, 100.0, 15.0], index=0, lines=[_line(caption)])
        )
    if body_markdown is not None:
        span = ChartSpan(bbox=[0.0, 15.0, 100.0, 50.0], content=body_markdown)
        span_line = Line(bbox=[0.0, 15.0, 100.0, 50.0], spans=[span])
        blocks.append(ChartBodyBlock(bbox=[0.0, 15.0, 100.0, 50.0], index=1, lines=[span_line]))
    if footnote is not None:
        blocks.append(
            ChartFootnoteBlock(bbox=[0.0, 50.0, 100.0, 60.0], index=2, lines=[_line(footnote)])
        )
    return ChartBlock(bbox=[0.0, 0.0, 100.0, 60.0], index=0, blocks=blocks)


def test_handle_chart_block_caption_produces_paragraph():
    result = handle_chart_block(_chart_block(caption="Figure 1: Revenue"), _page(), _context())
    assert any(isinstance(r.element, Paragraph) and "Figure 1" in r.element.html for r in result)


def test_handle_chart_block_markdown_body_produces_table_with_converted_html():
    """The chart's markdown body must be converted to HTML and emitted as a Table.

    This is the core of the fix: tables carry HTML directly, but charts carry
    markdown that must go through pandoc first.
    """
    result = handle_chart_block(
        _chart_block(body_markdown="| Quarter | Revenue |\n|---|---|\n| Q4 | 100 |"),
        _page(),
        _context(),
    )
    tables = [r.element for r in result if isinstance(r.element, Table)]
    assert len(tables) == 1
    # markdown pipe-table converted to an HTML <table>, not left as raw markdown
    assert "<table" in tables[0].html_content
    assert "Revenue" in tables[0].html_content
    assert "|" not in tables[0].html_content


def test_handle_chart_block_footnote_produces_raw_text():
    result = handle_chart_block(
        _chart_block(body_markdown="| A |\n|---|\n| 1 |", footnote="Source: 10-K"),
        _page(),
        _context(),
    )
    assert any(isinstance(r.element, RawText) and "10-K" in r.element.innerhtml for r in result)


def test_handle_chart_block_order_caption_table_footnote():
    result = handle_chart_block(
        _chart_block(caption="Figure 1", body_markdown="| A |\n|---|\n| 1 |", footnote="Note"),
        _page(),
        _context(),
    )
    assert len(result) == 3
    assert isinstance(result[0].element, Paragraph)
    assert isinstance(result[1].element, Table)
    assert isinstance(result[2].element, RawText)


def test_handle_chart_block_empty_body_content_produces_no_table():
    result = handle_chart_block(_chart_block(body_markdown="  "), _page(), _context())
    assert not any(isinstance(r.element, Table) for r in result)


def test_handle_chart_block_no_components_returns_empty():
    assert handle_chart_block(_chart_block(), _page(), _context()) == []


# ---------------------------------------------------------------------------
# handle_image_block
# ---------------------------------------------------------------------------


def _image_block(image_path: str | None = None, caption: str | None = None) -> ImageBlock:
    blocks = []
    if image_path is not None:
        img_span = ImageSpan(bbox=[0.0, 0.0, 100.0, 80.0], image_path=image_path)
        span_line = Line(bbox=[0.0, 0.0, 100.0, 80.0], spans=[img_span])
        blocks.append(ImageBodyBlock(bbox=[0.0, 0.0, 100.0, 80.0], index=0, lines=[span_line]))
    if caption is not None:
        blocks.append(
            ImageCaptionBlock(bbox=[0.0, 80.0, 100.0, 95.0], index=1, lines=[_line(caption)])
        )
    return ImageBlock(bbox=[0.0, 0.0, 100.0, 95.0], index=0, blocks=blocks)


def test_handle_image_block_no_body_no_caption_returns_empty():
    assert handle_image_block(_image_block(), _page(), _context()) == []


def test_handle_image_block_missing_source_dir_adds_warning():
    ctx = _context()
    handle_image_block(_image_block(image_path="img/fig1.png"), _page(), ctx)
    assert any("source_dir" in w for w in ctx.warnings)


def test_handle_image_block_missing_source_dir_produces_no_image():
    result = handle_image_block(_image_block(image_path="img/fig1.png"), _page(), _context())
    assert not any(isinstance(r.element, Image) for r in result)


def test_handle_image_block_caption_only_produces_paragraph():
    result = handle_image_block(_image_block(caption="Figure 1: Overview"), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, Paragraph)
    assert "Figure 1" in result[0].element.html


def test_handle_image_block_loads_image_with_pillow(tmp_path):
    pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image as PILImage

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "fig1.png").write_bytes(
        PILImage.new("RGB", (10, 10)).tobytes()
    )
    # Save properly
    PILImage.new("RGB", (10, 10), color=(255, 0, 0)).save(str(img_dir / "fig1.png"))

    ctx = _context()
    ctx.metadata["source_dir"] = tmp_path
    result = handle_image_block(_image_block(image_path="fig1.png"), _page(), ctx)

    images = [r for r in result if isinstance(r.element, Image)]
    assert len(images) == 1
    img = images[0].element
    assert img.image  # non-empty base64 string
    assert img.width == 10
    assert img.height == 10


def test_handle_image_block_missing_file_produces_no_image(tmp_path):
    pytest.importorskip("PIL", reason="Pillow not installed")
    ctx = _context()
    ctx.metadata["source_dir"] = tmp_path  # dir exists, file does not
    result = handle_image_block(_image_block(image_path="nonexistent.png"), _page(), ctx)
    assert not any(isinstance(r.element, Image) for r in result)
    assert any("not found" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# handle_discarded_as_footnote
# ---------------------------------------------------------------------------


def test_handle_discarded_as_footnote_numbered_produces_footnote():
    result = handle_discarded_as_footnote(_discarded("3 This is footnote text"), _page(), _context())
    assert len(result) == 1
    fn = result[0].element
    assert isinstance(fn, Footnote)
    assert fn.number == 3
    assert "This is footnote text" in fn.innerhtml


def test_handle_discarded_as_footnote_large_number():
    result = handle_discarded_as_footnote(
        _discarded("12 Long footnote with number twelve"), _page(), _context()
    )
    assert isinstance(result[0].element, Footnote)
    assert result[0].element.number == 12


def test_handle_discarded_as_footnote_unnumbered_adds_warning():
    ctx = _context()
    result = handle_discarded_as_footnote(_discarded("Not numbered text"), _page(), ctx)
    assert result == []
    assert len(ctx.warnings) == 1


def test_handle_discarded_as_footnote_empty_text_adds_warning():
    ctx = _context()
    result = handle_discarded_as_footnote(_discarded(""), _page(), ctx)
    assert result == []
    assert len(ctx.warnings) == 1


# ---------------------------------------------------------------------------
# handle_discarded_as_raw_text
# ---------------------------------------------------------------------------


def test_handle_discarded_as_raw_text_produces_raw_text():
    result = handle_discarded_as_raw_text(_discarded("Header text"), _page(), _context())
    assert len(result) == 1
    assert isinstance(result[0].element, RawText)
    assert "Header text" in result[0].element.innerhtml


def test_handle_discarded_as_raw_text_empty_returns_empty():
    assert handle_discarded_as_raw_text(_discarded(""), _page(), _context()) == []


# ---------------------------------------------------------------------------
# handle_discarded_as_metadata
# ---------------------------------------------------------------------------


def test_handle_discarded_as_metadata_stores_in_context():
    ctx = _context()
    handle_discarded_as_metadata(_discarded("Page Header", DiscardedBlockType.HEADER), _page(), ctx)
    assert ctx.metadata["headers"][0]["text"] == "Page Header"
    assert ctx.metadata["headers"][0]["page"] == 1


def test_handle_discarded_as_metadata_produces_no_elements():
    ctx = _context()
    result = handle_discarded_as_metadata(
        _discarded("Page Header", DiscardedBlockType.HEADER), _page(), ctx
    )
    assert result == []


def test_handle_discarded_as_metadata_accumulates_across_pages():
    ctx = _context()
    handle_discarded_as_metadata(
        _discarded("Header A", DiscardedBlockType.HEADER), _page(page_idx=0), ctx
    )
    handle_discarded_as_metadata(
        _discarded("Header B", DiscardedBlockType.HEADER), _page(page_idx=1), ctx
    )
    assert len(ctx.metadata["headers"]) == 2
    assert ctx.metadata["headers"][1]["page"] == 2


def test_handle_discarded_as_metadata_empty_text_skipped():
    ctx = _context()
    handle_discarded_as_metadata(_discarded("", DiscardedBlockType.HEADER), _page(), ctx)
    assert "headers" not in ctx.metadata


def test_handle_discarded_as_metadata_footer_key():
    ctx = _context()
    handle_discarded_as_metadata(_discarded("© 2024 Corp", DiscardedBlockType.FOOTER), _page(), ctx)
    assert "footers" in ctx.metadata


# ---------------------------------------------------------------------------
# handle_discarded_drop
# ---------------------------------------------------------------------------


def test_handle_discarded_drop_always_returns_empty():
    for block_type in DiscardedBlockType:
        result = handle_discarded_drop(_discarded("anything", block_type), _page(), _context())
        assert result == [], f"Expected [] for {block_type}"
