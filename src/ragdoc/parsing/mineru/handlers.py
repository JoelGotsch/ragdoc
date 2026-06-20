"""
Standalone handler functions for MinerU block types.

Each handler converts a single MinerU block into zero or more :class:`ParsedElement`
objects.  Handlers are plain functions — no class state — so they can be unit-tested
independently and composed via :class:`ExtractionConfig`.

Typical customisation::

    from ragdoc.parsing.mineru.handlers import (
        ExtractionConfig,
        handle_discarded_as_raw_text,
    )
    from ragdoc.parsing.mineru.base import DiscardedBlockType
    from ragdoc.parsing.mineru.parser import CoreExtractionMiddleware

    config = ExtractionConfig()
    config.discarded_handlers[DiscardedBlockType.HEADER] = handle_discarded_as_raw_text
    middleware = CoreExtractionMiddleware(config=config)
"""

from __future__ import annotations
import base64
import base64
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol

from pypandoc import convert_text as _pandoc_convert

from ragdoc.document import (
    DocumentList,
    Footnote,
    Heading,
    Image,
    Paragraph,
    RawText,
    Table,
)

from .base import (
    ChartBlock,
    ChartBodyBlock,
    ChartCaptionBlock,
    ChartFootnoteBlock,
    ChartSpan,
    CodeBlock,
    DiscardedBlock,
    DiscardedBlockType,
    ImageBlock,
    ImageBodyBlock,
    ImageCaptionBlock,
    ImageSpan,
    ListBlock,
    PageInfo,
    ParseContext,
    ParsedElement,
    TableBlock,
    TableBodyBlock,
    TableCaptionBlock,
    TableFootnoteBlock,
    TableSpan,
    TextBlock,
    TitleBlock,
    extract_text_from_lines,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _HasBBox(Protocol):
    bbox: list[float]


def _get_bounding_box(block: _HasBBox) -> tuple[float, float, float, float]:
    bbox = block.bbox
    return (bbox[0], bbox[1], bbox[2], bbox[3])


# ---------------------------------------------------------------------------
# Shared CSS helpers
# ---------------------------------------------------------------------------


def _css_from_block(
    block: "_HasBBox",
    page: PageInfo,
    lines: "list",
) -> str:
    """Build a CSS style string from block geometry and page context.

    Derives ``font-size`` from the average line height and ``text-align:
    center`` when the block is horizontally centred on the page (within 10 %
    tolerance).  Returns an empty string when no properties apply.
    """
    parts: list[str] = []

    if lines:
        avg_height = sum(line.bbox[3] - line.bbox[1] for line in lines) / len(lines)
        font_size = round(avg_height, 1)
        if font_size > 0:
            parts.append(f"font-size: {font_size}pt")

    page_width = page.page_width
    if page_width > 0:
        block_center_x = (block.bbox[0] + block.bbox[2]) / 2
        page_center_x = page.bounding_box[0] + page_width / 2
        if abs(block_center_x - page_center_x) < page_width * 0.10:
            parts.append("text-align: center")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Heading HTML builder (public so middlewares can reuse it)
# ---------------------------------------------------------------------------


def build_heading_html(text: str, block: TitleBlock, page: PageInfo, level: int) -> str:
    """
    Build heading HTML with inline CSS visual properties.

    Derives ``font-size`` from the block's average line height and detects
    centred alignment from the block's horizontal position relative to the page.
    """
    css = _css_from_block(block, page, block.lines)
    style_attr = f' style="{css}"' if css else ""
    return f"<h{level}{style_attr}>{text}</h{level}>"


# ---------------------------------------------------------------------------
# Para-block handlers
# ---------------------------------------------------------------------------


def handle_title_block(block: TitleBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """Convert a :class:`TitleBlock` to a :class:`Heading` element."""
    text = extract_text_from_lines(block.lines).strip()
    if not text:
        return []
    level: int = context.metadata.get("_default_heading_level", 1)
    heading = Heading(
        html_content=build_heading_html(text, block, page, level),
        bounding_box=_get_bounding_box(block),
        page=page.page_idx + 1,
    )
    return [ParsedElement(element=heading, source_block=block, source_page=page)]


def build_text_html(text: str, block: TextBlock, page: PageInfo) -> str:
    """Build paragraph HTML with inline CSS visual properties.

    Derives ``font-size`` from the average line height and ``text-align:
    center`` when the block is horizontally centred on the page.
    """
    css = _css_from_block(block, page, block.lines)
    style_attr = f' style="{css}"' if css else ""
    return f"<p{style_attr}>{text}</p>"


def handle_text_block(block: TextBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """Convert a :class:`TextBlock` to a :class:`Paragraph` element.

    The ``<p>`` tag receives inline CSS via :func:`build_text_html`:
    ``font-size`` derived from the average line height, and ``text-align:
    center`` when the block is horizontally centred on the page.
    """
    text = extract_text_from_lines(block.lines).strip()
    if not text:
        return []
    paragraph = Paragraph(
        html=build_text_html(text, block, page),
        bounding_box=_get_bounding_box(block),
        page=page.page_idx + 1,
    )
    return [ParsedElement(element=paragraph, source_block=block, source_page=page)]


def handle_list_block(block: ListBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """Convert a :class:`ListBlock` to a :class:`DocumentList` element.

    The ``<ul>`` tag receives inline CSS: ``font-size`` derived from the
    average line height across all list items, and ``text-align: center``
    when the block is horizontally centred on the page.
    """
    items = [
        f"<li>{extract_text_from_lines(item_block.lines).strip()}</li>"
        for item_block in block.blocks
    ]
    all_lines = [line for item in block.blocks for line in item.lines]
    css = _css_from_block(block, page, all_lines)
    style_attr = f' style="{css}"' if css else ""
    doc_list = DocumentList(
        html=f"<ul{style_attr}>{''.join(items)}</ul>",
        bounding_box=_get_bounding_box(block),
        page=page.page_idx + 1,
    )
    return [ParsedElement(element=doc_list, source_block=block, source_page=page)]


def handle_code_block(block: CodeBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """
    Convert a :class:`CodeBlock` to zero or more elements.

    Emits an optional :class:`Paragraph` for the caption followed by a
    :class:`RawText` for the code body.
    """
    results: list[ParsedElement] = []
    page_number = page.page_idx + 1

    if block.code_caption is not None:
        caption_text = extract_text_from_lines(block.code_caption.lines).strip()
        if caption_text:
            results.append(ParsedElement(
                element=Paragraph(
                    html=f"<p>{caption_text}</p>",
                    bounding_box=_get_bounding_box(block.code_caption),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    if block.code_body is not None:
        code_text = extract_text_from_lines(block.code_body.lines)
        if code_text.strip():
            results.append(ParsedElement(
                element=RawText(
                    innerhtml=code_text,
                    bounding_box=_get_bounding_box(block),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    return results


def handle_table_block(block: TableBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """
    Convert a :class:`TableBlock` to zero or more elements.

    Emits an optional :class:`Paragraph` for the caption, then the :class:`Table`,
    then an optional :class:`RawText` for any table-level footnote.
    """
    results: list[ParsedElement] = []
    page_number = page.page_idx + 1

    table_caption = next((b for b in block.blocks if isinstance(b, TableCaptionBlock)), None)
    if table_caption is not None:
        caption_text = extract_text_from_lines(table_caption.lines).strip()
        if caption_text:
            results.append(ParsedElement(
                element=Paragraph(
                    html=f"<p>{caption_text}</p>",
                    bounding_box=_get_bounding_box(table_caption),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    table_body = next((b for b in block.blocks if isinstance(b, TableBodyBlock)), None)
    if table_body is not None:
        table_span = next(
            (
                span
                for line in table_body.lines
                for span in line.spans
                if isinstance(span, TableSpan)
            ),
            None,
        )
        if table_span is not None and table_span.html.strip():
            results.append(ParsedElement(
                element=Table(
                    html_content=table_span.html.strip(),
                    bounding_box=_get_bounding_box(block),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    table_footnote = next((b for b in block.blocks if isinstance(b, TableFootnoteBlock)), None)
    if table_footnote is not None:
        fn_text = extract_text_from_lines(table_footnote.lines).strip()
        if fn_text:
            results.append(ParsedElement(
                element=RawText(
                    innerhtml=fn_text,
                    bounding_box=_get_bounding_box(table_footnote),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    return results


def handle_chart_block(block: ChartBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """
    Convert a :class:`ChartBlock` to zero or more elements.

    MinerU v3.1+ emits ``chart`` blocks for figures whose visual content has
    been OCR'd into a markdown table (bar charts, line plots, etc.).  Unlike
    :class:`TableBlock` — which carries HTML in :attr:`TableSpan.html` — a
    chart body stores its data as a markdown string in :attr:`ChartSpan.content`.

    The markdown is converted to HTML via pypandoc (a core dependency) and emitted
    as a :class:`~ragdoc.document.Table`.  If conversion fails for any reason the
    chart body is silently skipped and a ``WARNING`` is logged; caption and footnote
    elements are still emitted.  Captions are emitted as
    :class:`~ragdoc.document.Paragraph` elements and footnotes as
    :class:`~ragdoc.document.RawText`, following the same pattern as
    :func:`handle_table_block`.
    """
    results: list[ParsedElement] = []
    page_number = page.page_idx + 1

    chart_caption = next((b for b in block.blocks if isinstance(b, ChartCaptionBlock)), None)
    if chart_caption is not None:
        caption_text = extract_text_from_lines(chart_caption.lines).strip()
        if caption_text:
            results.append(ParsedElement(
                element=Paragraph(
                    html=f"<p>{caption_text}</p>",
                    bounding_box=_get_bounding_box(chart_caption),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    chart_body = next((b for b in block.blocks if isinstance(b, ChartBodyBlock)), None)
    if chart_body is not None:
        chart_span = next(
            (
                span
                for line in chart_body.lines
                for span in line.spans
                if isinstance(span, ChartSpan)
            ),
            None,
        )
        if chart_span is not None and chart_span.content.strip():
            try:
                html_content = _pandoc_convert(chart_span.content, to="html", format="markdown")
                results.append(ParsedElement(
                    element=Table(
                        html_content=html_content.strip(),
                        bounding_box=_get_bounding_box(block),
                        page=page_number,
                    ),
                    source_block=block,
                    source_page=page,
                ))
            except Exception as exc:
                logger.warning(
                    "Could not convert chart markdown to HTML on page %d (%s: %s); "
                    "chart body skipped. Content prefix: %r",
                    page_number,
                    type(exc).__name__,
                    exc,
                    chart_span.content[:100],
                )

    chart_footnote = next((b for b in block.blocks if isinstance(b, ChartFootnoteBlock)), None)
    if chart_footnote is not None:
        fn_text = extract_text_from_lines(chart_footnote.lines).strip()
        if fn_text:
            results.append(ParsedElement(
                element=RawText(
                    innerhtml=fn_text,
                    bounding_box=_get_bounding_box(chart_footnote),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    return results


def _load_image_bytes(path: Path) -> tuple[str, str, int, int] | None:
    """
    Open *path* with Pillow and return ``(base64_data, format, width, height)``.

    Returns ``None`` and logs a warning on any failure (file not found, corrupt
    image, unsupported format, …).
    """
    try:
        from PIL import Image as PILImage  # noqa: PLC0415 (lazy import for optional dep)

        with PILImage.open(path) as img:
            fmt = (img.format or "png").lower()
            width, height = img.size
            buf = BytesIO()
            img.save(buf, format=img.format or "PNG")
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return data, fmt, width, height
    except Exception as exc:
        logger.warning("Could not load image %s: %s", path, exc)
        return None


def handle_image_block(block: ImageBlock, page: PageInfo, context: ParseContext) -> list[ParsedElement]:
    """
    Convert an :class:`ImageBlock` to zero or more elements.

    Loads the image bytes from disk using Pillow and stores them as base64 in
    :attr:`Image.image` — the same representation used by the HTML and Azure DI
    parsers when downloading images from HTTP sources.

    MineRU always writes image files into an ``images/`` subdirectory alongside
    the ``_middle.json`` file, while ``image_path`` in the JSON stores only the
    bare filename (e.g. ``4ab1fbb6….jpg``).  The resolved path is therefore
    ``<source_dir>/images/<image_path>``.

    Requires ``context.metadata["source_dir"]`` (a :class:`~pathlib.Path`) to
    resolve the relative ``image_path`` stored by MinerU.  The path is set
    automatically when parsing via :func:`MinerUParser.parse` with a
    ``source_path`` argument.  When it is absent a warning is emitted and no
    :class:`Image` element is produced.

    An optional :class:`Paragraph` for the image caption follows the image.
    """
    results: list[ParsedElement] = []
    page_number = page.page_idx + 1
    source_dir: Path | None = context.metadata.get("source_dir")

    body = block.image_body
    if body is not None:
        image_span = next(
            (
                span
                for line in body.lines
                for span in line.spans
                if isinstance(span, ImageSpan)
            ),
            None,
        )
        if image_span is not None:
            if source_dir is None:
                context.warnings.append(
                    f"Cannot load image on page {page_number}: "
                    "source_dir not set in context.metadata"
                )
            else:
                img_path = source_dir / "images" / image_span.image_path
                if not img_path.exists():
                    context.warnings.append(
                        f"Image file not found on page {page_number}: {img_path}"
                    )
                    loaded = None
                else:
                    loaded = _load_image_bytes(img_path)
                if loaded is not None:
                    b64data, fmt, width, height = loaded
                    results.append(ParsedElement(
                        element=Image(
                            image=b64data,
                            image_type=fmt,
                            width=width,
                            height=height,
                            bounding_box=_get_bounding_box(block),
                            page=page_number,
                        ),
                        source_block=block,
                        source_page=page,
                    ))

    caption = block.image_caption
    if caption is not None:
        caption_text = extract_text_from_lines(caption.lines).strip()
        if caption_text:
            results.append(ParsedElement(
                element=Paragraph(
                    html=f"<p>{caption_text}</p>",
                    bounding_box=_get_bounding_box(caption),
                    page=page_number,
                ),
                source_block=block,
                source_page=page,
            ))

    return results


# ---------------------------------------------------------------------------
# Pre-flight helper
# ---------------------------------------------------------------------------


def check_mineru_images(middle_json_path: Path) -> list[Path]:
    """Return a list of image paths referenced by *middle_json_path* that are absent from disk.

    Each returned path is relative to the directory containing the JSON file
    (e.g. ``images/4ab1fbb6….jpg``).  An empty list means all images are
    present.

    MineRU always writes images into an ``images/`` subdirectory alongside the
    JSON.  This function checks ``<json_dir>/images/<image_path>`` for every
    image span referenced in the file.

    Useful as a pre-flight check before ingesting a batch of MineRU outputs::

        missing = check_mineru_images(path)
        if missing:
            print(f"Re-run MineRU for {path.parent.name}: {len(missing)} images missing")
    """
    from .base import parse_middle_json_file  # local import to avoid circular at module level

    source = parse_middle_json_file(middle_json_path)
    source_dir = Path(middle_json_path).parent
    missing: list[Path] = []
    for page in source.pdf_info:
        for block in page.para_blocks:
            if not isinstance(block, ImageBlock):
                continue
            body = block.image_body
            if body is None:
                continue
            for line in body.lines:
                for span in line.spans:
                    if isinstance(span, ImageSpan):
                        img_path = source_dir / "images" / span.image_path
                        if not img_path.exists():
                            missing.append(Path("images") / span.image_path)
    return missing


# ---------------------------------------------------------------------------
# Discarded-block handlers
# ---------------------------------------------------------------------------

DiscardedBlockHandler = Callable[[DiscardedBlock, PageInfo, ParseContext], list[ParsedElement]]

_NUMBERED_FOOTNOTE_RE = re.compile(r"^(\d+)\s*(.+)$", re.DOTALL)


def handle_discarded_as_footnote(
    block: DiscardedBlock, page: PageInfo, context: ParseContext
) -> list[ParsedElement]:
    """
    Parse a numbered footnote from a discarded block.

    Expects text of the form ``N rest of footnote text``.  If the block does not
    match this pattern a warning is added to *context* and an empty list is
    returned (the block is silently dropped).
    """
    text = extract_text_from_lines(block.lines).strip()
    match = _NUMBERED_FOOTNOTE_RE.match(text)
    if match:
        return [ParsedElement(
            element=Footnote(
                number=int(match.group(1)),
                innerhtml=match.group(2).strip(),
                bounding_box=_get_bounding_box(block),
                page=page.page_idx + 1,
            ),
            source_block=block,
            source_page=page,
        )]
    context.warnings.append(
        f"Could not parse numbered footnote on page {page.page_idx + 1}: {text[:60]!r}"
    )
    return []


def handle_discarded_as_raw_text(
    block: DiscardedBlock, page: PageInfo, context: ParseContext
) -> list[ParsedElement]:
    """Wrap a discarded block's text in a :class:`RawText` element."""
    text = extract_text_from_lines(block.lines).strip()
    if not text:
        return []
    return [ParsedElement(
        element=RawText(
            innerhtml=text,
            bounding_box=_get_bounding_box(block),
            page=page.page_idx + 1,
        ),
        source_block=block,
        source_page=page,
    )]



def handle_discarded_as_metadata(
    block: DiscardedBlock, page: PageInfo, context: ParseContext
) -> list[ParsedElement]:
    """
    Store a discarded block's text in ``context.metadata`` without producing
    any document element.

    Values are accumulated under a key derived from the block type (e.g.
    ``"headers"``, ``"footers"``), as a list of ``{"page": N, "text": "…"}``
    dicts.

    This handler does **not** propagate data to ``Document.metadata`` on its
    own.  It is intended for custom middleware authors who need to read
    accumulated values from ``context.metadata`` in a later middleware step.
    """
    text = extract_text_from_lines(block.lines).strip()
    if not text:
        return []
    key = block.type.value + "s"  # "headers", "footers", "page_numbers", …
    context.metadata.setdefault(key, []).append({"page": page.page_idx + 1, "text": text})
    return []


def handle_discarded_drop(
    block: DiscardedBlock, page: PageInfo, context: ParseContext
) -> list[ParsedElement]:
    """Suppress a discarded block — always returns an empty list."""
    return []


# ---------------------------------------------------------------------------
# Default discarded handler table
# ---------------------------------------------------------------------------


def _default_discarded_handlers() -> dict[DiscardedBlockType, DiscardedBlockHandler]:
    return {
        DiscardedBlockType.HEADER: handle_discarded_drop,
        DiscardedBlockType.FOOTER: handle_discarded_drop,
        DiscardedBlockType.PAGE_NUMBER: handle_discarded_drop,
        DiscardedBlockType.ASIDE_TEXT: handle_discarded_drop,
        DiscardedBlockType.PAGE_FOOTNOTE: handle_discarded_as_footnote,
    }


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------


@dataclass
class ExtractionConfig:
    """
    Handler table for :class:`~ragdoc.parsing.mineru.parser.CoreExtractionMiddleware`.

    Every field is a callable with the signature
    ``(block, page, context) -> list[ParsedElement]``.  Set any field to change
    how that block type is converted without subclassing or modifying the
    middleware.

    **Para-block handlers** are typed to their specific MinerU block class so
    that static type checkers can verify that custom handlers receive the right
    type.

    **Discarded-block handlers** share the :class:`DiscardedBlock` model and are
    keyed by :class:`~ragdoc.parsing.mineru.base.DiscardedBlockType`.  MinerU
    classifies some page regions as "discarded" (running headers, footers, page
    numbers, aside text, in-text footnotes).  The default behaviour is to drop
    all of them except numbered in-text footnotes:

    .. list-table::
       :header-rows: 1

       * - ``DiscardedBlockType``
         - Default handler
         - Produces
       * - ``HEADER``
         - :func:`handle_discarded_drop`
         - *(dropped)*
       * - ``FOOTER``
         - :func:`handle_discarded_drop`
         - *(dropped)*
       * - ``PAGE_NUMBER``
         - :func:`handle_discarded_drop`
         - *(dropped)*
       * - ``ASIDE_TEXT``
         - :func:`handle_discarded_drop`
         - *(dropped)*
       * - ``PAGE_FOOTNOTE``
         - :func:`handle_discarded_as_footnote`
         - :class:`~ragdoc.document.Footnote`

    To preserve running page headers as :class:`~ragdoc.document.RawText`
    elements instead of dropping them::

        from ragdoc.parsing.mineru.base import DiscardedBlockType
        from ragdoc.parsing.mineru.handlers import (
            ExtractionConfig,
            handle_discarded_as_raw_text,
        )

        config = ExtractionConfig()
        config.discarded_handlers[DiscardedBlockType.HEADER] = handle_discarded_as_raw_text

    Available discarded-block handlers:

    - :func:`handle_discarded_drop` — silently discard the block (default for
      most types).
    - :func:`handle_discarded_as_raw_text` — emit the text as a
      :class:`~ragdoc.document.RawText` element.
    - :func:`handle_discarded_as_footnote` — parse ``N text`` format and emit
      a :class:`~ragdoc.document.Footnote` element; drops the block with a
      warning when the format does not match.
    - :func:`handle_discarded_as_metadata` — accumulate text in
      ``context.metadata`` for use by a subsequent custom middleware; does not
      produce document elements and does not propagate to
      ``Document.metadata``.

    ``default_heading_level`` sets the heading level assigned to every
    :class:`~ragdoc.parsing.mineru.base.TitleBlock` before downstream
    middlewares (e.g. ``HeadingLevelProcessor``) refine it.
    """

    handle_title: Callable[[TitleBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_title_block
    )
    handle_text: Callable[[TextBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_text_block
    )
    handle_list: Callable[[ListBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_list_block
    )
    handle_code: Callable[[CodeBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_code_block
    )
    handle_image: Callable[[ImageBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_image_block
    )
    handle_table: Callable[[TableBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_table_block
    )
    handle_chart: Callable[[ChartBlock, PageInfo, ParseContext], list[ParsedElement]] = field(
        default_factory=lambda: handle_chart_block
    )
    discarded_handlers: dict[DiscardedBlockType, DiscardedBlockHandler] = field(
        default_factory=_default_discarded_handlers
    )
    default_heading_level: int = 1
