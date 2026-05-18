"""
Pydantic models for parsing MinerU _middle.json output files.

These models represent the intermediate processing results from MinerU PDF extraction.
"""

from enum import Enum
from typing import Annotated, Literal, TypeVar, Union, Any
from functools import cached_property

from pydantic import BaseModel, Field
from pathlib import Path
import json

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Iterator, Callable
from typing import Protocol
from dataclasses import dataclass, field
from ragdoc.document import ElementType

import re as _re

from pylatexenc.latex2text import LatexNodes2Text

_LATEX2TEXT = LatexNodes2Text(math_mode="text")


def _latex_to_text(latex: str) -> str:
    """Convert a LaTeX string to plain text using pylatexenc.

    Handles commands like ``\\mathrm{H_2O}`` → ``H2O``,
    ``\\mathrm { C O } _ { 2 }`` → ``CO2``, etc.

    Sub/superscript markers (``_`` / ``^``) are stripped so that
    chemical formulae and isotope names read naturally as plain text.

    Only used internally to normalize text for content-based comparisons in tests.
    """
    result = _LATEX2TEXT.latex_to_text(latex).strip()
    # Strip sub/superscript markers left by pylatexenc
    result = _re.sub(r"[_^]", "", result)
    # Collapse whitespace
    result = _re.sub(r"\s+", " ", result).strip()
    return result


def normalize_math_spaces(latex: str) -> str:
    r"""Collapse OCR-inserted spaces in numeric LaTeX math expressions.

    OCR pipelines (e.g. MinerU) often emit individual digits and decimal
    separators separated by spaces: ``2 0 0 4`` instead of ``2004``, or
    ``1 . 2`` instead of ``1.2``.  LaTeX math mode ignores these spaces
    visually, but ``latex2mathml`` treats each space-separated token as a
    separate element, producing fragmented ``<mn>`` and ``<mo>`` nodes.

    Two passes are applied:

    1. **Digit runs**: ``(\d)\s+(?=\d)`` — collapses any inter-digit spaces
       in a single left-to-right pass (lookahead avoids consuming the next
       digit, so ``1 2 3`` → ``123`` in one substitution).

    2. **Decimal points**: ``(\d)\s*\.\s*(\d)`` — collapses spaces around a
       decimal point surrounded by digits (``1 . 2`` → ``1.2``).  The
       compact form ``1.2`` is then tokenised by ``latex2mathml`` as a single
       ``<mn>1.2</mn>`` node rather than ``<mn>1</mn><mo>.</mo><mn>2</mn>``.
    """
    result = _re.sub(r"(\d)\s+(?=\d)", r"\1", latex)
    result = _re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", result)
    return result


def latex_to_mathml(latex: str, display: bool = False) -> str:
    """Convert a LaTeX string to a MathML ``<math>`` element.

    Before conversion, :func:`normalize_math_spaces` is applied to collapse
    OCR-fragmented digit sequences (``2 0 0 4`` → ``2004``), ensuring numbers
    appear as single ``<mn>`` nodes rather than one per digit.

    Falls back gracefully: ``latex2mathml`` never raises on unknown commands —
    it renders them as ``<mi>`` nodes.

    Args:
        latex: Raw LaTeX string (without surrounding ``$`` delimiters).
        display: ``True`` for block/display equations, ``False`` for inline.

    Returns:
        A ``<math>`` element string.
    """
    if not latex:
        return ""

    from latex2mathml.converter import convert

    display_mode = "block" if display else "inline"
    return convert(normalize_math_spaces(latex), display=display_mode)

E = TypeVar("E", bound=ElementType)
# =============================================================================
# Enums
# =============================================================================


class BlockAngle(int, Enum):
    """Valid rotation angles for blocks."""

    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270


class SpanType(str, Enum):
    """Types of spans within a line."""

    TEXT = "text"
    INLINE_EQUATION = "inline_equation"
    INTERLINE_EQUATION = "interline_equation"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"


class BlockType(str, Enum):
    """Types of content blocks in para_blocks (Level 1 and Level 2)."""

    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    CODE = "code"
    # Level 1 block types (containers for table/image/chart content)
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    # Level 2 block types (inside image/table/chart containers)
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"
    IMAGE_FOOTNOTE = "image_footnote"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"
    CHART_BODY = "chart_body"
    CHART_CAPTION = "chart_caption"
    CHART_FOOTNOTE = "chart_footnote"
    INTERLINE_EQUATION = "interline_equation"
    INDEX = "index"


class DiscardedBlockType(str, Enum):
    """Types of blocks that may appear in discarded_blocks."""

    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"


class ListSubType(str, Enum):
    """Sub-types for list blocks."""

    TEXT = "text"  # Ordinary list
    REF_TEXT = "ref_text"  # Reference / bibliography style list


class CodeSubType(str, Enum):
    """Sub-types for code blocks."""

    CODE = "code"
    ALGORITHM = "algorithm"


class CodeBlockType(str, Enum):
    """Types of blocks within a code block."""

    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"


# =============================================================================
# Bounding Box
# =============================================================================

BBox = Annotated[
    list[float],
    Field(
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x1, y1, x2, y2]",
    ),
]


# =============================================================================
# Span Models
# =============================================================================


class TextSpan(BaseModel):
    """A text span within a line."""

    bbox: BBox
    type: Literal["text"] = "text"
    content: str
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class InlineEquationSpan(BaseModel):
    """An inline equation span within a line (LaTeX content)."""

    bbox: BBox
    type: Literal["inline_equation"] = "inline_equation"
    content: str = Field(description="LaTeX equation content")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class InterlineEquationSpan(BaseModel):
    """An interline equation span (standalone formula)."""

    bbox: BBox
    type: Literal["interline_equation"] = "interline_equation"
    content: str = Field(default="", description="LaTeX equation content")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class ImageSpan(BaseModel):
    """An image span within a line."""

    bbox: BBox
    type: Literal["image"] = "image"
    image_path: str = Field(description="Path to the image file")


class TableSpan(BaseModel):
    """A table span within a line."""

    bbox: BBox
    type: Literal["table"] = "table"
    html: str = Field(default="", description="HTML representation of the table")
    image_path: str = Field(default="", description="Path to the table image file")


class ChartSpan(BaseModel):
    """A chart span within a line — OCR'd chart data as markdown plus its image path."""

    bbox: BBox
    type: Literal["chart"] = "chart"
    content: str = Field(default="", description="Markdown representation of the chart data")
    image_path: str = Field(default="", description="Path to the rendered chart image file")


Span = Annotated[
    Union[TextSpan, InlineEquationSpan, InterlineEquationSpan, ImageSpan, TableSpan, ChartSpan],
    Field(discriminator="type"),
]


# =============================================================================
# Line Model
# =============================================================================


class Line(BaseModel):
    """A line of text containing one or more spans."""

    bbox: BBox
    spans: list[Span]


# =============================================================================
# Block Models
# =============================================================================


class BaseBlock(BaseModel):
    """Base class for all block types."""

    bbox: BBox
    angle: BlockAngle = Field(default=BlockAngle.DEG_0, description="Rotation angle")
    index: int = Field(description="Block index within the page")


class TextBlock(BaseBlock):
    """A text block containing lines of text."""

    type: Literal["text"] = "text"
    lines: list[Line]


class TitleBlock(BaseBlock):
    """A title block containing lines of text."""

    type: Literal["title"] = "title"
    lines: list[Line]

    @property
    def average_line_height(self) -> float:
        """Calculate the average line height for this title block."""
        if not self.lines:
            return 0.0
        total_height = sum(line.bbox[3] - line.bbox[1] for line in self.lines)
        return total_height / len(self.lines)


# =============================================================================
# List Block Models
# =============================================================================


class ListItemBlock(BaseBlock):
    """An item within a list block.

    MinerU v3.1+ tags some list items as ``ref_text`` (bibliography / reference
    entries) instead of plain ``text``. Both shapes are otherwise identical, so
    accept either discriminator value.
    """

    type: Literal["text", "ref_text"] = "text"
    lines: list[Line]


class ListBlock(BaseBlock):
    """
    A list block containing multiple text blocks as items.

    List blocks are second-level blocks with a sub_type distinguishing categories:
    - text: ordinary list
    - ref_text: reference / bibliography style list
    """

    type: Literal["list"] = "list"
    blocks: list[ListItemBlock] = Field(description="List items as text blocks")
    sub_type: ListSubType = Field(
        default=ListSubType.TEXT, description="List category type"
    )


# =============================================================================
# Code Block Models
# =============================================================================


class CodeBodyBlock(BaseModel):
    """The body of a code block containing the actual code."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["code_body"] = "code_body"


class CodeCaptionBlock(BaseModel):
    """Optional caption for a code block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["code_caption"] = "code_caption"


CodeInnerBlock = Annotated[
    Union[CodeBodyBlock, CodeCaptionBlock], Field(discriminator="type")
]


class CodeBlock(BaseBlock):
    """
    A code block containing code body and optional caption.

    Code blocks have sub_type:
    - code: regular code
    - algorithm: algorithm pseudocode
    """

    type: Literal["code"] = "code"
    blocks: list[CodeInnerBlock] = Field(
        description="Code body and optional caption blocks"
    )
    sub_type: CodeSubType = Field(
        default=CodeSubType.CODE, description="Code block category"
    )

    @property
    def code_body(self) -> CodeBodyBlock | None:
        """Get the code body block if present."""
        for block in self.blocks:
            if isinstance(block, CodeBodyBlock):
                return block
        return None

    @property
    def code_caption(self) -> CodeCaptionBlock | None:
        """Get the code caption block if present."""
        for block in self.blocks:
            if isinstance(block, CodeCaptionBlock):
                return block
        return None


# =============================================================================
# Image Block Models (Level 1 with Level 2 inner blocks)
# =============================================================================


class ImageBodyBlock(BaseModel):
    """The body of an image block containing the image reference."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["image_body"] = "image_body"


class ImageCaptionBlock(BaseModel):
    """Caption for an image block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["image_caption"] = "image_caption"


class ImageFootnoteBlock(BaseModel):
    """Footnote for an image block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["image_footnote"] = "image_footnote"


ImageInnerBlock = Annotated[
    Union[ImageBodyBlock, ImageCaptionBlock, ImageFootnoteBlock],
    Field(discriminator="type"),
]


class ImageBlock(BaseModel):
    """
    An image block (Level 1) containing image body and optional caption/footnote.

    According to MinerU docs, Level 1 blocks (image | table) contain Level 2 blocks.
    """

    type: Literal["image"] = "image"
    bbox: BBox
    blocks: list[ImageInnerBlock] = Field(
        description="Image body and optional caption/footnote blocks"
    )
    index: int = Field(description="Block index within the page")

    @property
    def image_body(self) -> ImageBodyBlock | None:
        """Get the image body block if present."""
        for block in self.blocks:
            if isinstance(block, ImageBodyBlock):
                return block
        return None

    @property
    def image_caption(self) -> ImageCaptionBlock | None:
        """Get the image caption block if present."""
        for block in self.blocks:
            if isinstance(block, ImageCaptionBlock):
                return block
        return None


# =============================================================================
# Table Block Models (Level 1 with Level 2 inner blocks)
# =============================================================================


class TableBodyBlock(BaseModel):
    """The body of a table block containing the table content."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["table_body"] = "table_body"


class TableCaptionBlock(BaseModel):
    """Caption for a table block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["table_caption"] = "table_caption"


class TableFootnoteBlock(BaseModel):
    """Footnote for a table block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["table_footnote"] = "table_footnote"


TableInnerBlock = Annotated[
    Union[TableBodyBlock, TableCaptionBlock, TableFootnoteBlock],
    Field(discriminator="type"),
]


class TableBlock(BaseModel):
    """
    A table block (Level 1) containing table body and optional caption/footnote.

    According to MinerU docs, Level 1 blocks (table | image) contain Level 2 blocks.
    """

    type: Literal["table"] = "table"
    bbox: BBox
    blocks: list[TableInnerBlock] = Field(
        description="Table body and optional caption/footnote blocks"
    )
    index: int = Field(description="Block index within the page")

    @property
    def table_body(self) -> TableBodyBlock | None:
        """Get the table body block if present."""
        for block in self.blocks:
            if isinstance(block, TableBodyBlock):
                return block
        return None

    @property
    def table_caption(self) -> TableCaptionBlock | None:
        """Get the table caption block if present."""
        for block in self.blocks:
            if isinstance(block, TableCaptionBlock):
                return block
        return None


# =============================================================================
# Reference-text + Interline-equation top-level blocks
#
# MinerU v3.1+ emits these as top-level para_blocks (previously they appeared
# only as list items / spans). Their shape mirrors TextBlock — a flat list of
# lines — so we model them as such with their own type discriminator.
# =============================================================================


class RefTextBlock(BaseBlock):
    """A bibliography / reference-style block at top level (mineru ``ref_text``)."""

    type: Literal["ref_text"] = "ref_text"
    lines: list[Line]


class InterlineEquationBlock(BaseBlock):
    """A standalone interline equation occupying its own paragraph block.

    ``interline_equation`` also exists as a SpanType; this is the para_block
    variant introduced in mineru v3.1+.
    """

    type: Literal["interline_equation"] = "interline_equation"
    lines: list[Line]


# =============================================================================
# Chart Block Models (Level 1 with Level 2 inner blocks)
#
# MinerU v3.1+ emits `chart` blocks for figures whose visual content has been
# OCR'd into a markdown table (bar charts, line plots, etc.). Structurally
# identical to TableBlock — the inner span content is markdown — so we model
# them as a parallel set of pydantic types and reuse the table handler.
# =============================================================================


class ChartBodyBlock(BaseModel):
    """The body of a chart block containing the OCR'd chart data."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["chart_body"] = "chart_body"


class ChartCaptionBlock(BaseModel):
    """Caption for a chart block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["chart_caption"] = "chart_caption"


class ChartFootnoteBlock(BaseModel):
    """Footnote for a chart block."""

    bbox: BBox
    lines: list[Line]
    index: int
    angle: BlockAngle = Field(default=BlockAngle.DEG_0)
    type: Literal["chart_footnote"] = "chart_footnote"


ChartInnerBlock = Annotated[
    Union[ChartBodyBlock, ChartCaptionBlock, ChartFootnoteBlock],
    Field(discriminator="type"),
]


class ChartBlock(BaseModel):
    """A chart block (Level 1) containing chart body and optional caption/footnote."""

    type: Literal["chart"] = "chart"
    bbox: BBox
    blocks: list[ChartInnerBlock] = Field(
        description="Chart body and optional caption/footnote blocks"
    )
    index: int = Field(description="Block index within the page")

    @property
    def chart_body(self) -> ChartBodyBlock | None:
        for block in self.blocks:
            if isinstance(block, ChartBodyBlock):
                return block
        return None

    @property
    def chart_caption(self) -> ChartCaptionBlock | None:
        for block in self.blocks:
            if isinstance(block, ChartCaptionBlock):
                return block
        return None


# =============================================================================
# Discarded Block Model
# =============================================================================


class DiscardedBlock(BaseBlock):
    """
    A block that was discarded during processing.

    Discarded blocks may contain:
    - header: page headers
    - footer: page footers
    - page_number: page numbers
    - aside_text: sidebar or marginal text
    - page_footnote: footnotes
    """

    type: DiscardedBlockType
    lines: list[Line]


# =============================================================================
# Union Types for Paragraph Blocks
# =============================================================================

ParaBlock = Annotated[
    Union[
        TextBlock,
        TitleBlock,
        ListBlock,
        CodeBlock,
        ImageBlock,
        TableBlock,
        ChartBlock,
        RefTextBlock,
        InterlineEquationBlock,
    ],
    Field(discriminator="type"),
]


# =============================================================================
# Page Model
# =============================================================================


class PageInfo(BaseModel):
    """Information about a single page in the PDF."""

    para_blocks: list[ParaBlock] = Field(
        default_factory=list, description="Paragraph blocks on this page"
    )
    discarded_blocks: list[DiscardedBlock] = Field(
        default_factory=list, description="Blocks discarded during processing"
    )
    page_idx: int = Field(description="Zero-based page index")

    model_config = {"frozen": False}

    @cached_property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """
        Calculate the bounding box of the page based on all blocks.
        
        Returns:
            A tuple (x1, y1, x2, y2) representing the bounding box that
            encompasses all para_blocks and discarded_blocks on the page.
            Returns (0, 0, 612, 792) as default if no blocks exist (US Letter size in points).
        """
        all_blocks = list(self.para_blocks) + list(self.discarded_blocks)
        
        if not all_blocks:
            # Default to US Letter size in points
            return (0.0, 0.0, 612.0, 792.0)
        
        min_x = min(block.bbox[0] for block in all_blocks)
        min_y = min(block.bbox[1] for block in all_blocks)
        max_x = max(block.bbox[2] for block in all_blocks)
        max_y = max(block.bbox[3] for block in all_blocks)
        
        return (min_x, min_y, max_x, max_y)

    @property
    def page_width(self) -> float:
        """Get the estimated page width based on block extents."""
        return self.bounding_box[2] - self.bounding_box[0]

    @property
    def page_height(self) -> float:
        """Get the estimated page height based on block extents."""
        return self.bounding_box[3] - self.bounding_box[1]


# =============================================================================
# Document Model (Top-Level)
# =============================================================================


class MinerUMiddleDocument(BaseModel):
    """
    Top-level model for MinerU _middle.json output files.

    This represents the intermediate processing results from MinerU PDF extraction.
    File naming format: {original_filename}_middle.json
    """

    pdf_info: list[PageInfo] = Field(description="List of pages in the document")

    @classmethod
    def from_json_path(cls, path: Path|str) -> "MinerUMiddleDocument":
        """Load a MinerUMiddleDocument from a _middle.json file."""
        path = Path(path)
        if not path.is_file() or not path.name.endswith("_middle.json"):
            raise ValueError(f"Path {path} is not a valid _middle.json file")
        
        with open(path, encoding="utf-8") as f:
            json_data = json.load(f)
        
        return cls.model_validate(json_data)

    def get_page(self, page_idx: int) -> PageInfo | None:
        """Get a page by its index."""
        for page in self.pdf_info:
            if page.page_idx == page_idx:
                return page
        return None

    @property
    def num_pages(self) -> int:
        """Get the total number of pages."""
        return len(self.pdf_info)

    def get_all_text_content(self) -> str:
        """Extract all text content from the document."""
        text_parts: list[str] = []

        for page in self.pdf_info:
            for block in page.para_blocks:
                text_parts.extend(_extract_text_from_block(block))

        return "\n".join(text_parts)

    def get_all_blocks_by_type(
        self, block_type: BlockType
    ) -> list[TextBlock | TitleBlock | ListBlock | CodeBlock | ImageBlock | TableBlock]:
        """Get all blocks of a specific type across all pages."""
        blocks: list[TextBlock | TitleBlock | ListBlock | CodeBlock | ImageBlock | TableBlock] = []

        for page in self.pdf_info:
            for block in page.para_blocks:
                if block.type == block_type.value:
                    blocks.append(block)

        return blocks


# =============================================================================
# Helper Functions
# =============================================================================


def extract_text_from_lines(lines: list[Line]) -> str:
    """Extract HTML content from a list of lines.

    Equation spans are converted to MathML ``<math>`` elements via
    :func:`latex_to_mathml` (with the original LaTeX preserved in ``alttext``).
    Text spans are included verbatim.
    """
    text_parts: list[str] = []

    for line in lines:
        span_texts: list[str] = []
        for span in line.spans:
            content = getattr(span, "content", "")
            if isinstance(span, InlineEquationSpan):
                content = latex_to_mathml(content, display=False)
            elif isinstance(span, InterlineEquationSpan):
                content = latex_to_mathml(content, display=True)
            span_texts.append(content)
        text_parts.append(" ".join(span_texts))

    return " ".join(text_parts)


def _extract_text_from_block(block: ParaBlock) -> list[str]:
    """Extract text content from a paragraph block."""
    text_parts: list[str] = []

    if isinstance(block, (TextBlock, TitleBlock)):
        text_parts.append(extract_text_from_lines(block.lines))

    elif isinstance(block, ListBlock):
        for item in block.blocks:
            text_parts.append(extract_text_from_lines(item.lines))

    elif isinstance(block, CodeBlock):
        if block.code_body:
            text_parts.append(extract_text_from_lines(block.code_body.lines))

    elif isinstance(block, ImageBlock):
        # Extract caption text if present
        if block.image_caption:
            text_parts.append(extract_text_from_lines(block.image_caption.lines))

    elif isinstance(block, TableBlock):
        # Extract caption text if present
        if block.table_caption:
            text_parts.append(extract_text_from_lines(block.table_caption.lines))

    return text_parts


def parse_middle_json(json_data: dict) -> MinerUMiddleDocument:
    """
    Parse a MinerU _middle.json file into a structured document.

    Args:
        json_data: The parsed JSON data from a _middle.json file.

    Returns:
        A MinerUMiddleDocument instance.

    Example:
        >>> import json
        >>> with open("document_middle.json") as f:
        ...     data = json.load(f)
        >>> doc = parse_middle_json(data)
        >>> print(f"Document has {doc.num_pages} pages")
    """
    return MinerUMiddleDocument.model_validate(json_data)

def parse_middle_json_file(path: Path|str) -> MinerUMiddleDocument:
    """
    Parse a MinerU _middle.json file from disk.

    Args:
        path: Path to the _middle.json file or to a parent (or ancestor) directory containing the file.
    Returns:
        A MinerUMiddleDocument instance.

    """
    path: Path = Path(path)
    if path.is_file() and path.name.endswith("_middle.json"):
        json_path = path
    else:
        # Search for *_middle.json in the directory
        json_files = list(path.rglob("*_middle.json"))
        if not json_files:
            raise FileNotFoundError(f"No *_middle.json file found in {path}")
        if len(json_files) > 1:
            raise ValueError(f"Multiple *_middle.json files found in {path}, please specify one: {[f.name for f in json_files]}")
        json_path = json_files[0]

    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    return parse_middle_json(json_data)

def parse_directory_middle_jsons(dir_path: Path|str) -> list[MinerUMiddleDocument]:
    """
    Parse all *_middle.json files in a directory.

    Args:
        dir_path: Path to a directory containing one or more *_middle.json files.
    Returns:
        A list of MinerUMiddleDocument instances, one for each *_middle.json file found.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")

    return [parse_middle_json_file(json_file) for json_file in dir_path.rglob("*_middle.json")]

SourceBlock = ParaBlock | DiscardedBlock


# =============================================================================
# ParsedElement - Unified Wrapper with Source Linkage
# =============================================================================


@dataclass
class ParsedElement:
    """
    Wrapper linking a parsed document element to its source block and page.
    
    This is the core data structure that enables the middleware pipeline to:
    - Access the original MinerU block data for any element
    - Modify elements in place while preserving source linkage
    - Filter elements by type for specialized processing
    
    Design Rationale:
    -----------------
    Instead of maintaining separate lists for each element type (headings, 
    paragraphs, etc.), we use a single list of ParsedElement wrappers. This:
    
    1. Preserves document order naturally (all elements in reading order)
    2. Provides uniform access to source data for any element type
    3. Simplifies middleware implementation (filter by type, access source)
    4. Makes it easy to add new element types without changing the context
    
    The source_block and source_page fields enable middlewares to make decisions
    based on visual properties (font size, position, centering) that aren't
    captured in the parsed element itself.
    
    Attributes:
        element: The parsed document element (Heading, Paragraph, etc.)
        source_block: The original MinerU block this element was extracted from
        source_page: The page containing the source block
        metadata: Optional metadata dictionary for middlewares to store additional info (usually for other middlewares).
          e.g. if heading is centered, its average height, etc.
    """
    
    element: ElementType
    source_block: SourceBlock
    source_page: PageInfo
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Parsing Context - Shared State Between Middlewares
# =============================================================================


@dataclass
class ParseContext:
    """
    Context object passed through the parsing pipeline.
    
    The context maintains a unified list of ParsedElement wrappers, preserving
    document order and source linkage. Middlewares can:
    - Filter elements by type using helper methods
    - Modify elements in place
    - Access source blocks for visual property analysis
    - Add metadata and warnings
    
    Design Rationale:
    -----------------
    The single `elements` list (vs. separate typed lists) ensures:
    - Document order is preserved naturally
    - Source linkage is always available
    - Middlewares can be written generically or for specific types
    - Adding new element types doesn't require context changes
    
    Attributes:
        source: The original MinerU document being parsed
        elements: All parsed elements with source linkage (in document order)
        metadata: Accumulated metadata from middlewares thats not block-specific
        warnings: Errors/warnings collected during parsing
        document_title: The detected document title (set by TitleDetectionMiddleware)
    """
    
    # Source document
    source: MinerUMiddleDocument
    
    # All parsed elements with source linkage (in document order)
    elements: list[ParsedElement] = field(default_factory=list)
    
    # Metadata accumulated during parsing
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Errors/warnings collected during parsing
    warnings: list[str] = field(default_factory=list)
    
    # Document title detected during parsing (set by TitleDetectionMiddleware)
    document_title: str | None = None
    
    # -------------------------------------------------------------------------
    # Element Filtering Methods
    # -------------------------------------------------------------------------
    
    def get_elements_by_type(
        self, element_type: type[E]
    ) -> list[tuple[int, ParsedElement]]:
        """
        Get all elements of a specific type with their indices.
        
        Returns tuples of (index, parsed_element) to allow in-place updates.
        
        Example:
            for idx, parsed in context.get_elements_by_type(Heading):
                # Access the heading
                heading = parsed.element
                # Access source block for visual properties
                source = parsed.source_block
                # Modify in place
                context.elements[idx] = ParsedElement(
                    element=Heading(...),
                    source_block=parsed.source_block,
                    source_page=parsed.source_page,
                )
        """
        return [
            (i, elem) for i, elem in enumerate(self.elements)
            if isinstance(elem.element, element_type)
        ]
    
    def iter_elements_by_type(
        self, element_type: type[E]
    ) -> Iterator[tuple[ParsedElement, E]]:
        """
        Iterate over elements of a specific type with index and typed element.
        
        Yields (parsed_element, typed_element) tuples for convenient access.
        
        Example:
            for parsed, heading in context.iter_elements_by_type(Heading):
                # heading is already typed as Heading
                new_level = compute_level(parsed.source_block)
                ...
        """
        for elem in self.elements:
            if isinstance(elem.element, element_type):
                yield (elem, elem.element)
    
    # -------------------------------------------------------------------------
    # Convenience Properties (for quick access without filtering)
    # -------------------------------------------------------------------------
    
    @property
    def headings(self) -> list[ParsedElement]:
        """Get all heading elements (without indices)."""
        from ragdoc.document import Heading
        return [e for e in self.elements if isinstance(e.element, Heading)]
    
    @property
    def paragraphs(self) -> list[ParsedElement]:
        """Get all paragraph elements (without indices)."""
        from ragdoc.document import Paragraph
        return [e for e in self.elements if isinstance(e.element, Paragraph)]
    
    @property
    def tables(self) -> list[ParsedElement]:
        """Get all table elements (without indices)."""
        from ragdoc.document import Table
        return [e for e in self.elements if isinstance(e.element, Table)]
    
    @property
    def lists(self) -> list[ParsedElement]:
        """Get all list elements (without indices)."""
        from ragdoc.document import DocumentList
        return [e for e in self.elements if isinstance(e.element, DocumentList)]
    
    @property
    def images(self) -> list[ParsedElement]:
        """Get all image elements (without indices)."""
        from ragdoc.document import Image
        return [e for e in self.elements if isinstance(e.element, Image)]
    
    @property
    def raw_texts(self) -> list[ParsedElement]:
        """Get all raw text elements (without indices)."""
        from ragdoc.document import RawText
        return [e for e in self.elements if isinstance(e.element, RawText)]
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the context."""
        self.warnings.append(message)
    

# =============================================================================
# Middleware Protocol
# =============================================================================


# Type alias for async call_next function
AsyncCallNext = Callable[[ParseContext], Awaitable[ParseContext]]


class ParserMiddleware(Protocol):
    """
    Protocol for parser middlewares (async-first).
    
    Middlewares can modify the ParseContext during parsing.
    They receive the context and an async `call_next` function to continue the chain.
    """
    
    async def __call__(
        self,
        context: ParseContext,
        call_next: AsyncCallNext,
    ) -> ParseContext:
        """
        Process the context and optionally call the next middleware.
        
        Args:
            context: The parsing context with accumulated state
            call_next: Async function to call the next middleware in the chain
            
        Returns:
            The (possibly modified) ParseContext
        """
        ...


# =============================================================================
# Base Middleware Classes
# =============================================================================


class BaseMiddleware(ABC):
    """Abstract base class for middlewares with common functionality (async-first)."""
    
    @abstractmethod
    async def process(self, context: ParseContext) -> ParseContext:
        """Process the context. Override in subclasses."""
        ...
    
    async def __call__(
        self,
        context: ParseContext,
        call_next: AsyncCallNext,
    ) -> ParseContext:
        """Execute this middleware and continue the chain."""
        context = await self.process(context)
        return await call_next(context)


class PreProcessingMiddleware(BaseMiddleware):
    """Middleware that runs before the main extraction."""
    pass


class PostProcessingMiddleware(BaseMiddleware):
    """Middleware that runs after the main extraction."""
    pass

