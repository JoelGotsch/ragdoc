"""
Heading level processing.

This module provides processors for refining heading levels based on
visual properties extracted from CSS in innerHTML.

Classes:
    HeadingLevelProcessor: Refine heading levels based on font size
    TitleDetectionProcessor: Detect and extract document title
    HeadingNormalizationProcessor: Compact levels to eliminate gaps

Functions:
    extract_font_size: Extract font size from HTML/CSS
    is_centered: Check if HTML content is centered
    is_bold: Check if HTML content is bold
    is_all_caps: Check if text is all caps
    compute_size_to_level_mapping: Map effective sizes to heading levels
    normalize_heading_levels: Remap heading levels to a gap-free sequence
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ragdoc.document import Heading
from ragdoc.processing.base import DocumentProcessor

if TYPE_CHECKING:
    from ragdoc.document import Document

logger = logging.getLogger(__name__)


# =============================================================================
# Standalone Functions for HTML Property Extraction
# =============================================================================


def extract_font_size(html: str) -> float | None:
    """
    Extract font size from CSS in HTML.

    Looks for font-size in style attributes and converts to points.
    Supports pt, px, em, and rem units.

    Args:
        html: HTML string potentially containing style attributes

    Returns:
        Font size in points, or None if not found

    Example:
        >>> extract_font_size('<span style="font-size: 14pt;">Text</span>')
        14.0
        >>> extract_font_size('<span style="font-size: 20px;">Text</span>')
        15.0  # 20 * 0.75
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(style=True):
        style = tag.get("style", "")

        match = re.search(r"font-size:\s*([\d.]+)(pt|px|em|rem)", style)
        if match:
            size = float(match.group(1))
            unit = match.group(2)

            # Convert to points (approximate)
            if unit == "px":
                size = size * 0.75  # 1px ≈ 0.75pt
            elif unit in ("em", "rem"):
                size = size * 12  # Assume base 12pt

            return size

    return None


def is_centered(html: str) -> bool:
    """
    Check if HTML content is centered based on CSS.

    Args:
        html: HTML string potentially containing style attributes

    Returns:
        True if text-align: center is found in any style attribute

    Example:
        >>> is_centered('<span style="text-align: center;">Text</span>')
        True
        >>> is_centered('<span>Text</span>')
        False
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        if "text-align: center" in style or "text-align:center" in style:
            return True

    return False


def is_bold(html: str) -> bool:
    """
    Check if HTML content is bold based on CSS or tags.

    Checks for:
    - <b> and <strong> tags
    - font-weight: bold in style
    - font-weight >= 600 in style

    Args:
        html: HTML string to check

    Returns:
        True if content appears bold

    Example:
        >>> is_bold('<b>Bold text</b>')
        True
        >>> is_bold('<span style="font-weight: 700;">Bold</span>')
        True
    """
    soup = BeautifulSoup(html, "html.parser")

    # Check for bold tags
    if soup.find(["b", "strong"]):
        return True

    # Check for font-weight in style
    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        if "font-weight: bold" in style or "font-weight:bold" in style:
            return True
        # font-weight: 600+ is typically bold
        match = re.search(r"font-weight:\s*(\d+)", style)
        if match and int(match.group(1)) >= 600:
            return True

    return False


def is_all_caps(text: str) -> bool:
    """
    Check if text is all uppercase.

    Only considers alphabetic characters; numbers and punctuation are ignored.
    Returns False for empty strings or strings with no letters.

    Args:
        text: Plain text to check

    Returns:
        True if all alphabetic characters are uppercase

    Example:
        >>> is_all_caps("HELLO WORLD")
        True
        >>> is_all_caps("Hello World")
        False
        >>> is_all_caps("ABC 123")
        True
    """
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)


def compute_size_to_level_mapping(
    effective_sizes: list[float],
    max_levels: int = 6,
) -> dict[float, int]:
    """
    Compute a mapping from effective sizes to heading levels.

    The algorithm:
    1. Get unique sizes (excluding zero)
    2. Sort descending (largest = most important = h1)
    3. Assign levels 1 to max_levels based on rank
    4. If more unique sizes than levels, compress proportionally

    Args:
        effective_sizes: List of effective sizes for all headings
        max_levels: Maximum number of heading levels (default 6)

    Returns:
        Mapping from effective_size to heading level (1-based)

    Example:
        >>> compute_size_to_level_mapping([24.0, 18.0, 12.0, 24.0])
        {24.0: 1, 18.0: 2, 12.0: 3}
    """
    if not effective_sizes:
        return {}

    # Get unique sizes > 0, sorted descending
    unique_sizes = sorted(set(s for s in effective_sizes if s > 0), reverse=True)

    if not unique_sizes:
        return {}

    size_to_level: dict[float, int] = {}
    num_unique = len(unique_sizes)

    for i, size in enumerate(unique_sizes):
        if num_unique <= max_levels:
            level = i + 1
        else:
            # Compress proportionally
            level = min(int(i * max_levels / num_unique) + 1, max_levels)
        size_to_level[size] = level

    return size_to_level


# Type alias for the mapping function
SizeToLevelMapper = Callable[[list[float], int], dict[float, int]]


# =============================================================================
# Pydantic Models
# =============================================================================


class HeadingVisualInfo(BaseModel):
    """
    Visual information about a heading for level computation.

    This model captures the visual properties of a heading element
    that are used to determine its appropriate heading level.
    """

    index: int = Field(..., description="Index of the heading in the document's elements list")
    font_size: float | None = Field(
        default=None, description="Font size in points, extracted from CSS. None if not available."
    )
    is_all_caps: bool = Field(default=False, description="Whether the heading text is all uppercase")
    is_centered: bool = Field(default=False, description="Whether the heading is centered (from CSS text-align)")
    is_bold: bool = Field(default=False, description="Whether the heading is bold (from CSS or <b>/<strong> tags)")

    @property
    def effective_size(self) -> float:
        """
        Calculate effective size with style boosts.

        Boosts are applied multiplicatively:
        - All caps: 20% boost (visually more prominent)
        - Centered: 20% boost (typically main titles)
        - Bold: 10% boost

        Returns:
            Effective size rounded to 1 decimal, or 0.0 if font_size is None
        """
        if self.font_size is None:
            return 0.0

        multiplier = 1.0
        if self.is_all_caps:
            multiplier *= 1.2
        if self.is_centered:
            multiplier *= 1.2
        if self.is_bold:
            multiplier *= 1.1

        return round(self.font_size * multiplier, 1)


# =============================================================================
# Processors
# =============================================================================


class HeadingLevelProcessor(DocumentProcessor):
    """
    Processor that refines heading levels based on visual properties.

    This processor analyzes CSS styles in heading innerHTML to determine
    appropriate heading levels (1-6) based on:
    - Font size (larger = higher level)
    - All caps text (boost in visual hierarchy)
    - Centered text (boost in visual hierarchy)
    - Bold text (slight boost)

    The processor groups headings by effective visual size and maps them
    to heading levels 1-6, with the largest becoming h1.

    This is a universal processor that works with any parser output,
    as long as visual properties are stored as CSS in innerHTML.

    Example:
        ```python
        processor = HeadingLevelProcessor()
        doc = processor.process(doc)

        # With custom mapping function
        def custom_mapper(sizes, max_levels):
            # Custom logic
            return {size: 1 for size in sizes}

        processor = HeadingLevelProcessor(size_to_level_mapper=custom_mapper)
        ```

    Attributes:
        max_heading_levels: Maximum distinct heading levels (default 6)
        trust_parser_levels: If True and document.parser is 'html',
            trust existing levels and skip processing
        size_to_level_mapper: Function to compute size-to-level mapping
    """

    def __init__(
        self,
        max_heading_levels: int = 6,
        trust_parser_levels: bool = True,
        size_to_level_mapper: SizeToLevelMapper | None = None,
    ):
        """
        Initialize the processor.

        Args:
            max_heading_levels: Maximum number of distinct heading levels (1-6)
            trust_parser_levels: Whether to trust existing levels for parsers
                that already have reliable heading levels (e.g., 'html')
            size_to_level_mapper: Custom function to compute size-to-level mapping.
                Defaults to compute_size_to_level_mapping.
        """
        self.max_heading_levels = max_heading_levels
        self.trust_parser_levels = trust_parser_levels
        self.size_to_level_mapper = size_to_level_mapper or compute_size_to_level_mapping

    async def process(self, document: Document) -> Document:
        """Analyze headings and update their levels based on visual properties."""

        # Skip if parser already has reliable heading levels
        if self.trust_parser_levels and document.parser in ("html", "pandoc"):
            logger.debug(f"HeadingLevelProcessor: skipping (parser={document.parser!r}, trust_parser_levels=True)")
            return document

        # Collect visual info for all headings
        heading_infos = self._collect_heading_infos(document)

        if not heading_infos:
            return document

        # Compute level mapping using the mapper function
        effective_sizes = [info.effective_size for info in heading_infos]
        size_to_level = self.size_to_level_mapper(effective_sizes, self.max_heading_levels)
        logger.debug(f"HeadingLevelProcessor: {len(heading_infos)} headings, {len(size_to_level)} distinct sizes")

        # Update heading levels
        for info in heading_infos:
            heading = document.elements[info.index]
            if isinstance(heading, Heading) and info.effective_size in size_to_level:
                heading.level = size_to_level[info.effective_size]

        return document

    def _collect_heading_infos(self, document: Document) -> list[HeadingVisualInfo]:
        """Collect visual information for all headings."""

        infos: list[HeadingVisualInfo] = []

        for idx, element in enumerate(document.elements):
            if not isinstance(element, Heading):
                continue

            infos.append(
                HeadingVisualInfo(
                    index=idx,
                    font_size=extract_font_size(element.html),
                    is_all_caps=is_all_caps(element.text),
                    is_centered=is_centered(element.html),
                    is_bold=is_bold(element.html),
                )
            )

        return infos


class TitleDetectionProcessor(DocumentProcessor):
    """
    Processor that detects and extracts the document title from headings.

    This processor identifies the document title using heuristics:
    1. Must be on pages 1-N (early in the document)
    2. Should have level 1 (most prominent heading)
    3. Prefer centered headings
    4. Should have substantive text (not codes/dates)

    Once identified, the title is:
    - Set as document.title
    - Optionally removed from elements
    - Optionally triggers removal of preceding elements (metadata)
    - Optionally triggers heading level recalculation

    Example:
        ```python
        processor = TitleDetectionProcessor(
            remove_elements_before_title=True,
            recalculate_heading_levels=True,
        )
        doc = processor.process(doc)
        print(doc.title)  # The detected document title
        ```
    """

    def __init__(
        self,
        max_title_page: int = 4,
        remove_title_from_elements: bool = True,
        remove_elements_before_title: bool = False,
        recalculate_heading_levels: bool = True,
        min_title_length: int = 3,
    ):
        """
        Initialize the processor.

        Args:
            max_title_page: Maximum page number where title can appear (1-based)
            remove_title_from_elements: Whether to remove the title heading from elements
            remove_elements_before_title: Whether to remove elements before the title
            recalculate_heading_levels: Whether to shift heading levels after title removal
            min_title_length: Minimum character length for a valid title
        """
        self.max_title_page = max_title_page
        self.remove_title_from_elements = remove_title_from_elements
        self.remove_elements_before_title = remove_elements_before_title
        self.recalculate_heading_levels = recalculate_heading_levels
        self.min_title_length = min_title_length

    async def process(self, document: Document) -> Document:
        """Detect and extract the document title from headings."""

        # Skip if document already has a title
        if document.title:
            logger.debug("TitleDetectionProcessor: document already has title, skipping")
            return document

        # Find candidate title headings
        candidates = self._find_title_candidates(document)

        if not candidates:
            logger.debug("TitleDetectionProcessor: no title candidates found")
            return document

        # Select the best candidate
        best_idx, best_heading = self._select_best_candidate(candidates)

        if best_heading is None:
            return document

        # Set document title
        document.title = best_heading.text
        logger.info(f"TitleDetectionProcessor: detected title: {document.title!r}")

        # Track original level for recalculation
        title_level = best_heading.level

        # Remove elements before title first
        if self.remove_elements_before_title and best_idx > 0:
            del document.elements[:best_idx]
            best_idx = 0

        # Remove the title from elements
        if self.remove_title_from_elements:
            document.elements.pop(best_idx)

        # Recalculate heading levels
        if self.remove_title_from_elements and self.recalculate_heading_levels:
            self._recalculate_heading_levels(document, title_level)

        return document

    def _find_title_candidates(self, document: Document) -> list[tuple[int, Heading]]:
        """Find headings that could be document titles."""

        candidates: list[tuple[int, Heading]] = []

        for idx, element in enumerate(document.elements):
            if not isinstance(element, Heading):
                continue

            # Must be on early pages
            if element.page is not None and element.page > self.max_title_page:
                continue

            # Must have level 1
            if element.level != 1:
                continue

            # Must have substantive text
            text = element.text.strip()
            if len(text) < self.min_title_length:
                continue

            # Skip if looks like metadata
            if self._looks_like_metadata(text):
                continue

            candidates.append((idx, element))

        return candidates

    def _looks_like_metadata(self, text: str) -> bool:
        """Check if text looks like metadata rather than a title."""

        # Skip if mostly numbers/special characters
        letters = [c for c in text if c.isalpha()]
        if len(letters) < len(text) * 0.5:
            return True

        # Skip common metadata patterns
        metadata_patterns = [
            r"^\d{1,2}[-/]\w+[-/]\d{2,4}$",  # Dates
            r"^[A-Z]{2,4}[/-]\d+",  # Document codes
            r"^Page\s+\d+",  # Page numbers
            r"^\d+\s*$",  # Just numbers
        ]

        for pattern in metadata_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False

    def _select_best_candidate(
        self,
        candidates: list[tuple[int, Heading]],
    ) -> tuple[int, Heading] | tuple[None, None]:
        """Select the best title candidate based on scoring."""

        if not candidates:
            return None, None

        def score_candidate(item: tuple[int, Heading]) -> tuple[float, float, int]:
            """Score a candidate (higher is better)."""
            idx, heading = item

            # Extract visual properties using standalone functions
            font_size = extract_font_size(heading.html) or 0.0
            centered = is_centered(heading.html)

            centered_bonus = 1.0 if centered else 0.0

            return (font_size, centered_bonus, -idx)

        # Sort by score and return best
        candidates_sorted = sorted(candidates, key=score_candidate, reverse=True)
        return candidates_sorted[0]

    def _recalculate_heading_levels(self, document: Document, removed_title_level: int) -> None:
        """Recalculate heading levels after removing the title."""

        if removed_title_level != 1:
            return

        # Find minimum heading level
        min_level = None
        for element in document.elements:
            if isinstance(element, Heading) and element.level > 0:
                if min_level is None or element.level < min_level:
                    min_level = element.level

        if min_level is None or min_level == 1:
            return

        # Shift all headings up
        level_shift = min_level - 1
        for element in document.elements:
            if isinstance(element, Heading) and element.level > 0:
                element.level = max(1, element.level - level_shift)


# =============================================================================
# Heading Level Normalization
# =============================================================================


def normalize_heading_levels(document: Document) -> Document:
    """
    Remap heading levels to a compact gap-free sequence.

    Collects all distinct levels present, sorts them, and remaps each to its
    1-based rank — so any gaps (e.g. h1 → h3) are eliminated.

    Args:
        document: Document whose heading levels will be normalized in-place.

    Returns:
        The same document with heading levels compacted.

    Example:
        >>> # Levels [1, 3, 3, 5] become [1, 2, 2, 3]
    """
    levels_present = sorted({el.level for el in document.elements if isinstance(el, Heading) and el.level > 0})
    if not levels_present:
        return document

    remap: dict[int, int] = {level: rank for rank, level in enumerate(levels_present, start=1)}
    for el in document.elements:
        if isinstance(el, Heading) and el.level > 0:
            el.level = remap[el.level]
    return document


class HeadingNormalizationProcessor(DocumentProcessor):
    """
    Processor that compacts heading levels to eliminate gaps.

    After parsing or level reassignment, heading levels can jump non-contiguously
    (e.g. h1 → h3 with no h2). This processor remaps all levels to a compact
    sequence while preserving their relative order.

    Run this as the last step in any heading-processing chain.

    Example:
        ```python
        pipeline = ProcessingPipeline([
            HeadingLevelProcessor(),
            TitleDetectionProcessor(),
            HeadingNormalizationProcessor(),
        ])
        ```
    """

    async def process(self, document: Document) -> Document:
        """Normalize heading levels to eliminate gaps."""
        return normalize_heading_levels(document)
