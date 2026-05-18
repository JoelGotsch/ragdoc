"""Tests for heading processors and related helpers."""
import pytest

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.processing.heading import (
    HeadingLevelProcessor,
    HeadingNormalizationProcessor,
    HeadingVisualInfo,
    TitleDetectionProcessor,
    compute_size_to_level_mapping,
    extract_font_size,
    is_all_caps,
    is_bold,
    is_centered,
    normalize_heading_levels,
)


# --- TestHeadingVisualInfo ---


@pytest.mark.parametrize(
    "font_size, is_all_caps_val, is_centered_val, is_bold_val, expected",
    [
        pytest.param(14.0, False, False, False, 14.0, id="basic_no_boosts"),
        pytest.param(14.0, True, False, False, 14.0 * 1.2, id="all_caps_boost"),
        pytest.param(14.0, False, True, False, 14.0 * 1.2, id="centered_boost"),
        pytest.param(14.0, False, False, True, 14.0 * 1.1, id="bold_boost"),
        pytest.param(10.0, True, True, True, 10.0 * 1.2 * 1.2 * 1.1, id="combined_boosts"),
        pytest.param(None, True, True, True, 0.0, id="none_font"),
    ],
)
def test_heading_visual_info_effective_size(
    font_size, is_all_caps_val, is_centered_val, is_bold_val, expected
):
    info = HeadingVisualInfo(
        index=0,
        font_size=font_size,
        is_all_caps=is_all_caps_val,
        is_centered=is_centered_val,
        is_bold=is_bold_val,
    )
    assert info.effective_size == pytest.approx(expected, rel=0.01)


# --- TestExtractFontSize ---


@pytest.mark.parametrize(
    "html, expected",
    [
        pytest.param('<span style="font-size: 14pt;">Text</span>', 14.0, id="pt"),
        pytest.param('<span style="font-size: 20px;">Text</span>', 15.0, id="px"),
        pytest.param('<span style="font-size: 2em;">Text</span>', 24.0, id="em"),
        pytest.param('<span style="font-size: 1.5rem;">Text</span>', 18.0, id="rem"),
        pytest.param('<span>Text</span>', None, id="no_font_size"),
        pytest.param('Plain text', None, id="plain_text"),
        pytest.param(
            '<div><span style="font-size: 18pt;">Nested</span></div>', 18.0, id="nested"
        ),
    ],
)
def test_extract_font_size(html, expected):
    result = extract_font_size(html)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# --- TestIsCentered ---


def test_is_centered_with_space():
    """Detects text-align: center with space."""
    assert is_centered('<span style="text-align: center;">Text</span>')


def test_is_centered_without_space():
    """Detects text-align:center without space."""
    assert is_centered('<span style="text-align:center;">Text</span>')


def test_is_centered_not_centered():
    """Returns False for non-centered text."""
    assert not is_centered('<span style="text-align: left;">Text</span>')
    assert not is_centered('<span>Text</span>')


def test_is_centered_in_div():
    """Detects centering in div element."""
    assert is_centered('<div style="text-align: center;">Content</div>')


# --- TestIsBold ---


def test_is_bold_tag():
    """Detects <b> tag."""
    assert is_bold('<b>Bold</b>')


def test_is_bold_strong_tag():
    """Detects <strong> tag."""
    assert is_bold('<strong>Strong</strong>')


def test_is_bold_css_keyword():
    """Detects font-weight: bold in CSS."""
    assert is_bold('<span style="font-weight: bold;">Text</span>')
    assert is_bold('<span style="font-weight:bold;">Text</span>')


def test_is_bold_css_numeric():
    """Detects numeric font-weight >= 600."""
    assert is_bold('<span style="font-weight: 700;">Text</span>')
    assert is_bold('<span style="font-weight: 600;">Text</span>')
    assert not is_bold('<span style="font-weight: 400;">Text</span>')


def test_is_bold_not_bold():
    """Returns False for non-bold text."""
    assert not is_bold('<span>Normal</span>')


# --- TestIsAllCaps ---


def test_is_all_caps_all_caps():
    """Detects all caps text."""
    assert is_all_caps("HELLO WORLD")
    assert is_all_caps("ABC")


def test_is_all_caps_mixed_with_numbers():
    """All caps with numbers."""
    assert is_all_caps("ABC 123")
    assert is_all_caps("2024 REPORT")


def test_is_all_caps_not_all_caps():
    """Returns False for mixed case."""
    assert not is_all_caps("Hello World")
    assert not is_all_caps("hello")
    assert not is_all_caps("HELLO world")


def test_is_all_caps_empty_string():
    """Returns False for empty string (no letters)."""
    assert not is_all_caps("")
    assert not is_all_caps("123")


# --- TestComputeSizeToLevelMapping ---


def test_compute_size_to_level_mapping_basic():
    """Basic mapping with 3 unique sizes."""
    sizes = [24.0, 18.0, 12.0]
    mapping = compute_size_to_level_mapping(sizes)

    assert mapping[24.0] == 1  # Largest = h1
    assert mapping[18.0] == 2  # Medium = h2
    assert mapping[12.0] == 3  # Smallest = h3


def test_compute_size_to_level_mapping_duplicate_sizes():
    """Handles duplicate sizes correctly."""
    sizes = [24.0, 24.0, 18.0, 12.0, 12.0]
    mapping = compute_size_to_level_mapping(sizes)

    assert len(mapping) == 3
    assert mapping[24.0] == 1
    assert mapping[18.0] == 2
    assert mapping[12.0] == 3


def test_compute_size_to_level_mapping_empty_list():
    """Returns empty dict for empty list."""
    assert compute_size_to_level_mapping([]) == {}


def test_compute_size_to_level_mapping_zero_sizes_ignored():
    """Zero sizes are ignored."""
    sizes = [24.0, 0.0, 18.0, 0.0]
    mapping = compute_size_to_level_mapping(sizes)

    assert 0.0 not in mapping
    assert mapping[24.0] == 1
    assert mapping[18.0] == 2


def test_compute_size_to_level_mapping_compression():
    """More sizes than levels get compressed."""
    sizes = [30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0]
    mapping = compute_size_to_level_mapping(sizes, max_levels=4)

    # All sizes should map to levels 1-4
    assert all(1 <= level <= 4 for level in mapping.values())
    # Largest should be h1
    assert mapping[30.0] == 1


def test_compute_size_to_level_mapping_custom_max_levels():
    """Respects custom max_levels parameter."""
    sizes = [24.0, 18.0, 12.0]
    mapping = compute_size_to_level_mapping(sizes, max_levels=2)

    # Should compress to 2 levels
    assert all(1 <= level <= 2 for level in mapping.values())


# --- TestHeadingLevelProcessor ---


def _make_heading_level_doc(parser: str, elements: list) -> Document:
    return Document(parser=parser, elements=elements)


@pytest.mark.anyio
async def test_heading_level_processor_skips_html_parser():
    """Processor skips documents from HTML parser (already reliable)."""
    doc = Document(
        parser="html",
        elements=[
            Heading(innerhtml="Title", level=1),
            Heading(innerhtml="Section", level=2),
        ]
    )

    processor = HeadingLevelProcessor()
    result = await processor.process(doc)

    assert result.elements[0].level == 1
    assert result.elements[1].level == 2


@pytest.mark.anyio
async def test_heading_level_processor_skips_pandoc_parser():
    """Processor skips documents from Pandoc parser (already reliable)."""
    doc = Document(
        parser="pandoc",
        elements=[
            Heading(innerhtml="Title", level=1),
        ]
    )

    processor = HeadingLevelProcessor()
    result = await processor.process(doc)

    assert result.elements[0].level == 1


@pytest.mark.anyio
async def test_heading_level_processor_processes_mineru_parser():
    """Processor processes documents from MinerU parser."""
    doc = Document(
        parser="mineru",
        elements=[
            Heading(
                innerhtml='<span style="font-size: 24pt;">Big Title</span>',
                level=1
            ),
            Heading(
                innerhtml='<span style="font-size: 14pt;">Small Section</span>',
                level=1
            ),
        ]
    )

    processor = HeadingLevelProcessor()
    result = await processor.process(doc)

    assert result.elements[0].level == 1
    assert result.elements[1].level == 2


@pytest.mark.anyio
async def test_heading_level_processor_custom_size_to_level_mapper():
    """Processor accepts custom size_to_level_mapper function."""
    doc = Document(
        parser="mineru",
        elements=[
            Heading(
                innerhtml='<span style="font-size: 24pt;">Big Title</span>',
                level=1
            ),
            Heading(
                innerhtml='<span style="font-size: 14pt;">Small Section</span>',
                level=1
            ),
        ]
    )

    def custom_mapper(sizes: list[float], max_levels: int) -> dict[float, int]:
        return {size: 3 for size in sizes if size > 0}

    processor = HeadingLevelProcessor(size_to_level_mapper=custom_mapper)
    result = await processor.process(doc)

    assert result.elements[0].level == 3
    assert result.elements[1].level == 3


@pytest.mark.anyio
async def test_heading_level_processor_custom_mapper_called_with_effective_sizes():
    """Custom mapper receives effective sizes (with boosts applied)."""
    doc = Document(
        parser="mineru",
        elements=[
            Heading(
                innerhtml='<span style="font-size: 10pt; text-align: center;">CENTERED CAPS</span>',
                level=1
            ),
        ]
    )

    received_sizes = []

    def capture_mapper(sizes: list[float], max_levels: int) -> dict[float, int]:
        received_sizes.extend(sizes)
        return compute_size_to_level_mapping(sizes, max_levels)

    processor = HeadingLevelProcessor(size_to_level_mapper=capture_mapper)
    await processor.process(doc)

    # Should receive effective size with all-caps (1.2) and centered (1.2) boosts
    # 10 * 1.2 * 1.2 = 14.4
    assert len(received_sizes) == 1
    assert received_sizes[0] == pytest.approx(14.4, rel=0.01)


# --- TestTitleDetectionProcessor ---


@pytest.mark.anyio
async def test_title_detection_skips_if_title_exists():
    """Processor skips if document already has a title."""
    doc = Document(
        title="Existing Title",
        elements=[
            Heading(innerhtml="Another Title", level=1),
        ]
    )

    processor = TitleDetectionProcessor()
    result = await processor.process(doc)

    assert result.title == "Existing Title"
    assert len(result.elements) == 1  # Heading not removed


@pytest.mark.anyio
async def test_title_detection_detects_simple_title():
    """Processor detects and extracts a simple title."""
    doc = Document(
        elements=[
            Heading(innerhtml="Document Title", level=1, page=1),
            Paragraph(html="<p>Some content</p>"),
        ]
    )

    processor = TitleDetectionProcessor()
    result = await processor.process(doc)

    assert result.title == "Document Title"
    assert len(result.elements) == 1  # Title removed


@pytest.mark.anyio
async def test_title_detection_skips_metadata_looking_headings():
    """Processor skips headings that look like metadata."""
    doc = Document(
        elements=[
            Heading(innerhtml="REF-2024/123", level=1, page=1),  # Doc code
            Heading(innerhtml="Real Document Title", level=1, page=1),
            Paragraph(html="<p>Content</p>"),
        ]
    )

    processor = TitleDetectionProcessor()
    result = await processor.process(doc)

    assert result.title == "Real Document Title"


@pytest.mark.anyio
async def test_title_detection_remove_elements_before_title():
    """Processor can remove elements before the title."""
    doc = Document(
        elements=[
            Paragraph(html="<p>Metadata stuff</p>"),
            Heading(innerhtml="12-March-2024", level=2, page=1),
            Heading(innerhtml="The Real Title", level=1, page=1),
            Paragraph(html="<p>Content</p>"),
        ]
    )

    processor = TitleDetectionProcessor(
        remove_elements_before_title=True,
        remove_title_from_elements=True,
    )
    result = await processor.process(doc)

    assert result.title == "The Real Title"
    assert len(result.elements) == 1
    assert result.elements[0].html == "<p>Content</p>"


@pytest.mark.anyio
async def test_title_detection_recalculates_heading_levels():
    """Processor shifts heading levels after title removal."""
    doc = Document(
        elements=[
            Heading(innerhtml="Title", level=1, page=1),
            Heading(innerhtml="Section", level=2),
            Heading(innerhtml="Subsection", level=3),
        ]
    )

    processor = TitleDetectionProcessor(recalculate_heading_levels=True)
    result = await processor.process(doc)

    # Title removed, h2 -> h1, h3 -> h2
    assert result.title == "Title"
    assert len(result.elements) == 2
    assert result.elements[0].level == 1  # Was h2
    assert result.elements[1].level == 2  # Was h3


@pytest.mark.anyio
async def test_title_detection_respects_max_title_page():
    """Processor only considers headings on early pages."""
    doc = Document(
        elements=[
            Heading(innerhtml="Late Title", level=1, page=10),
        ]
    )

    processor = TitleDetectionProcessor(max_title_page=4)
    result = await processor.process(doc)

    assert result.title is None
    assert len(result.elements) == 1


@pytest.mark.anyio
async def test_title_detection_respects_min_title_length():
    """Processor skips very short headings."""
    doc = Document(
        elements=[
            Heading(innerhtml="OK", level=1, page=1),  # Too short
            Heading(innerhtml="A Real Title Here", level=1, page=1),
        ]
    )

    processor = TitleDetectionProcessor(min_title_length=3)
    result = await processor.process(doc)

    assert result.title == "A Real Title Here"


# --- TestNormalizeHeadingLevels ---


def _make_doc_with_heading_levels(levels: list[int]) -> Document:
    return Document(elements=[Heading(innerhtml=f"H{l}", level=l) for l in levels])


def test_normalize_heading_levels_no_gaps_unchanged():
    doc = _make_doc_with_heading_levels([1, 2, 3])
    result = normalize_heading_levels(doc)
    assert [el.level for el in result.elements] == [1, 2, 3]


def test_normalize_heading_levels_gap_after_h1():
    """h1 -> h3 should become h1 -> h2."""
    doc = _make_doc_with_heading_levels([1, 3])
    result = normalize_heading_levels(doc)
    assert [el.level for el in result.elements] == [1, 2]


def test_normalize_heading_levels_multiple_gaps():
    """[1, 3, 5] should compact to [1, 2, 3]."""
    doc = _make_doc_with_heading_levels([1, 3, 5])
    result = normalize_heading_levels(doc)
    assert [el.level for el in result.elements] == [1, 2, 3]


def test_normalize_heading_levels_duplicate_levels_preserved():
    """Headings sharing a level stay at the same remapped level."""
    doc = _make_doc_with_heading_levels([1, 3, 3, 5])
    result = normalize_heading_levels(doc)
    assert [el.level for el in result.elements] == [1, 2, 2, 3]


def test_normalize_heading_levels_no_headings():
    doc = Document(elements=[Paragraph(innerhtml="text")])
    result = normalize_heading_levels(doc)
    assert result is doc


@pytest.mark.anyio
async def test_normalize_heading_levels_processor_delegates():
    doc = _make_doc_with_heading_levels([1, 3, 5])
    result = await HeadingNormalizationProcessor().process(doc)
    assert [el.level for el in result.elements] == [1, 2, 3]
