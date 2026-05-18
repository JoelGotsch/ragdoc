"""Tests for latex_to_mathml and the MathML-aware extract_text_from_lines."""

import pytest
from bs4 import BeautifulSoup

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")
pytest.importorskip("latex2mathml", reason="pdf_mineru extra not installed")

from ragdoc.parsing.mineru.base import (
    InlineEquationSpan,
    InterlineEquationSpan,
    Line,
    TextSpan,
    extract_text_from_lines,
    normalize_math_spaces,
    latex_to_mathml,
)

# ---------------------------------------------------------------------------
# normalize_math_spaces unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latex, expected",
    [
        # OCR-fragmented digit runs → collapsed
        ("2 0 0 4", "2004"),
        ("1 2", "12"),
        ("1 2 3 4 5", "12345"),
        # Digits in exponent: { 1 2 } → {12}
        (r"2 0 0 4 ^ { 1 2 }", r"2004 ^ { 12 }"),
        # Mixed: letters between digits are NOT collapsed
        (r"\mathrm { I 1 1 }", r"\mathrm { I 11 }"),
        # Decimal point between digits is collapsed too
        (r"1 . 2 \%", r"1.2 \%"),
        # Already compact — unchanged
        (r"\mathrm{SF_6}", r"\mathrm{SF_6}"),
        ("", ""),
    ],
    ids=[
        "four_digits",
        "two_digits",
        "five_digits",
        "year_with_exponent",
        "letter_stops_run",
        "decimal_point_stops_run",
        "already_compact",
        "empty",
    ],
)
def test_normalize_math_spaces(latex: str, expected: str) -> None:
    assert normalize_math_spaces(latex) == expected


# Structural: fragmented digits produce a single <mn> node after normalization
@pytest.mark.parametrize(
    "latex, expected_number",
    [
        (r"2 0 0 4 ^ { 1 2 }", "12"),   # footnote number in exponent
        (r"\mathrm { I 1 1 }", "11"),   # footnote number after letter
    ],
    ids=["year_footnote_exponent", "heu_footnote_exponent"],
)
def test_fragmented_digits_become_single_mn_node(latex: str, expected_number: str) -> None:
    """After space normalization, footnote numbers must appear as a single <mn> text node."""
    result = latex_to_mathml(latex, display=False)
    soup = BeautifulSoup(result, "html.parser")
    mn_texts = [tag.get_text() for tag in soup.find_all("mn")]
    assert expected_number in mn_texts, (
        f"Expected <mn>{expected_number}</mn> in MathML, got mn nodes: {mn_texts}\nMathML: {result}"
    )


# ---------------------------------------------------------------------------
# latex_to_mathml unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latex, display",
    [
        (r"\mathrm{SF_6}", False),
        (r"H_2O", False),
        (r"\alpha + \beta", False),
        (r"\int_0^\infty f(x)\,dx", True),
        (r"x^{2} + y^{2} = z^{2}", False),
    ],
    ids=[
        "mathrm_SF6",
        "subscript_H2O",
        "greek_letters",
        "integral_display",
        "pythagorean",
    ],
)
def test_latex_to_mathml_returns_math_element(latex: str, display: bool) -> None:
    """Output must be a <math> element."""
    result = latex_to_mathml(latex, display=display)
    assert result.startswith("<math ")
    soup = BeautifulSoup(result, "html.parser")
    assert soup.find("math") is not None


def test_latex_to_mathml_empty_returns_empty_string() -> None:
    """Empty input must return an empty string (latex2mathml raises on empty)."""
    assert latex_to_mathml("") == ""
    assert latex_to_mathml("", display=True) == ""


@pytest.mark.parametrize(
    "latex, display, expected_display_attr",
    [
        (r"\mathrm{SF_6}", False, "inline"),
        (r"\int_0^\infty f(x)\,dx", True, "block"),
    ],
    ids=["inline", "block"],
)
def test_latex_to_mathml_display_attribute(
    latex: str, display: bool, expected_display_attr: str
) -> None:
    """<math> must carry the correct display= attribute."""
    result = latex_to_mathml(latex, display=display)
    soup = BeautifulSoup(result, "html.parser")
    math_tag = soup.find("math")
    assert math_tag is not None
    assert math_tag.get("display") == expected_display_attr


def test_latex_to_mathml_no_alttext() -> None:
    """<math> must NOT carry an alttext attribute (plain MathML only)."""
    result = latex_to_mathml(r"\mathrm{SF_6}", display=False)
    soup = BeautifulSoup(result, "html.parser")
    math_tag = soup.find("math")
    assert math_tag is not None
    assert math_tag.get("alttext") is None


def test_latex_to_mathml_unknown_command_does_not_raise() -> None:
    """latex2mathml never raises on unknown commands — it emits <mi> nodes."""
    result = latex_to_mathml(r"\unknowncmd{x}", display=False)
    assert "<math" in result


# ---------------------------------------------------------------------------
# extract_text_from_lines integration tests
# ---------------------------------------------------------------------------

_DUMMY_BBOX: list[float] = [0.0, 0.0, 100.0, 20.0]


def _make_line(*spans: TextSpan | InlineEquationSpan | InterlineEquationSpan) -> Line:
    return Line(bbox=_DUMMY_BBOX, spans=list(spans))


def test_extract_text_inline_equation_produces_math_tag() -> None:
    """InlineEquationSpan must become a <math display="inline"> element."""
    line = _make_line(InlineEquationSpan(bbox=_DUMMY_BBOX, content=r"\mathrm{SF_6}"))
    result = extract_text_from_lines([line])
    soup = BeautifulSoup(result, "html.parser")
    math = soup.find("math")
    assert math is not None
    assert math.get("display") == "inline"
    assert math.get("alttext") is None


def test_extract_text_interline_equation_produces_block_math_tag() -> None:
    """InterlineEquationSpan must become a <math display="block"> element."""
    line = _make_line(
        InterlineEquationSpan(bbox=_DUMMY_BBOX, content=r"\int_0^\infty f(x)\,dx")
    )
    result = extract_text_from_lines([line])
    soup = BeautifulSoup(result, "html.parser")
    math = soup.find("math")
    assert math is not None
    assert math.get("display") == "block"


def test_extract_text_mixed_line_preserves_text_and_math() -> None:
    """A line mixing TextSpan and InlineEquationSpan must keep both parts."""
    line = _make_line(
        TextSpan(bbox=_DUMMY_BBOX, content="The compound"),
        InlineEquationSpan(bbox=_DUMMY_BBOX, content=r"\mathrm{SF_6}"),
        TextSpan(bbox=_DUMMY_BBOX, content="is reactive."),
    )
    result = extract_text_from_lines([line])
    assert "The compound" in result
    assert "is reactive." in result
    soup = BeautifulSoup(result, "html.parser")
    assert soup.find("math") is not None


def test_extract_text_multiple_equations_in_one_line() -> None:
    """Multiple equation spans in one line must all become <math> elements."""
    line = _make_line(
        InlineEquationSpan(bbox=_DUMMY_BBOX, content=r"\alpha"),
        TextSpan(bbox=_DUMMY_BBOX, content="and"),
        InlineEquationSpan(bbox=_DUMMY_BBOX, content=r"\beta"),
    )
    result = extract_text_from_lines([line])
    soup = BeautifulSoup(result, "html.parser")
    math_tags = soup.find_all("math")
    assert len(math_tags) == 2


def test_extract_text_no_equations_returns_plain_text() -> None:
    """Lines with only TextSpans must not produce any <math> elements."""
    line = _make_line(
        TextSpan(bbox=_DUMMY_BBOX, content="Hello"),
        TextSpan(bbox=_DUMMY_BBOX, content="world"),
    )
    result = extract_text_from_lines([line])
    assert "<math" not in result
    assert "Hello" in result
    assert "world" in result
