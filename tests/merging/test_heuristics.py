"""Tests for merging/heuristics.py -- HTML selection heuristics."""
import pytest

from ragdoc.document import DocumentList, Footnote, Heading, Paragraph, Table
from ragdoc.merging.heuristics import (
    markup_richness_score,
    select_footnote,
    select_heading,
    select_paragraph,
    select_table,
)


# ---------------------------------------------------------------------------
# --- TestMarkupRichnessScore ---
# ---------------------------------------------------------------------------


def test_markup_richness_plain_text_scores_zero():
    assert markup_richness_score("<p>plain text</p>") == 0


@pytest.mark.parametrize(
    "html",
    [
        pytest.param("<p><b>bold</b></p>", id="bold"),
        pytest.param("<p><strong>bold</strong></p>", id="strong"),
        pytest.param("<p><em>italic</em></p>", id="em"),
        pytest.param("<p><i>italic</i></p>", id="italic"),
        pytest.param("<p><math>x\u00b2</math></p>", id="math"),
        pytest.param("<p>H<sub>2</sub>O</p>", id="sub"),
        pytest.param("<p>x<sup>2</sup></p>", id="sup"),
        pytest.param("<p><code>fn()</code></p>", id="code"),
        pytest.param('<p><span style="color:red">red</span></p>', id="span_with_style"),
    ],
)
def test_markup_richness_tag_scores_positive(html):
    assert markup_richness_score(html) > 0


def test_markup_richness_span_without_style_scores_zero():
    assert markup_richness_score("<p><span>no style</span></p>") == 0


def test_markup_richness_richer_html_scores_higher():
    plain = "<p>plain text here</p>"
    rich = "<p><b>bold</b> text with <em>italic</em></p>"
    assert markup_richness_score(rich) > markup_richness_score(plain)


def test_markup_richness_math_formula_scores_high():
    plain = "<p>some formula result</p>"
    math = "<p><math><mrow><msup><mi>x</mi><mn>2</mn></msup></mrow></math></p>"
    assert markup_richness_score(math) > markup_richness_score(plain)


def test_markup_richness_multiple_tags_accumulate():
    one_tag = "<p><b>one</b></p>"
    two_tags = "<p><b>one</b> and <em>two</em></p>"
    assert markup_richness_score(two_tags) > markup_richness_score(one_tag)


# ---------------------------------------------------------------------------
# --- TestSelectHeading ---
# ---------------------------------------------------------------------------


def test_select_heading_trusted_parser_level_wins():
    """html parser is in trust_parsers -> its heading level wins."""
    h_a = Heading(html_content="<h3>Section</h3>")
    h_a.metadata["parser"] = "mineru"
    h_b = Heading(html_content="<h2>Section</h2>")
    h_b.metadata["parser"] = "html"

    result = select_heading(h_a, h_b, parser_a="mineru", parser_b="html")
    assert result.level == 2


def test_select_heading_untrusted_parsers_keeps_innerhtml_from_richer_side():
    """Neither parser is trusted -> keep innerhtml from richer side."""
    h_a = Heading(html_content="<h2>Plain heading</h2>")
    h_b = Heading(html_content="<h3><strong>Bold</strong> heading</h3>")

    result = select_heading(h_a, h_b, parser_a="mineru", parser_b="mineru")
    # h_b has richer markup; result should have h_b's innerhtml
    assert "<strong>" in result.html


def test_select_heading_trust_parsers_customizable():
    """Custom trust_parsers set."""
    h_a = Heading(html_content="<h1>Title</h1>")
    h_b = Heading(html_content="<h3>Title</h3>")
    # Only "azure_di" is trusted
    result = select_heading(
        h_a, h_b,
        parser_a="azure_di", parser_b="mineru",
        trust_parsers=frozenset({"azure_di"}),
    )
    assert result.level == 1  # from azure_di (trusted)


def test_select_heading_both_trusted_uses_lower_level_number():
    """Both parsers trusted -> lower level number (more prominent heading) wins."""
    h_a = Heading(html_content="<h1>Title</h1>")
    h_b = Heading(html_content="<h2>Title</h2>")
    result = select_heading(
        h_a, h_b,
        parser_a="html", parser_b="pandoc",
        trust_parsers=frozenset({"html", "pandoc"}),
    )
    assert result.level == 1


def test_select_heading_neither_trusted_tiebreak_by_level():
    """Neither trusted, equal richness -> lower level number wins."""
    h_a = Heading(html_content="<h2>Plain</h2>")
    h_b = Heading(html_content="<h4>Plain</h4>")
    result = select_heading(h_a, h_b, parser_a="mineru", parser_b="mineru")
    assert result.level == 2


# ---------------------------------------------------------------------------
# --- TestSelectParagraph ---
# ---------------------------------------------------------------------------


def test_select_paragraph_richer_markup_wins():
    p_a = Paragraph(html_content="<p>plain text</p>")
    p_b = Paragraph(html_content="<p><em>italic text</em></p>")
    result = select_paragraph(p_a, p_b)
    assert result.id == p_b.id


def test_select_paragraph_equal_richness_tiebreak_by_text_length():
    p_a = Paragraph(html_content="<p>short</p>")
    p_b = Paragraph(html_content="<p>much longer text here</p>")
    result = select_paragraph(p_a, p_b)
    assert result.id == p_b.id


def test_select_paragraph_a_richer_than_b():
    p_a = Paragraph(html_content="<p><b>bold</b> and <em>italic</em></p>")
    p_b = Paragraph(html_content="<p>plain</p>")
    result = select_paragraph(p_a, p_b)
    assert result.id == p_a.id


def test_select_paragraph_math_formula_preferred():
    p_plain = Paragraph(html_content="<p>E equals mc squared</p>")
    p_math = Paragraph(html_content="<p>E = <math>mc\u00b2</math></p>")
    result = select_paragraph(p_plain, p_math)
    assert result.id == p_math.id


# ---------------------------------------------------------------------------
# --- TestSelectTable ---
# ---------------------------------------------------------------------------


def test_select_table_more_th_elements_wins():
    t_a = Table(html_content="<table><tr><td>A</td><td>B</td></tr></table>")
    t_b = Table(html_content="<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    result = select_table(t_a, t_b)
    assert result.id == t_b.id


def test_select_table_equal_headers_tiebreak_by_row_count():
    t_a = Table(html_content="<table><tr><td>A</td></tr></table>")
    t_b = Table(
        html_content="<table><tr><td>A</td></tr><tr><td>B</td></tr><tr><td>C</td></tr></table>"
    )
    result = select_table(t_a, t_b)
    assert result.id == t_b.id


def test_select_table_a_has_more_headers():
    t_a = Table(html_content="<table><tr><th>X</th><th>Y</th><th>Z</th></tr></table>")
    t_b = Table(html_content="<table><tr><td>1</td><td>2</td></tr></table>")
    result = select_table(t_a, t_b)
    assert result.id == t_a.id


# ---------------------------------------------------------------------------
# --- TestSelectFootnote ---
# ---------------------------------------------------------------------------


def test_select_footnote_longer_innerhtml_wins():
    f_a = Footnote(number=1, innerhtml="Short.")
    f_b = Footnote(number=1, innerhtml="Much longer and more complete footnote text.")
    result = select_footnote(f_a, f_b)
    assert result.id == f_b.id


def test_select_footnote_a_longer_than_b():
    f_a = Footnote(number=1, innerhtml="Longer text with more detail here.")
    f_b = Footnote(number=1, innerhtml="Tiny.")
    result = select_footnote(f_a, f_b)
    assert result.id == f_a.id


def test_select_footnote_equal_length_returns_a():
    f_a = Footnote(number=1, innerhtml="Same.")
    f_b = Footnote(number=1, innerhtml="Same.")
    result = select_footnote(f_a, f_b)
    assert result.id == f_a.id
