"""HTML selection heuristics for the document merge pipeline.

All functions are standalone so they can be tested and reused independently.
Reuses ``extract_font_size``, ``is_bold``, ``is_centered`` from
``ragdoc.processing.heading`` — no duplication.

Both Approach A (element-alignment) and Approach B (render-merge-reparse) use
these heuristics to pick the richer or more accurate element when two aligned
elements differ.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from ragdoc.document import Document, Footnote, Heading, Paragraph, Table

# Tags whose presence signals rich inline markup (each occurrence adds 1 to the score).
_RICH_TAGS = frozenset({"b", "strong", "em", "i", "math", "sub", "sup", "code"})


def has_heading_hierarchy(doc: Document) -> bool:
    """Return True if the document uses more than one distinct heading level.

    A document that emits all headings at the same level clearly did not
    perform hierarchy detection; one with multiple distinct levels did.
    """
    return len({h.level for h in doc.headings}) > 1


def markup_richness_score(html: str) -> int:
    """Count inline markup tags as a proxy for content richness.

    Counts occurrences of: ``<b>``, ``<strong>``, ``<em>``, ``<i>``, ``<math>``,
    ``<sub>``, ``<sup>``, ``<code>``, and ``<span>`` elements that carry a
    ``style`` attribute.

    ``<ref>`` placeholders (inline footnote/image references) count as 2 points
    each — they signal properly structured cross-references, which are more
    valuable than inline formatting alone.

    Higher scores indicate more preserved formatting or structural richness.

    Args:
        html: HTML string to score.

    Returns:
        Non-negative integer; 0 for plain text.
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0
    for tag in soup.find_all(True):
        if tag.name in _RICH_TAGS:
            score += 1
        elif tag.name == "span" and tag.get("style"):
            score += 1
        elif tag.name == "ref":
            score += 2
    return score


def inject_inline_markup(base_html: str, source_html: str) -> str:
    """Inject inline markup from *source_html* into *base_html* for matching words.

    For each text span in *source_html* that is directly wrapped in an inline
    markup tag (``<strong>``, ``<em>``, ``<b>``, ``<i>``, ``<code>``,
    ``<sub>``, ``<sup>``), if the same plain text appears as unformatted text
    in *base_html*, it is wrapped in the same tag.

    Only single-level markup is injected (not deeply nested structures).
    Injections that would create duplicate markup are skipped.

    Args:
        base_html: The winning element's HTML — structure preserved, markup enhanced.
        source_html: The losing element's HTML — supplies inline markup to carry over.

    Returns:
        A (possibly modified) HTML string.
    """
    soup = BeautifulSoup(source_html, "html.parser")
    result = base_html
    for tag in soup.find_all(_RICH_TAGS):
        text = tag.get_text()
        if not text or not text.strip():
            continue
        already_marked = f"<{tag.name}>{text}</{tag.name}>"
        if text in result and already_marked not in result:
            result = result.replace(text, already_marked, 1)
    return result


def select_heading(
    h_a: Heading,
    h_b: Heading,
    parser_a: str | None = None,
    parser_b: str | None = None,
    trust_parsers: frozenset[str] = frozenset({"html", "pandoc"}),
) -> Heading:
    """Select the better heading from two aligned headings.

    Priority:
    1. If exactly one parser is in ``trust_parsers``, use that parser's
       heading level; take innerhtml from the side with higher markup richness.
    2. If both parsers are trusted, use the lower level number (more prominent)
       and take innerhtml from the richer side.
    3. If neither is trusted, prefer the side with higher markup richness for
       innerhtml; tiebreak level by the lower number (more prominent).

    Args:
        h_a: Heading from document A.
        h_b: Heading from document B.
        parser_a: Parser name that produced h_a (e.g. ``"mineru"``).
        parser_b: Parser name that produced h_b (e.g. ``"html"``).
        trust_parsers: Set of parser names whose heading levels are reliable.

    Returns:
        A new Heading combining the winning level and the richer innerhtml.
    """
    a_trusted = parser_a in trust_parsers
    b_trusted = parser_b in trust_parsers

    # Determine winning level
    if a_trusted and not b_trusted:
        winning_level = h_a.level
    elif b_trusted and not a_trusted:
        winning_level = h_b.level
    else:
        # Both trusted or neither: lower level number = more prominent heading
        winning_level = min(h_a.level, h_b.level)

    # Determine winning innerhtml by markup richness
    score_a = markup_richness_score(h_a.html)
    score_b = markup_richness_score(h_b.html)
    winning_inner = h_b.innerhtml if score_b > score_a else h_a.innerhtml

    # Build result: apply winning level to winning innerhtml
    result = Heading(html=f"<h{winning_level}>{winning_inner}</h{winning_level}>")
    return result


def select_paragraph(p_a: Paragraph, p_b: Paragraph) -> Paragraph:
    """Select the paragraph with richer inline markup.

    Args:
        p_a: Paragraph from document A.
        p_b: Paragraph from document B.

    Returns:
        The paragraph with the higher markup richness score.
        Tiebreak: longer text length wins; further tiebreak: p_a.
    """
    score_a = markup_richness_score(p_a.html)
    score_b = markup_richness_score(p_b.html)

    if score_b > score_a:
        return p_b
    if score_a > score_b:
        return p_a
    # Equal richness: prefer longer text
    return p_b if len(p_b.text) > len(p_a.text) else p_a


def _count_table_headers(html: str) -> int:
    """Count ``<th>`` elements in a table HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all("th"))


def _count_table_rows(html: str) -> int:
    """Count ``<tr>`` elements in a table HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all("tr"))


def select_table(t_a: Table, t_b: Table) -> Table:
    """Select the table with better structural quality.

    Prefers the table with more ``<th>`` header cells (header detection proxy).
    Tiebreak: more ``<tr>`` rows (more content).  Further tiebreak: t_a.

    Args:
        t_a: Table from document A.
        t_b: Table from document B.

    Returns:
        The table with higher structural quality.
    """
    headers_a = _count_table_headers(t_a.html)
    headers_b = _count_table_headers(t_b.html)

    if headers_b > headers_a:
        return t_b
    if headers_a > headers_b:
        return t_a

    # Equal headers: prefer more rows
    rows_a = _count_table_rows(t_a.html)
    rows_b = _count_table_rows(t_b.html)
    return t_b if rows_b > rows_a else t_a


def select_footnote(f_a: Footnote, f_b: Footnote) -> Footnote:
    """Select the footnote with more complete text.

    Prefers the footnote whose ``innerhtml`` is longer (more complete).
    Tiebreak: f_a.

    Args:
        f_a: Footnote from document A.
        f_b: Footnote from document B.

    Returns:
        The footnote with longer innerhtml.
    """
    return f_b if len(f_b.innerhtml) > len(f_a.innerhtml) else f_a
