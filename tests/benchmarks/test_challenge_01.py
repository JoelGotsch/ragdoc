"""
Benchmark tests for challenge_01.docx — Slice D (known-evil adversarial DOCX).

Each challenge section has a comment block describing the DOCX pattern.
Tests that assert IDEAL behaviour the parser cannot yet deliver are marked
with @pytest.mark.xfail(strict=True, ...).  Remove the mark + update this
comment when the feature is implemented.

Ambiguous cases (where the correct parser behaviour is debatable) use
strict=False so an unexpected pass is noted but not treated as a build error.
"""

import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ragdoc.document import Document
from ragdoc.parsing.pandoc import load_pandoc

DOCX_PATH = Path(__file__).parent / "fixtures" / "slice_d" / "challenge_01.docx"

# ── Session fixture ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def doc() -> Document:
    return load_pandoc(DOCX_PATH)


def _table_text(doc: Document) -> str:
    """All table cells joined into one string for substring checks."""
    return "\n".join(BeautifulSoup(t.html, "html.parser").get_text() for t in doc.tables)


def _heading_names(doc: Document) -> set[str]:
    """Normalised heading texts (collapse internal whitespace)."""
    return {re.sub(r"\s+", " ", h.text).strip() for h in doc.headings}


def _para_texts(doc: Document) -> list[str]:
    return [p.text for p in doc.paragraphs]


# ════════════════════════════════════════════════════════════════════════════
# §1  Parser metadata
# ════════════════════════════════════════════════════════════════════════════


def test_parser_is_pandoc(doc: Document) -> None:
    assert doc.parser == "pandoc"


# ════════════════════════════════════════════════════════════════════════════
# §2  Headings
# Challenge: real H1/H2/H3 headings must be detected; two visually prominent
#            Normal-style paragraphs must NOT be promoted.
# ════════════════════════════════════════════════════════════════════════════


def test_heading_count(doc: Document) -> None:
    assert len(doc.headings) == 9


@pytest.mark.parametrize(
    "text, level",
    [
        ("The Footnote Gauntlet", 1),
        ("Headings and Their Impostors", 1),
        ("The Heading Hierarchy", 2),
        # H3 directly under H1 — no implicit H2 should be inserted
        ("Genuine Subsection", 3),
        ("The List Labyrinth", 1),
        ("Tables: Truth Versus Decoration", 1),
        ("Nested Structures", 1),
        ("The Caption Confusion", 1),
        ("The Chaos Zone", 1),
    ],
)
def test_real_heading_detected(doc: Document, text: str, level: int) -> None:
    names = _heading_names(doc)
    assert text in names, f"Expected heading {text!r} not found in {names}"
    matching = [h for h in doc.headings if re.sub(r"\s+", " ", h.text).strip() == text]
    assert matching[0].level == level


# Known limitation: pandoc wraps long heading text at ~80 chars, injecting \r\n
@pytest.mark.xfail(
    strict=True,
    reason="Pandoc wraps long heading text, injecting \\r\\n inside Heading.text",
)
def test_heading_text_has_no_internal_linebreak(doc: Document) -> None:
    bad = [h for h in doc.headings if "\r\n" in h.text or "\n" in h.text]
    assert not bad, f"Headings with internal newlines: {[h.text for h in bad]}"


# ── Fake headings must stay as Paragraphs ─────────────────────────────────────


def test_fake_heading_bold_not_promoted(doc: Document) -> None:
    """Bold 14pt Normal-style paragraph — must NOT become a Heading."""
    assert "The Subcommittee on Subheadings" not in _heading_names(doc)


def test_fake_heading_allcaps_not_promoted(doc: Document) -> None:
    """All-caps Normal-style paragraph — must NOT become a Heading."""
    assert "THIS ENTIRELY UPPERCASE PARAGRAPH COULD FOOL A NAIVE CLASSIFIER" not in _heading_names(doc)


def test_fake_heading_bold_is_paragraph(doc: Document) -> None:
    assert any("The Subcommittee on Subheadings" in t for t in _para_texts(doc))


def test_fake_heading_allcaps_is_paragraph(doc: Document) -> None:
    assert any("THIS ENTIRELY UPPERCASE PARAGRAPH COULD FOOL A NAIVE CLASSIFIER" in t for t in _para_texts(doc))


# ════════════════════════════════════════════════════════════════════════════
# §3  Footnotes
# Challenge: three real OOXML footnotes (IDs 1–3) + one fake footnote that
#            is plain body text with a Unicode superscript '4'.
#
# KNOWN FAILURE: pandoc converts OOXML footnotes to
#   <section class="footnotes"><ol><li id="fn1">...</li></ol></section>
# which the HTML parser does not handle.  Result: 0 Footnote elements.
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(
    strict=True,
    reason="HTML parser does not handle pandoc-style <section class='footnotes'> output",
)
def test_real_footnote_count(doc: Document) -> None:
    assert len(doc.footnotes) == 3


@pytest.mark.xfail(
    strict=True,
    reason="HTML parser does not handle pandoc-style footnotes",
)
@pytest.mark.parametrize(
    "snippet, insertion_point",
    [
        ("IVC/2024/Report-7", "published its annual report in March 2024"),
        ("Vienna, Geneva, and Tokyo", "findings from three independent laboratories"),
        ("Appendix C", "stability under pressure"),
    ],
)
def test_footnote_text_content(doc: Document, snippet: str, insertion_point: str) -> None:
    fn_texts = [fn.text for fn in doc.footnotes]
    assert any(snippet in t for t in fn_texts), f"Footnote containing {snippet!r} not found"


@pytest.mark.xfail(
    strict=True,
    reason="HTML parser does not handle pandoc-style footnotes — inline anchors stay as raw <a> tags",
)
def test_footnote_inline_refs_resolved(doc: Document) -> None:
    """Body paragraph should have <ref> placeholders, not raw <a href='#fn1'> anchors."""
    para_htmls = [p.html for p in doc.paragraphs]
    raw_anchor_in_body = any("footnote-ref" in h or "fnref" in h for h in para_htmls)
    assert not raw_anchor_in_body


# ── Fake footnote must NOT become a Footnote element ─────────────────────────


def test_fake_footnote_not_in_footnotes(doc: Document) -> None:
    """The '4th footnote' is plain body text — must never be parsed as Footnote."""
    fn_texts = [fn.text for fn in doc.footnotes]
    assert not any("seasonal correction factor" in t for t in fn_texts)


def test_fake_footnote_stays_as_paragraph(doc: Document) -> None:
    assert any("seasonal correction factor" in t for t in _para_texts(doc))


# ════════════════════════════════════════════════════════════════════════════
# §4  Lists
# Challenge A: real bullet list (3 nesting levels) and real numbered list
#              must be recognised as DocumentList elements.
# Challenge B: four manually typed "1.", "2.", "3.", "a." paragraphs must
#              NOT be parsed as list items.
#
# KNOWN FAILURE: python-docx creates separate numbering definitions for
# List Bullet / List Bullet 2 / List Bullet 3, and pandoc emits a separate
# <ul> per level.  Result: 9 DocumentList elements instead of 2.
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(
    strict=True,
    reason="Separate Word list styles produce separate <ul>/<ol> blocks in pandoc HTML; "
    "parser sees 9 DocumentList elements instead of 2 nested lists",
)
def test_list_count_nested(doc: Document) -> None:
    """One bullet list + one numbered list = 2 DocumentList elements."""
    assert len(doc.lists) == 2


def test_bullet_list_content_present(doc: Document) -> None:
    """All bullet levels appear somewhere in the parsed lists, even if unnested."""
    all_list_text = "\n".join(lst.text for lst in doc.lists)
    for item in (
        "Alpha-level item",
        "Beta-level item (nested once)",
        "Gamma-level item (nested twice)",
    ):
        assert item in all_list_text, f"List item {item!r} missing"


@pytest.mark.xfail(
    strict=True,
    reason="Separate <ul> blocks prevent nested structure; bullet list items are at flat level",
)
def test_bullet_list_is_nested(doc: Document) -> None:
    """The bullet list must be a single nested DocumentList with ilvl > 0 for sub-items."""
    # A nested list HTML would have <ul><li>...<ul><li>...</li></ul></li></ul>
    bullet_lists = [lst for lst in doc.lists if "<ul" in lst.html]
    assert len(bullet_lists) == 1
    html = bullet_lists[0].html
    assert html.count("<ul") > 1, "Expected nested <ul> inside the bullet list"


def test_numbered_list_content_present(doc: Document) -> None:
    all_list_text = "\n".join(lst.text for lst in doc.lists)
    assert "Genuine first numbered item" in all_list_text
    assert "Genuine second numbered item" in all_list_text


# ── Fake numbered list must stay as Paragraphs ───────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "1. The first pseudo-item stands alone as a plain paragraph.",
        "2. The second pseudo-item looks numbered but lacks list formatting.",
        "3. The third pseudo-item completes the impostor sequence.",
        "a. This sub-item is also fake",
    ],
)
def test_fake_list_item_is_paragraph(doc: Document, text: str) -> None:
    # Normalise whitespace: pandoc wraps long lines with \r\n inside paragraph text
    normalised = [re.sub(r"\s+", " ", t) for t in _para_texts(doc)]
    assert any(text in t for t in normalised), f"Expected para containing {text!r}"


def test_fake_list_items_not_in_lists(doc: Document) -> None:
    list_text = "\n".join(lst.text for lst in doc.lists)
    assert "pseudo-item" not in list_text
    assert "This sub-item is also fake" not in list_text


# ════════════════════════════════════════════════════════════════════════════
# §5  Tables
# Challenge: data table + 1×1 separator + 2-col layout table + nested table.
# All four tables are currently parsed identically; a smart parser would
# classify and normalise the separator and layout tables differently.
# ════════════════════════════════════════════════════════════════════════════


def test_table_count(doc: Document) -> None:
    """Four tables: data, separator, layout, and outer-with-nested."""
    assert len(doc.tables) == 4


@pytest.mark.parametrize(
    "expected",
    ["Annual Performance Summary", "YoY Growth", "North", "South", "Total", "+14 units"],
)
def test_data_table_content(doc: Document, expected: str) -> None:
    assert expected in _table_text(doc), f"Expected {expected!r} in table content"


def test_nested_table_content(doc: Document) -> None:
    txt = _table_text(doc)
    assert "Inner metric" in txt
    assert "1 240 req/s" in txt


# Ideal: separator table (1×1 empty, thick border) should be dropped
@pytest.mark.xfail(
    strict=True,
    reason="layout_table_to_separator normalisation not implemented; 1×1 separator table kept",
)
def test_separator_table_dropped(doc: Document) -> None:
    assert len(doc.tables) == 3  # separator removed


# Ideal: layout table (2-col, borderless) should be flattened to paragraphs
@pytest.mark.xfail(
    strict=True,
    reason="layout_table_to_columns normalisation not implemented; layout table kept as Table",
)
def test_layout_table_converted_to_paragraphs(doc: Document) -> None:
    # "Key Findings" and "Recommendations" should be Paragraphs, not in a Table
    para_text = "\n".join(_para_texts(doc))
    assert "Key Findings" in para_text
    assert "Recommendations" in para_text
    table_text = _table_text(doc)
    assert "Key Findings" not in table_text


# ════════════════════════════════════════════════════════════════════════════
# §6  Fake code block
# Challenge: a single paragraph in Courier New + tab indentation + w:br breaks
#            with Normal paragraph style — visually identical to a code block.
# ════════════════════════════════════════════════════════════════════════════


def test_code_block_content_present(doc: Document) -> None:
    """Code content must appear somewhere in the document, even as a Paragraph."""
    all_text = "\n".join(_para_texts(doc))
    assert "def analyse_report" in all_text
    assert "mean = sum(data)" in all_text


def test_code_block_is_one_element(doc: Document) -> None:
    """All four code lines must be inside a single Paragraph (joined by line breaks)."""
    code_paras = [p for p in doc.paragraphs if "def analyse_report" in p.text]
    assert len(code_paras) == 1
    text = code_paras[0].text
    assert "mean = sum(data)" in text
    assert "variance = sum" in text


@pytest.mark.xfail(
    strict=True,
    reason="indented_paragraph_to_code normalisation not implemented; fake code block stays as Paragraph, not RawText",
)
def test_code_block_is_raw_text(doc: Document) -> None:
    assert any("def analyse_report" in r.text for r in doc.raw_texts)


# ════════════════════════════════════════════════════════════════════════════
# §7  Image and unlinked caption
# Challenge: image followed by an italic paragraph "Figure 1: ..." that has
#            no OOXML semantic link to the image.
# ════════════════════════════════════════════════════════════════════════════


def test_image_count(doc: Document) -> None:
    assert len(doc.images) == 1


def test_caption_paragraph_present(doc: Document) -> None:
    """Caption is a plain Paragraph with no link to the image."""
    assert any("Figure 1" in t for t in _para_texts(doc))
    assert any("parsing and chunking pipeline" in t for t in _para_texts(doc))


@pytest.mark.xfail(
    strict=True,
    reason="orphan_caption_to_linked normalisation not implemented; caption paragraph stays unlinked",
)
def test_caption_linked_to_image(doc: Document) -> None:
    """Ideal: caption text should become Image.text_representation after normalisation."""
    assert any(img.text_representation and "Figure 1" in img.text_representation for img in doc.images)


# ════════════════════════════════════════════════════════════════════════════
# §8  Chaos paragraph (manual line breaks)
# Challenge: ONE Word paragraph containing four sentences joined by w:br.
#            Visually looks like four paragraphs but is structurally one.
# ════════════════════════════════════════════════════════════════════════════


def test_chaos_lines_in_one_paragraph(doc: Document) -> None:
    """All four chaos lines must appear in exactly one Paragraph."""
    chaos_paras = [p for p in doc.paragraphs if "Chaos line one" in p.text]
    assert len(chaos_paras) == 1, "Expected exactly one paragraph containing all chaos lines"
    combined = chaos_paras[0].text
    assert "Chaos line two" in combined
    assert "Chaos line three" in combined
    assert "Chaos line four" in combined


def test_chaos_is_single_paragraph_not_multiple(doc: Document) -> None:
    chaos_count = sum(1 for p in doc.paragraphs if "Chaos line" in p.text)
    assert chaos_count == 1


# ════════════════════════════════════════════════════════════════════════════
# §9  Unicode math
# Challenge: σ² = Σ(xᵢ − μ)² / n with Unicode codepoints — must be preserved.
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "char, name",
    [
        ("\u03c3", "sigma"),
        ("\u03a3", "Sigma"),
        ("\u03bc", "mu"),
    ],
)
def test_unicode_math_char_preserved(doc: Document, char: str, name: str) -> None:
    assert any(char in t for t in _para_texts(doc)), f"Unicode {name} ({char}) not found"


# ════════════════════════════════════════════════════════════════════════════
# §10  Footer-like body paragraph
# Challenge: grey 9pt centred paragraph "CONFIDENTIAL | … | Page 12 of 15"
#            is plain body text (no Word footer section) — ambiguous whether
#            a smart parser should drop it or keep it.
# ════════════════════════════════════════════════════════════════════════════


def test_footer_text_currently_in_paragraphs(doc: Document) -> None:
    """Basic parser includes the footer-like paragraph — document as-is."""
    assert any("TMP-2024-001" in t for t in _para_texts(doc))


@pytest.mark.xfail(
    strict=False,  # ambiguous: over-normalisation risk is real
    reason="Footer-pattern body text detection not implemented; paragraph is currently included, not dropped",
)
def test_footer_text_dropped_by_smart_parser(doc: Document) -> None:
    """Ideal (aspirational): parser detects and drops footer-patterned body text."""
    assert not any("TMP-2024-001" in t for t in _para_texts(doc))
