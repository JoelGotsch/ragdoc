"""Tests for render_raw idempotency: html_source -> parse -> render -> parse yields equivalent Document.

Starting from HTML that our own renderer produces, parsing it, re-rendering, and
re-parsing must yield a structurally equivalent document (same element types, same text
content, same inline-ref relationships). Element IDs are regenerated on every parse and
are never asserted equal -- except for Footnote UUIDs which are preserved via the
id="footnote-{uuid}" attribute.
"""

import pytest

from ragdoc.document import Document, DocumentList, Footnote, Heading, Image, Paragraph, Table
from ragdoc.merging.patch import validate_inline_refs
from ragdoc.parsing.html.load import HTML, generate_document
from ragdoc.rendering import OutputFormat, Renderer, render_raw

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjkB6QAAAABJRU5ErkJggg=="


@pytest.fixture(scope="module")
def reference_html():
    """Render a known document to produce the html_source for idempotency tests."""
    fn1 = Footnote(number=1, innerhtml="First footnote body.")
    fn2 = Footnote(number=2, innerhtml="Second footnote with <em>italic</em>.")
    fn_orphan = Footnote(number=3, innerhtml="Orphan footnote -- no paragraph references this.")
    img_inline = Image(image=_PNG_B64, image_type="png", alt="1px test image", width=1, height=1)
    img_standalone = Image(image=_PNG_B64, image_type="png", alt="Standalone image", width=1, height=1)

    doc = Document(
        elements=[
            Heading(html="<h1>Document Title</h1>"),
            Heading(html="<h2>Section with <em>italic heading</em></h2>"),
            Heading(html="<h3>Subsection</h3>"),
            Heading(html="<h4>Deep heading</h4>"),
            Paragraph(html="<p>Plain paragraph.</p>"),
            Paragraph(html="<p><strong>Bold</strong>, <em>italic</em>, <code>code</code>.</p>"),
            Paragraph(html="<p>Math: E\u2009=\u2009<math>mc<sup>2</sup></math></p>"),
            Paragraph(html='<p style="text-align: center;">Centred paragraph.</p>'),
            Paragraph(html="<p>Entities: a &amp; b, x &lt; y, z &gt; w.</p>"),
            Paragraph(html=f'<p>Text with footnote<ref id="{fn1.id}" rel="footnote"/>.</p>'),
            fn1,
            Paragraph(html=f'<p>See <ref id="{img_inline.id}" rel="image"/> for details.</p>'),
            img_inline,
            Paragraph(
                html=(
                    f'<p>Combined<ref id="{fn2.id}" rel="footnote"/> and <ref id="{img_inline.id}" rel="image"/>.</p>'
                )
            ),
            fn2,
            fn_orphan,
            img_standalone,
            Table(
                html=(
                    "<table><thead><tr><th>Col A</th><th>Col B</th></tr></thead>"
                    "<tbody><tr><td>R1C1</td><td>R1C2 with <em>markup</em></td></tr></tbody></table>"
                )
            ),
            DocumentList(html="<ul><li>Apple</li><li>Banana</li></ul>"),
            DocumentList(html="<ol><li>First</li><li>Second</li></ol>"),
        ]
    )
    renderer = Renderer(format=OutputFormat.HTML, element_renderer=render_raw)
    return renderer.render(doc)


@pytest.fixture(scope="module")
def doc1(reference_html):
    return generate_document(HTML(content=reference_html))


@pytest.fixture(scope="module")
def doc2(doc1):
    renderer = Renderer(format=OutputFormat.HTML, element_renderer=render_raw)
    return generate_document(HTML(content=renderer.render(doc1)))


@pytest.fixture(scope="module", params=["doc1", "doc2"])
def doc(request, doc1, doc2):
    """Yield each document variant for parametrized roundtrip tests."""
    return doc1 if request.param == "doc1" else doc2


# --- TestRoundtripHeadings ---


def test_roundtrip_heading_count(doc):
    assert len(doc.headings) == 4


def test_roundtrip_heading_levels(doc):
    levels = {h.level for h in doc.headings}
    assert levels == {1, 2, 3, 4}


def test_roundtrip_heading_inline_markup(doc):
    h2s = [h for h in doc.headings if h.level == 2]
    assert any("<em>" in h.innerhtml for h in h2s)


# --- TestRoundtripParagraphs ---


def test_roundtrip_paragraph_count(doc):
    assert len(doc.paragraphs) == 8


def test_roundtrip_rich_markup_preserved(doc):
    htmls = [p.html for p in doc.paragraphs]
    assert any("<strong>" in h and "<em>" in h and "<code>" in h for h in htmls)


def test_roundtrip_math_preserved(doc):
    htmls = [p.html for p in doc.paragraphs]
    assert any("<math>" in h and "<sup>" in h for h in htmls)


def test_roundtrip_inline_style_preserved(doc):
    htmls = [p.html for p in doc.paragraphs]
    assert any("text-align: center" in h for h in htmls)


def test_roundtrip_entities(doc):
    combined = " ".join(p.html for p in doc.paragraphs)
    # BS4 may normalise &amp; -> & in text but keep it encoded in HTML attr context;
    # accept either form as the round-trip is structure-preserving, not byte-exact.
    assert "a &amp; b" in combined or "a & b" in combined


# --- TestRoundtripFootnotes ---


def test_roundtrip_footnote_count(doc):
    assert len(doc.footnotes) == 3


def test_roundtrip_footnote_numbers(doc):
    numbers = {f.number for f in doc.footnotes}
    assert numbers == {1, 2, 3}


def test_roundtrip_footnote_innerhtml(doc):
    innerhtmls = {f.innerhtml for f in doc.footnotes}
    assert any("First footnote body." in h for h in innerhtmls)
    assert any("Second footnote with" in h for h in innerhtmls)


def test_roundtrip_footnote_inline_refs_present(doc):
    """Paragraphs that cite footnotes must have footnote_ids after round-trip."""
    paragraphs_with_refs = [p for p in doc.paragraphs if p.footnote_ids]
    assert len(paragraphs_with_refs) >= 2


def test_roundtrip_validate_inline_refs_clean(doc1, doc2):
    """All <ref> tags in both docs must resolve to existing elements."""
    assert validate_inline_refs(doc1) == []
    assert validate_inline_refs(doc2) == []


# --- TestRoundtripImages ---


def test_roundtrip_inline_image_data_preserved(doc):
    image_datas = {img.image for img in doc.images}
    assert _PNG_B64 in image_datas


def test_roundtrip_standalone_image_data_preserved(doc):
    assert any(img.alt == "Standalone image" for img in doc.images)


def test_roundtrip_inline_image_ref_present(doc):
    """At least one paragraph must carry a <ref rel='image'> after round-trip."""
    paragraphs_with_image_refs = [p for p in doc.paragraphs if p.image_ids]
    assert len(paragraphs_with_image_refs) >= 1


# --- TestRoundtripTable ---


def test_roundtrip_table_present(doc):
    assert len(doc.tables) == 1


def test_roundtrip_table_has_headers(doc):
    table = doc.tables[0]
    assert "<th>" in table.html


def test_roundtrip_cell_markup_preserved(doc):
    table = doc.tables[0]
    assert "<em>" in table.html


# --- TestRoundtripLists ---


def test_roundtrip_ul_present(doc):
    assert any("<ul>" in lst.html for lst in doc.lists)


def test_roundtrip_ol_present(doc):
    assert any("<ol>" in lst.html for lst in doc.lists)


def test_roundtrip_list_item_content(doc):
    all_list_html = " ".join(lst.html for lst in doc.lists)
    assert "Apple" in all_list_html
    assert "Banana" in all_list_html
    assert "First" in all_list_html
    assert "Second" in all_list_html
