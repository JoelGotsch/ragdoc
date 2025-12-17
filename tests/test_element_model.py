"""Behavioral tests for the Phase-6 element model: stored ``html`` + normalizing validators.

Heading/Paragraph/Table/DocumentList/RawText store ``html`` as a plain field with a
normalizing ``field_validator``.  Image and Footnote keep *derived* ``html`` (property +
setter) because their html is a projection of structured fields.  All derived accessors
share one identity-keyed BeautifulSoup cache per element.
"""

from __future__ import annotations

import logging

import pytest
from bs4 import BeautifulSoup

import ragdoc.document as document_module
from ragdoc.document import (
    DocumentList,
    Footnote,
    Heading,
    Image,
    Paragraph,
    RawText,
    Table,
)

# ---------------------------------------------------------------------------
# Stored html field
# ---------------------------------------------------------------------------


def test_stored_html_is_a_field():
    dump = Paragraph(html="<p>x</p>").model_dump()
    assert dump["html"] == "<p>x</p>"
    assert "html_content" not in dump
    assert "innerhtml" not in dump


@pytest.mark.parametrize(
    ("cls", "html"),
    [
        (Heading, "<h2>t</h2>"),
        (Paragraph, "<p>t</p>"),
        (Table, "<table><tr><td>t</td></tr></table>"),
        (DocumentList, "<ul><li>t</li></ul>"),
        (RawText, "<div>t</div>"),
    ],
)
def test_stored_html_roundtrips_through_serialization(cls, html):
    element = cls(html=html)
    restored = cls.model_validate(element.model_dump())
    assert restored.html == element.html == html


def test_image_and_footnote_dumps_have_no_html_key():
    assert "html" not in Image(image="abc").model_dump()
    assert "html" not in Footnote(number=1, innerhtml="note").model_dump()


# ---------------------------------------------------------------------------
# Heading normalization
# ---------------------------------------------------------------------------


def test_heading_normalizes_on_construction():
    h = Heading(html='junk <h2 style="font-size: 12pt">t</h2> trailing')
    assert h.html == '<h2 style="font-size: 12pt">t</h2>'
    assert h.level == 2


def test_heading_no_tag_wraps_h1_and_escapes():
    h = Heading(html="a < b")
    assert h.html == "<h1>a &lt; b</h1>"
    assert h.level == 1
    assert h.text == "a < b"


def test_heading_attrs_are_escaped_roundtrip():
    # The old manual attr rebuild interpolated attribute values unescaped (escape site #1).
    h = Heading(html="<h2>t</h2>")
    soup = BeautifulSoup(h.html, "html.parser")
    tag = soup.find("h2")
    assert tag is not None
    tag["data-note"] = 'say "hi" & <bye>'
    h.html = str(tag)
    reparsed = BeautifulSoup(h.html, "html.parser").find("h2")
    assert reparsed is not None
    assert reparsed["data-note"] == 'say "hi" & <bye>'


def test_heading_level_setter_rebuilds_html():
    h = Heading(html="<h2>t</h2>")
    h.level = 4
    assert h.html == "<h4>t</h4>"
    assert h.level == 4


def test_heading_level_zero_raises():
    h = Heading(html="<h2>t</h2>")
    with pytest.raises(ValueError, match="level 0"):
        h.level = 0


def test_heading_innerhtml_property():
    h = Heading(html="<h3><em>fancy</em> title</h3>")
    assert h.innerhtml == "<em>fancy</em> title"


# ---------------------------------------------------------------------------
# Assignment validates + normalizes (replaces the old fset blocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("element", "new_html", "expected"),
    [
        (Heading(html="<h1>a</h1>"), "  <h3>y</h3>  ", "<h3>y</h3>"),
        (Paragraph(html="<p>a</p>"), "  <p>y</p>  ", "<p>y</p>"),
        (Table(html="<table></table>"), "  <table><tr><td>y</td></tr></table>  ", "<table><tr><td>y</td></tr></table>"),
        (DocumentList(html="<ul></ul>"), "  <ul><li>y</li></ul>  ", "<ul><li>y</li></ul>"),
        (RawText(html="a"), "plain y", "<div>plain y</div>"),
    ],
)
def test_html_assignment_validates_and_normalizes(element, new_html, expected):
    element.html = new_html
    assert element.html == expected


def test_heading_assignment_updates_level():
    h = Heading(html="<h1>a</h1>")
    h.html = "<h3>y</h3>"
    assert h.level == 3


def test_validate_assignment_rejects_bad_types():
    element = Paragraph(html="<p>x</p>")
    with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError; any raise is the contract
        element.page = "not-a-page"  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Soup cache (identity-keyed)
# ---------------------------------------------------------------------------


@pytest.fixture
def parse_counter(monkeypatch: pytest.MonkeyPatch):
    """Count BeautifulSoup parses performed inside ragdoc.document."""
    counter = {"n": 0}
    real_bs = document_module.BeautifulSoup

    def counting_bs(*args: object, **kwargs: object):
        counter["n"] += 1
        return real_bs(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(document_module, "BeautifulSoup", counting_bs)
    return counter


def test_soup_cache_reused_within_instance(parse_counter):
    p = Paragraph(html="<p>See <ref id='i1' rel='image'/> now</p>")
    parse_counter["n"] = 0
    _ = p.text
    _ = p.text
    _ = p.inline_refs
    assert parse_counter["n"] == 1


def test_soup_cache_invalidated_on_html_assignment(parse_counter):
    h = Heading(html="<h2>old</h2>")
    assert h.text == "old"
    h.html = "<h4>new</h4>"
    assert h.text == "new"
    assert h.level == 4


def test_html_tag_returns_independent_tag():
    p = Paragraph(html="<p>original</p>")
    tag = p.html_tag
    tag.string = "mutated"
    assert p.html == "<p>original</p>"
    assert p.text == "original"


# ---------------------------------------------------------------------------
# RawText normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain text", "<div>plain text</div>"),
        ("<span>x</span>", "<div>x</div>"),
        ("<div>already</div>", "<div>already</div>"),
        ("", "<div></div>"),
    ],
)
def test_rawtext_normalizes_to_div(raw, expected):
    assert RawText(html=raw).html == expected


def test_rawtext_default_is_empty_div():
    assert RawText().html == "<div></div>"


def test_rawtext_innerhtml_property():
    assert RawText(html="hello <b>world</b> bye").innerhtml == "hello <b>world</b> bye"


# ---------------------------------------------------------------------------
# Image: derived html, before-validator, escape
# ---------------------------------------------------------------------------


def test_image_html_kwarg_parses_fields():
    img = Image(html='<img src="data:image/png;base64,AAAA" alt="chart" width="10" height="20"/>')
    assert img.image == "AAAA"
    assert img.image_type == "png"
    assert img.alt == "chart"
    assert img.width == 10
    assert img.height == 20


def test_image_explicit_kwargs_beat_html():
    img = Image(html='<img src="data:image/png;base64,AAAA" alt="from-html"/>', alt="explicit")
    assert img.alt == "explicit"
    assert img.image == "AAAA"


def test_image_alt_is_escaped_in_html():
    img = Image(image="AAAA", alt='she said "<hi>"')
    assert 'alt="she said &quot;&lt;hi&gt;&quot;"' in img.html
    # And the value survives a parse round-trip
    tag = BeautifulSoup(img.html, "html.parser").find("img")
    assert tag is not None
    assert tag["alt"] == 'she said "<hi>"'


def test_image_html_setter_updates_fields():
    img = Image()
    img.html = '<img src="data:image/jpeg;base64,BBBB" width="5"/>'
    assert img.image == "BBBB"
    assert img.image_type == "jpeg"
    assert img.width == 5


# ---------------------------------------------------------------------------
# Footnote: derived html, before-validator + setter
# ---------------------------------------------------------------------------


def test_footnote_html_kwarg_and_setter_roundtrip():
    fn = Footnote(html='<aside class="footnote" data-number="3">the note</aside>')
    assert fn.number == 3
    assert fn.innerhtml == "the note"
    assert f'id="footnote-{fn.id}"' in fn.html  # html still embeds the live id

    fn.html = '<aside class="footnote" data-number="7">changed</aside>'
    assert fn.number == 7
    assert fn.innerhtml == "changed"


# ---------------------------------------------------------------------------
# from_markdown typed signature
# ---------------------------------------------------------------------------


def test_from_markdown_typed_signature():
    p = Paragraph.from_markdown("**bold**", page=2)
    assert "<strong>bold</strong>" in p.html
    assert p.page == 2
    with pytest.raises(TypeError):
        Paragraph.from_markdown("x", bounding_box=(0, 0, 1, 1))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# inline_refs unknown-rel handling
# ---------------------------------------------------------------------------


def test_inline_refs_unknown_rel_logs_once(caplog: pytest.LogCaptureFixture):
    html = (
        "<p>a <ref id='x1' rel='bogus-rel-once'/> b "
        "<ref id='x2' rel='bogus-rel-once'/> c <ref id='x3' rel='image'/></p>"
    )
    p = Paragraph(html=html)
    with caplog.at_level(logging.WARNING, logger="ragdoc.document"):
        refs = p.inline_refs
    warnings = [r for r in caplog.records if "bogus-rel-once" in r.getMessage()]
    assert len(warnings) == 1
    assert [r.rel_type for r in refs] == ["image"]
