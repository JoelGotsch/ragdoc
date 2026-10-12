"""RawText content must survive rendering in every output format.

Covers the two ways literal ``<``/``>``/``&`` reach a RawText: escaped code blocks (as built
by the MinerU code handler) and markup the caller deliberately stored as HTML.
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from ragdoc.document import Document, Paragraph, RawText
from ragdoc.parsing.mineru.handlers import build_code_html
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt, render_raw

CODE_SNIPPETS = [
    "<config>\n <a>1</a>\n</config>",
    "if a < b && c > d:\n    return x[i]  # <ref id='nope' rel='image'/>",
    'print("<div>not a wrapper</div>")',
]

STRATEGIES = [render_for_prompt, render_raw]


def _code_lines(code: str) -> list[str]:
    return [line.strip() for line in code.splitlines() if line.strip()]


@pytest.mark.parametrize("code", CODE_SNIPPETS)
def test_code_html_round_trips_through_the_element(code):
    raw = RawText(html=build_code_html(code))
    assert raw.html == build_code_html(code)  # validator leaves the wrapper and content alone
    assert BeautifulSoup(raw.html, "html.parser").get_text() == code
    assert raw.text == code.strip("\n -")
    assert raw.inline_refs == []  # an escaped "<ref .../>" in code is text, not a reference


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.__name__)
@pytest.mark.parametrize("code", CODE_SNIPPETS)
def test_code_block_renders_as_html_with_exact_text(code, strategy):
    doc = Document(elements=[Paragraph(html="<p>before</p>"), RawText(html=build_code_html(code))])
    rendered = Renderer(OutputFormat.HTML, strategy).render(doc)
    pre = BeautifulSoup(rendered, "html.parser").find("pre")
    assert pre is not None
    assert pre.get_text() == code


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.__name__)
@pytest.mark.parametrize("fmt", [OutputFormat.MARKDOWN, OutputFormat.GFM, OutputFormat.RST, OutputFormat.PLAIN])
@pytest.mark.parametrize("code", CODE_SNIPPETS)
def test_code_block_renders_literally_in_text_formats(code, fmt, strategy):
    doc = Document(elements=[Paragraph(html="<p>before</p>"), RawText(html=build_code_html(code))])
    rendered = Renderer(fmt, strategy).render(doc)
    rendered_lines = [line.strip() for line in rendered.splitlines()]
    for line in _code_lines(code):
        assert line in rendered_lines, f"{line!r} missing or altered in {fmt.value} output:\n{rendered}"
    assert "&lt;" not in rendered
    assert "&amp;" not in rendered


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.__name__)
def test_stored_markup_keeps_its_tags_in_html(strategy):
    """Markup stored in a RawText is HTML: its tags reach HTML output intact.

    (The renderer re-serializes through bs4, which collapses whitespace-only text between
    tags outside ``<pre>`` — insignificant in HTML — hence the whitespace-insensitive compare.)
    """
    raw = RawText(html="<config>\n <a>1</a>\n</config>")
    rendered = Renderer(OutputFormat.HTML, strategy).render(Document(elements=[raw]))
    assert "".join(rendered.split()) == "<div><config><a>1</a></config></div>"
