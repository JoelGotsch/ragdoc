"""Tests for footnote inline rendering correctness.

Covers three issues with inline footnote rendering:
1. Footnote number [N] must not appear as a bare token at the inline site.
2. <sup> tags must not wrap the inline footnote render.
3. <span class="footnote-inline"> must not survive markdown/GFM conversion
   — only the text content should appear.

Fix for issue 3: use <span> without a class attribute (or plain text). Pandoc
treats an attribute-free <span> as transparent and emits only the inner text.
A classed span like <span class="footnote-inline"> is kept as [{.footnote-inline}].

Fixture: tests/rendering/data/inline_footnote_fixture.json
  A parsed Document with footnote inline-refs. Tests gated on its existence
  skip when the fixture is absent (regenerate via the heading_llm/rendering
  fixture pipeline).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ragdoc.document import Document, Footnote, Paragraph
from ragdoc.rendering import OutputFormat, Renderer
from ragdoc.rendering.elements import render_for_prompt, render_raw

FIXTURE_PATH = Path(__file__).parent / "data" / "inline_footnote_fixture.json"
_FIXTURE_FOOTNOTE_TEXT = "Service Agreement, section 4."

_requires_fixture = pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="rendering fixture not present")


# ---------------------------------------------------------------------------
# Fixtures — documents
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_document() -> Document:
    return Document.model_validate_json(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inline_fn() -> Footnote:
    return Footnote(number=2, innerhtml="Service Agreement, section 4.")


@pytest.fixture(scope="module")
def inline_doc(inline_fn: Footnote) -> Document:
    para = Paragraph(
        html=f"<p>As previously reported,<ref id='{inline_fn.id}' rel='footnote'/> on 2 December.</p>",
    )
    return Document(elements=[para, inline_fn])


# ---------------------------------------------------------------------------
# Fixtures — renderers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompt_html() -> Renderer:
    return Renderer(format=OutputFormat.HTML, element_renderer=render_for_prompt)


@pytest.fixture(scope="module")
def raw_html() -> Renderer:
    return Renderer(format=OutputFormat.HTML, element_renderer=render_raw)


@pytest.fixture(scope="module")
def prompt_md() -> Renderer:
    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


@pytest.fixture(scope="module")
def prompt_gfm() -> Renderer:
    return Renderer(format=OutputFormat.GFM, element_renderer=render_for_prompt)


# --- TestNoFootnoteNumberInline ---


def test_no_footnote_number_prompt_html_no_bare_number(
    prompt_html: Renderer, inline_doc: Document, inline_fn: Footnote
):
    result = prompt_html.render(inline_doc)

    # [2] is only meaningful in the standalone footnote block, not inline
    assert "[2]" not in result
    # inline ref tag must be resolved
    assert f'<ref id="{inline_fn.id}" rel="footnote"/>' not in result
    # footnote must not also render as a separate block (it's inline-only)
    assert f'<p id="footnote-{inline_fn.id}">' not in result
    assert "Footnote" in result


def test_no_footnote_number_raw_html_uses_anchor_link(raw_html: Renderer, inline_doc: Document, inline_fn: Footnote):
    """render_raw uses an <a href="#footnote-..."> anchor for inline refs."""
    result = raw_html.render(inline_doc)

    assert f'href="#footnote-{inline_fn.id}"' in result


@_requires_fixture
def test_no_footnote_number_fixture_prompt_no_bare_numbers(prompt_html: Renderer, fixture_document: Document):
    """No bare [N] footnote numbers appear inline in the real document."""
    result = prompt_html.render(fixture_document)

    # Keep only lines that are NOT standalone footnote blocks; those may contain [N].
    lines = [l for l in result.splitlines() if 'id="footnote-' not in l]
    bare_numbers = re.findall(r"\[\d+\]", "\n".join(lines))
    assert bare_numbers == [], f"Bare footnote numbers found inline: {bare_numbers}"


# --- TestNoSupTagsInline ---


@pytest.mark.parametrize("renderer_name", ["prompt_html", "raw_html"])
def test_no_sup_tags_html_no_sup(renderer_name: str, prompt_html: Renderer, raw_html: Renderer, inline_doc: Document):
    renderer = prompt_html if renderer_name == "prompt_html" else raw_html
    result = renderer.render(inline_doc)
    assert "<sup>" not in result


@_requires_fixture
def test_no_sup_tags_fixture_prompt_no_sup(prompt_html: Renderer, fixture_document: Document):
    result = prompt_html.render(fixture_document)
    assert "<sup>" not in result


# --- TestNoSpanInMarkdown ---


def test_no_span_prompt_markdown_no_span(prompt_md: Renderer, inline_doc: Document):
    result = prompt_md.render(inline_doc)
    assert "footnote-inline" not in result


def test_no_span_prompt_markdown_footnote_text_present(prompt_md: Renderer, inline_doc: Document):
    """Footnote text must still appear even though the span annotation is gone."""
    result = prompt_md.render(inline_doc)
    # Normalize whitespace — pandoc may line-wrap within inline content
    assert "Service Agreement, section 4." in re.sub(r"\s+", " ", result)


def test_no_span_prompt_gfm_no_span(prompt_gfm: Renderer, inline_doc: Document):
    result = prompt_gfm.render(inline_doc)
    assert "footnote-inline" not in result


@_requires_fixture
def test_no_span_fixture_prompt_markdown_no_span(prompt_md: Renderer, fixture_document: Document):
    result = prompt_md.render(fixture_document)
    assert "footnote-inline" not in result


@_requires_fixture
def test_no_span_fixture_prompt_markdown_footnote_content_present(prompt_md: Renderer, fixture_document: Document):
    """Footnote content must still appear in markdown after span annotation is stripped."""
    result = prompt_md.render(fixture_document)
    # Normalize whitespace — pandoc may line-wrap footnote text
    assert _FIXTURE_FOOTNOTE_TEXT in re.sub(r"\s+", " ", result)
