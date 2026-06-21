"""Tests for element renderer functions (rendering/elements.py)."""

from ragdoc.document import Document, Footnote, Image, Paragraph
from ragdoc.rendering import OutputFormat, Renderer
from ragdoc.rendering.base import RenderContext
from ragdoc.rendering.elements import render_for_prompt, render_raw

# --- TestElementRenderers ---


def test_element_render_for_prompt_returns_full_content():
    """render_for_prompt returns full content, not summary."""
    para = Paragraph(html="<p>Test content</p>")
    doc = Document(elements=[para])
    ctx = RenderContext(document=doc)

    result = render_for_prompt(para, ctx)

    assert result == "<p>Test content</p>"


def test_element_render_for_prompt_image_text_representation():
    """render_for_prompt uses text_representation for image content."""
    img = Image(
        image="base64data",
        image_type="png",
        alt="Chart",
        text_representation="A bar chart showing Q3 revenue",
    )
    doc = Document(elements=[img])
    ctx = RenderContext(document=doc)

    result = render_for_prompt(img, ctx)

    assert "A bar chart showing Q3 revenue" in result


def test_element_render_for_prompt_image_text_rep_fallback():
    """render_for_prompt falls back to text_representation."""
    img = Image(
        image="base64data",
        image_type="png",
        text_representation="<table>Revenue data</table>",
    )
    doc = Document(elements=[img])
    ctx = RenderContext(document=doc)

    result = render_for_prompt(img, ctx)

    assert "<table>Revenue data</table>" in result


def test_element_render_raw_includes_full_image():
    """render_raw includes base64 image data."""
    img = Image(image="base64data", image_type="png")
    doc = Document(elements=[img])
    ctx = RenderContext(document=doc)

    result = render_raw(img, ctx)

    assert "data:image/png;base64,base64data" in result


# --- TestInlineRendering ---


def test_inline_footnote_is_compact():
    """Footnotes render compactly when inline=True."""
    footnote = Footnote(number=1, innerhtml="This is the footnote text")
    doc = Document(elements=[footnote])
    ctx = RenderContext(document=doc)

    standalone = render_for_prompt(footnote, ctx, inline=False)
    assert '<aside class="footnote"' in standalone
    assert 'data-number="1"' in standalone

    inline = render_for_prompt(footnote, ctx, inline=True)
    assert "<span" in inline
    assert "[Footnote:" in inline


def test_inline_footnote_in_prompt():
    """Footnotes use compact format in prompts when inline."""
    footnote = Footnote(number=42, innerhtml="Important note")
    doc = Document(elements=[footnote])
    ctx = RenderContext(document=doc)

    inline = render_for_prompt(footnote, ctx, inline=True)

    assert "[Footnote:" in inline
    assert "Important note" in inline
    assert "<span" in inline


def test_inline_image_is_compact():
    """Images render with compact text when inline."""
    img = Image(
        image="base64data",
        image_type="png",
        text_representation="Chart showing revenue",
    )
    doc = Document(elements=[img])
    ctx = RenderContext(document=doc)

    standalone = render_for_prompt(img, ctx, inline=False)
    assert "<p>" in standalone

    inline = render_for_prompt(img, ctx, inline=True)
    assert "<span>" in inline
    assert "Chart showing revenue" in inline


def test_inline_renderer_uses_inline_for_refs():
    """Renderer uses inline=True when substituting refs."""
    footnote = Footnote(number=1, innerhtml="Important detail")
    para = Paragraph(
        html=f"<p>See this<ref id='{footnote.id}' rel='footnote'/> for more.</p>",
    )
    doc = Document(elements=[para, footnote])

    renderer = Renderer(format=OutputFormat.HTML)
    result = renderer.render(doc)

    assert "[Footnote:" in result
    assert "Important detail" in result
    assert result.count("Important detail") == 1


def test_inline_unreferenced_footnote_rendered_standalone():
    """Unreferenced footnotes render as standalone blocks."""
    footnote = Footnote(number=1, innerhtml="Orphaned footnote")
    para = Paragraph(html="<p>No references here.</p>")
    doc = Document(elements=[para, footnote])

    renderer = Renderer(format=OutputFormat.HTML)
    result = renderer.render(doc)

    assert 'id="footnote-' in result
    assert "Orphaned footnote" in result
