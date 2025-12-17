"""Tests for the Renderer class (rendering/__init__.py)."""

from pydantic import BaseModel

from ragdoc.document import Document, Heading, Image, Paragraph
from ragdoc.rendering import OutputFormat, Renderer
from ragdoc.rendering.elements import (
    render_for_prompt,
    render_raw,
)

# --- TestRenderer ---


def _doc_with_title() -> Document:
    return Document(
        title="My Document Title",
        elements=[Paragraph(html="<p>Body content.</p>")],
    )


def _doc_with_styled_heading() -> Document:
    return Document(
        elements=[
            Heading(html='<h2 style="font-size: 14pt; text-align: center;">Centered Heading</h2>'),
            Paragraph(html="<p>Body text.</p>"),
        ]
    )


def _metadata_doc() -> Document:
    return Document(elements=[Paragraph(html="<p>Body.</p>")])


def test_renderer_render_simple_document():
    """Renderer can render a simple document."""
    doc = Document(
        elements=[
            Heading(html="<h1>Title</h1>"),
            Paragraph(html="<p>Content here.</p>"),
        ]
    )

    renderer = Renderer(format=OutputFormat.HTML)
    result = renderer.render(doc)

    assert "<h1>Title</h1>" in result
    assert "<p>Content here.</p>" in result


def test_renderer_render_with_metadata():
    """Renderer includes metadata when keys specified."""
    doc = Document(
        elements=[Paragraph(html="<p>Content</p>")],
        metadata={"source": "test.pdf", "author": "John"},
    )

    renderer = Renderer(
        format=OutputFormat.HTML,
        metadata_keys=["source"],
    )
    result = renderer.render(doc)

    assert "source" in result
    assert "test.pdf" in result
    assert "author" not in result


def test_renderer_render_skips_inline_only_elements():
    """Renderer skips elements that are only referenced inline."""
    img = Image(image="base64data", image_type="png", alt="Inline image")
    para = Paragraph(
        html=f"<p>See <ref id='{img.id}' rel='image'/></p>",
    )
    doc = Document(elements=[para, img])

    renderer = Renderer(format=OutputFormat.HTML)
    result = renderer.render(doc)

    assert result.count("Inline image") == 1
    assert "base64data" not in result


def test_renderer_render_resolves_inline_refs():
    """Renderer resolves <ref id='...'/> tags."""
    img = Image(image="base64data", image_type="png", alt="Test image")
    para = Paragraph(
        html=f"<p>See <ref id='{img.id}' rel='image'/> for details.</p>",
    )
    doc = Document(elements=[para, img])

    renderer = Renderer(format=OutputFormat.HTML)
    result = renderer.render(doc)

    assert "[Image: Test image]" in result
    assert f"<ref id='{img.id}'/>" not in result


def test_renderer_render_handles_missing_refs():
    """Renderer handles missing refs gracefully."""
    para = Paragraph(
        html="<p>See <ref id='nonexistent'/> for details.</p>",
    )
    doc = Document(elements=[para])

    renderer = Renderer(format=OutputFormat.HTML)
    result = renderer.render(doc)

    assert "[missing ref: nonexistent]" in result


def test_renderer_render_to_markdown():
    """Renderer can convert to markdown."""
    doc = Document(
        elements=[
            Heading(html="<h1>Title</h1>"),
            Paragraph(html="<p>Content</p>"),
        ]
    )

    renderer = Renderer(format=OutputFormat.MARKDOWN)
    result = renderer.render(doc)

    assert "# Title" in result or "Title\n====" in result


def test_renderer_render_to_plain_text():
    """Renderer can convert to plain text."""
    doc = Document(
        elements=[
            Heading(html="<h1>Title</h1>"),
            Paragraph(html="<p>Content here.</p>"),
        ]
    )

    renderer = Renderer(format=OutputFormat.PLAIN)
    result = renderer.render(doc)

    assert "Title" in result
    assert "Content here." in result
    assert "<" not in result


# --- TestTitleRendering ---


def test_title_prompt_html():
    """include_title prepends <h1> title using render_for_prompt, HTML output."""
    doc = _doc_with_title()
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        include_title=True,
    )
    result = renderer.render(doc)
    assert "<h1>My Document Title</h1>" in result
    assert "<p>Body content.</p>" in result


def test_title_prompt_markdown():
    """include_title produces h1 heading in markdown output for render_for_prompt."""
    doc = _doc_with_title()
    renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        include_title=True,
    )
    result = renderer.render(doc)
    assert "My Document Title" in result
    assert "# My Document Title" in result or "My Document Title\n" in result


def test_title_raw_html():
    """include_title prepends <h1> title using render_raw, HTML output."""
    doc = _doc_with_title()
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_raw,
        include_title=True,
    )
    result = renderer.render(doc)
    assert "<h1>My Document Title</h1>" in result


def test_title_raw_markdown():
    """include_title produces h1 heading in markdown for render_raw."""
    doc = _doc_with_title()
    renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_raw,
        include_title=True,
    )
    result = renderer.render(doc)
    assert "My Document Title" in result
    assert "# My Document Title" in result or "My Document Title\n" in result


def test_title_none_does_not_render():
    """include_title=True with no title set produces no spurious heading."""
    doc = Document(elements=[Paragraph(html="<p>Content.</p>")])
    renderer = Renderer(format=OutputFormat.HTML, include_title=True)
    result = renderer.render(doc)
    assert "<h1>" not in result


def test_title_precedes_body_content():
    """Title appears before body elements in rendered output."""
    doc = _doc_with_title()
    renderer = Renderer(format=OutputFormat.HTML, include_title=True)
    result = renderer.render(doc)
    title_pos = result.index("My Document Title")
    body_pos = result.index("Body content.")
    assert title_pos < body_pos


# --- TestHeadingCssStripping ---


def test_heading_css_no_style_attr_in_markdown():
    """render_for_prompt to MARKDOWN strips style= from headings."""
    doc = _doc_with_styled_heading()
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    result = renderer.render(doc)
    assert "Centered Heading" in result
    assert "style=" not in result


def test_heading_css_no_css_properties_in_markdown():
    """render_for_prompt to MARKDOWN strips CSS property values from headings."""
    doc = _doc_with_styled_heading()
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    result = renderer.render(doc)
    assert "text-align" not in result
    assert "font-size" not in result


def test_heading_css_level_preserved():
    """Heading level (h2 -> ##) is preserved after style stripping."""
    doc = _doc_with_styled_heading()
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    result = renderer.render(doc)
    assert "## Centered Heading" in result or "Centered Heading\n-" in result


# =============================================================================
# DocumentMetadata value type tests
# =============================================================================


class _SampleModel(BaseModel):
    name: str
    count: int


# --- TestDocumentMetadataRendering ---


def test_metadata_rendering_int_value_in_html():
    doc = _metadata_doc()
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        metadata_keys=["year"],
    )
    doc.metadata["year"] = 2024
    result = renderer.render(doc)
    assert "2024" in result


def test_metadata_rendering_none_value_skipped():
    doc = _metadata_doc()
    doc.metadata["missing"] = None
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        metadata_keys=["missing"],
    )
    result = renderer.render(doc)
    assert "missing" not in result


def test_metadata_rendering_dict_value_json_in_html():
    doc = _metadata_doc()
    doc.metadata["tags"] = {"primary": "finance", "secondary": "report"}
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        metadata_keys=["tags"],
    )
    result = renderer.render(doc)
    assert "finance" in result


def test_metadata_rendering_list_value_json_in_html():
    doc = _metadata_doc()
    doc.metadata["authors"] = ["Alice", "Bob"]
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        metadata_keys=["authors"],
    )
    result = renderer.render(doc)
    assert "Alice" in result
    assert "Bob" in result


def test_metadata_rendering_pydantic_model_value_in_html():
    doc = _metadata_doc()
    doc.metadata["source"] = _SampleModel(name="report.pdf", count=42)
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        metadata_keys=["source"],
    )
    result = renderer.render(doc)
    assert "report.pdf" in result
    assert "42" in result


def test_metadata_rendering_html_special_chars_escaped():
    """Values containing HTML special characters are escaped in output."""
    doc = _metadata_doc()
    doc.metadata["note"] = "<script>alert('xss')</script>"
    renderer = Renderer(
        format=OutputFormat.HTML,
        element_renderer=render_for_prompt,
        metadata_keys=["note"],
    )
    result = renderer.render(doc)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_metadata_rendering_markdown_yaml_frontmatter_with_int():
    """Non-string scalar values appear in YAML frontmatter."""
    doc = Document(
        title="My Doc",
        elements=[Paragraph(html="<p>Body.</p>")],
        metadata={"year": 2024},
    )
    renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        metadata_keys=["year"],
    )
    result = renderer.render(doc)
    assert "---" in result
    assert "My Doc" in result
    assert "2024" in result


# --- Escape site #3: metadata header -> pandoc <head> (title + meta tags) ---


def test_metadata_head_escapes_title_and_meta():
    """Title/metadata containing < and " must be escaped in the synthesized <head> (escape site #3)."""
    doc = Document(
        title='Q<1> "Report"',
        elements=[Paragraph(html="<p>body</p>")],
        metadata={'ke"y': 'va<l>ue "quoted"'},
    )
    renderer = Renderer(
        format=OutputFormat.MARKDOWN,
        element_renderer=render_for_prompt,
        metadata_keys=['ke"y'],
    )
    result = renderer.render(doc)
    # pandoc --standalone emits YAML frontmatter from the escaped <head>. Without the
    # html.escape fix the raw `<`/`"` would be parsed as markup and the content dropped
    # or the meta tag broken. pandoc backslash-escapes specials in YAML; strip those.
    unescaped = result.replace("\\", "")
    assert "Q<1>" in unescaped  # title content survives intact
    assert "va<l>ue" in unescaped  # meta value content survives intact
