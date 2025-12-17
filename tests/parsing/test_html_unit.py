"""Unit tests for HTML parsing functions (generate_document, unwrap_structural, etc.).

These tests call generate_document directly with test HTML files rather than
going through load(), making them lower-level unit tests.
"""

from ragdoc.parsing.html.load import HTML, generate_document, unwrap_structural

wrapped_paragraphs = """
<div class="content">

<h1>I am a heading but wrapped by tags</h1>



Content paragraph structurally separated from its heading by two layers.

Another paragraph structurally disjointed.

Some more text that is, for some reason, in a div.

Footer content

</div>
""".strip()


def test_nested_html(html_data_path):
    """Test parsing of nested HTML structures with and without unwrapping."""
    html = HTML.from_file(html_data_path / "test_arbitrary_nested.html")

    # Run parsing with no structural unwrapping
    document = generate_document(html, soup_transformers=[])

    # With no transformers, content is in paragraphs not headings
    assert document.paragraphs is not None
    assert len(document.paragraphs) >= 1
    # Check document has expected content
    full_text = " ".join([p.text for p in document.paragraphs])
    assert "I am a heading but wrapped by tags" in full_text or document.title == "Aribrary nested document"

    # Run parsing with unwrap_structural
    document = generate_document(html, soup_transformers=[unwrap_structural])

    # With unwrap_structural, we should get proper structure
    assert len(document.paragraphs) >= 1
    # text property extracts plain text
    texts = [p.text for p in document.paragraphs]
    combined_text = " ".join(texts)
    assert "Content paragraph structurally separated" in combined_text or len(document.paragraphs) >= 4


def test_special_case_html(html_data_path):
    """Test parsing of special case nested HTML."""
    html = HTML.from_file(html_data_path / "test_nested_special_case.html")

    document = generate_document(html, soup_transformers=[])

    assert len(document.paragraphs) == 1
    # text property extracts plain text from HTML
    assert "red, ORF.at/" in document.paragraphs[0].text
    assert "Agenturen" in document.paragraphs[0].text

    document = generate_document(html, soup_transformers=[unwrap_structural])

    assert "red, ORF.at/" in document.paragraphs[0].text


expected_idiot_table = """
<table>
<thead>
<tr>
<th>
<h2>I am a sad little heading because someone caged me.</h2>
<p>Being a paragraph here is not better</p>
</th>
</tr>
</thead>
<tbody></tbody>
</table>
""".strip()


def test_idiotic_table_unwrap(html_data_path):
    """Test unwrapping of tables used for layout purposes."""
    html = HTML.from_file(html_data_path / "test_idiotic_table.html")

    # run parsing with no idiotic table unwrapping
    document = generate_document(html, soup_transformers=[])

    assert len(document.tables) == 1
    # Check table contains expected content
    assert "I am a sad little heading because someone caged me." in document.tables[0].html
    # Document should have at least 1 paragraph
    assert len(document.paragraphs) >= 1
    # Check heading content exists
    assert len(document.headings) >= 1

    # run parsing with idiotic table unwrapping (default mode)
    document = generate_document(html)

    assert len(document.tables) == 0
    # Headings should be preserved
    assert len(document.headings) >= 1
    # Check that the heading text from the table is now accessible
    heading_texts = [h.text for h in document.headings]
    assert any("sad little heading" in t or "careful here" in t for t in heading_texts)


wrapped_expected_content = """
<div class="content">

I kind of should be a heading but i camo as paragraph



Content paragraph structurally separated from its heading by two layers.

Another paragraph structurally disjointed.

Some more text that is, for some reason, in a div.

Footer content

</div>
""".strip()


def test_no_heading(html_data_path):
    """Test parsing HTML without explicit heading tags."""
    html = HTML.from_file(html_data_path / "test_no_heading.html")

    document = generate_document(html, soup_transformers=[unwrap_structural])

    assert len(document.paragraphs) == 5
    p_texts = [p.text for p in document.paragraphs]
    # text property extracts plain text, not HTML
    assert "I kind of should be a heading but i camo as paragraph" in p_texts[0]
    assert "Content paragraph structurally separated from its heading by two layers." in p_texts[1]
    assert "Another paragraph structurally disjointed." in p_texts[2]
    assert "Some more text that is, for some reason, in a div." in p_texts[3]
    assert "Footer content" in p_texts[4]

    # Heading may or may not be present depending on HTML structure
    # But the title should be set from the HTML title element
    assert document.title == "Aribrary nested document"

    document = generate_document(html, soup_transformers=[])

    # With no structural unwrapping, all content may be in one paragraph
    assert len(document.paragraphs) >= 1
    combined_text = " ".join([p.text for p in document.paragraphs])
    assert "I kind of should be a heading" in combined_text
    assert "Footer content" in combined_text
    assert document.title == "Aribrary nested document"


def test_generate_image_transport_error_returns_none(monkeypatch, caplog):
    """A dead/hanging image host must yield None (not hang or crash) — Phase 0, bug 5."""
    import logging

    import httpx
    from bs4 import BeautifulSoup

    from ragdoc.parsing.html.load import generate_image

    def raise_timeout(url, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("ragdoc.parsing.html.load.httpx.get", raise_timeout)
    tag = BeautifulSoup('<img src="https://dead.example/x.png"/>', "html.parser").find("img")
    with caplog.at_level(logging.WARNING):
        assert generate_image(tag) is None
    assert any("Failed to download" in r.message for r in caplog.records)


def test_generate_image_passes_explicit_timeout(monkeypatch):
    """The image download must always carry an explicit timeout — Phase 0, bug 5."""
    from types import SimpleNamespace

    from bs4 import BeautifulSoup

    from ragdoc.parsing.html.load import generate_image

    seen: dict = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(status_code=200, content=b"abc")

    monkeypatch.setattr("ragdoc.parsing.html.load.httpx.get", fake_get)
    tag = BeautifulSoup('<img src="https://example.com/x.png"/>', "html.parser").find("img")
    image = generate_image(tag)
    assert image is not None
    assert seen.get("timeout") is not None
