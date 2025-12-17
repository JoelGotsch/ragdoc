"""Pytest fixtures scoped to the parsing test suite."""
import pytest

from pathlib import Path

from ragdoc.document import Document, Footnote, Paragraph


@pytest.fixture(scope="session")
def parsing_data_path() -> Path:
    """Root directory for parser-specific test data."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def html_data_path(parsing_data_path: Path) -> Path:
    """Directory containing HTML-parser-specific test files."""
    return parsing_data_path / "html"


@pytest.fixture(scope="session")
def html_preface_file_path(html_data_path: Path) -> Path:
    return html_data_path / "test_preface.html"


@pytest.fixture(scope="session")
def nested_image_file_path(parsing_data_path: Path) -> Path:
    return parsing_data_path / "pandoc" / "test_nested_images.docx"


# ---------------------------------------------------------------------------
# Anchor-footnote HTML and its expected parsed Document
# ---------------------------------------------------------------------------

#: HTML with anchor-linked footnotes: body links via <a href="#footnote-N">
#: and footnote paragraphs carry a matching id="footnote-N" attribute.
#: Footnote 1 is short; footnote 2 is deliberately long (multi-sentence) to
#: exercise rendering pipelines that must not truncate footnote content.
ANCHOR_FOOTNOTE_HTML = """\
<html>
<body>
<h1>Test Document</h1>
<p>
  This is the main body of my content.
  I have a footnote link for this line
  <a href="#footnote-1">[1]</a>.
  Then, I have some more content.
  Some of it is interesting and it has some
  footnotes as well <a href="#footnote-2">[2]</a>.
</p>
<p id="footnote-1">[1] Here is my first footnote.</p>
<p id="footnote-2">[2] This is a considerably longer footnote that spans multiple
sentences. It provides detailed background on the claim, including context from
prior research and caveats that apply in edge cases. The length of this footnote
is intentional: it verifies that the rendering pipeline does not truncate or
misrepresent multi-sentence footnote content when converting to any output format.</p>
</body>
</html>
"""


@pytest.fixture
def anchor_footnote_html() -> str:
    """Return the raw HTML string for the anchor-footnote test document."""
    return ANCHOR_FOOTNOTE_HTML


@pytest.fixture
def anchor_footnote_document() -> Document:
    """Return the Document that ``generate_document`` *should* produce for ANCHOR_FOOTNOTE_HTML.

    This fixture bypasses the HTML parser (which does not yet detect
    anchor-linked footnotes) to supply a correctly-structured Document for
    rendering tests.  It is also the authoritative description of the desired
    parser output:

    * Body paragraph: plain text with ``<ref id="…"/>`` placeholders where
      the ``<a href="#footnote-N">`` anchors appeared in the source HTML.
    * Two ``Footnote`` elements: fn1 (short) and fn2 (long/multi-sentence).
    * ``InlineRef`` entries on the body paragraph linking each placeholder to
      its ``Footnote`` element.
    """
    fn1 = Footnote(number=1, innerhtml="Here is my first footnote.")
    fn2 = Footnote(
        number=2,
        innerhtml=(
            "This is a considerably longer footnote that spans multiple "
            "sentences. It provides detailed background on the claim, including "
            "context from prior research and caveats that apply in edge cases. "
            "The length of this footnote is intentional: it verifies that the "
            "rendering pipeline does not truncate or misrepresent multi-sentence "
            "footnote content when converting to any output format."
        ),
    )

    body = Paragraph(
        html_content=(
            "<p>"
            "This is the main body of my content. "
            "I have a footnote link for this line "
            f'<ref id="{fn1.id}" rel="footnote"/>. '
            "Then, I have some more content. "
            "Some of it is interesting and it has some "
            f'footnotes as well <ref id="{fn2.id}" rel="footnote"/>.'
            "</p>"
        )
    )

    doc = Document()
    doc.elements = [body, fn1, fn2]
    return doc
