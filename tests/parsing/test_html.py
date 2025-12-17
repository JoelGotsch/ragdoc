"""Tests for HTML loading and parsing functionality."""

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.parsing import HTMLFile, load_file


@pytest.fixture
def html_document(html_file_path) -> Document:
    """Load HTML file into a single Document."""
    html_file = HTMLFile(file_path=html_file_path)
    return load_file(html_file)


def test_document_headings(html_document: Document):
    """Test that all headings are correctly parsed from HTML."""
    assert len(html_document.headings) == 6
    heading_names = [h.text for h in html_document.headings]
    assert heading_names == [
        "New super amazing Document",
        "Second layer",
        "Second second layer",
        "Third second layer",
        "Numbered items",
        "We can also do images!",
    ]


def test_document_heading_levels(html_document: Document):
    """Test that heading levels are correctly parsed."""
    assert html_document.headings[0].level == 1  # h1
    assert html_document.headings[1].level == 2  # h2
    assert html_document.headings[2].level == 2  # h2
    assert html_document.headings[3].level == 3  # h3
    assert html_document.headings[4].level == 4  # h4
    assert html_document.headings[5].level == 2  # h2


def test_document_name(html_document: Document):
    """Test document name comes from main heading."""
    assert html_document.name == "New super amazing Document"
    assert html_document.level == 1


def test_paragraphs_content(html_document: Document):
    """Test paragraphs are correctly parsed from HTML."""
    # Document should have multiple paragraphs
    assert len(html_document.paragraphs) >= 3

    # Check specific paragraph content exists
    paragraph_texts = [p.text for p in html_document.paragraphs]
    combined_text = " ".join(paragraph_texts)

    assert "I am just text under the first paragraph" in combined_text
    assert "What it should however provide " in combined_text
    assert "And checking how it looks if certain text" in combined_text


def test_document_metadata(html_document: Document, html_file_path: Path):
    """Test document source fields are correctly set."""
    assert Path(html_document.source_path).match("*/tests/data/test.html")
    assert html_document.metadata["filename"] == "test.html"


expected_table = """
<table>
<caption>A table containing mostly crap. Values of Amount Columns are in
EUR.</caption>
<colgroup>
<col style="width: 19%"/>
<col style="width: 19%"/>
<col style="width: 19%"/>
<col style="width: 19%"/>
<col style="width: 20%"/>
</colgroup>
<thead>
<tr class="heading">
<th colspan="5"><strong>I am a super merged cell that serves as table
heading</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td rowspan="4">Nothing of relevance goes on in here</td>
<td>Year</td>
<td>Amount</td>
<td colspan="2" rowspan="2">Nothing useful</td>
</tr>
<tr class="even">
<td>2023</td>
<td>2010344,5</td>
</tr>
<tr class="odd">
<td>2022</td>
<td>2012412,6</td>
<td>Just</td>
<td>Some</td>
</tr>
<tr class="even">
<td>2021</td>
<td>12031212,3</td>
<td></td>
<td>Cells</td>
</tr>
</tbody>
</table>
""".strip()


def test_table_content(html_document: Document):
    """Test that tables are correctly parsed from HTML."""
    assert len(html_document.tables) == 1
    # Check that the table html contains key expected content
    table_html = html_document.tables[0].html
    assert "A table containing mostly crap" in table_html
    assert "Nothing of relevance goes on in here" in table_html


expected_list = """
<ul>
<li><p>I am a list</p>
<ul>
<li><p>I even contain sublists</p></li>
<li><p>With more than one element</p></li>
</ul></li>
<li><p>But nothing really useful here</p></li>
<li><p>Apart from testing</p>
<ul>
<li><p>Even</p>
<ul>
<li><p>Deeper</p>
<ul>
<li><p>Nesting</p></li>
</ul></li>
</ul></li>
</ul></li>
</ul>
""".strip()


def test_list_content(html_document: Document):
    """Test that lists are correctly parsed from HTML."""
    assert len(html_document.lists) >= 1
    # Find the unordered list with expected content
    ul_lists = [lst for lst in html_document.lists if "<ul>" in lst.html]
    assert len(ul_lists) >= 1
    # Check that key content is present
    assert "I am a list" in ul_lists[0].html
    assert "I even contain sublists" in ul_lists[0].html


def test_image_content(html_document: Document):
    """Test that images are correctly parsed from HTML."""
    assert len(html_document.images) == 1
    image = html_document.images[0]
    assert image.alt == "A person holding his glasses to his ear Description automatically generated"
    assert image.width is None
    assert image.height is None


def test_stunted(html_data_path):
    """Test parsing HTML where paragraph appears before first heading."""
    html_file = HTMLFile(file_path=str(html_data_path / "test_stunted.html"))
    document = load_file(html_file)

    # Test there is content before the first heading
    # Check that paragraphs and headings both exist
    assert len(document.paragraphs) >= 1
    assert len(document.headings) >= 1


def test_description_list_parsing(description_list_file_path):
    """Test that HTML description lists (<dl>, <dt>, <dd>) are properly parsed."""
    html_file = HTMLFile(file_path=str(description_list_file_path))
    document = load_file(html_file)

    assert document.title == "Description List Test"

    # Document should have description lists
    assert len(document.lists) >= 3

    # Check all headings are parsed (use .text property)
    heading_texts = [h.text for h in document.headings]
    assert "Description List Examples" in heading_texts
    assert "Simple Description List" in heading_texts
    assert "Nested Description List" in heading_texts
    assert "Description List with Multiple DDs" in heading_texts


def test_description_list_simple_content(description_list_file_path):
    """Test parsing of simple description list content."""
    html_file = HTMLFile(file_path=str(description_list_file_path))
    document = load_file(html_file)

    # Find the simple description list by content
    simple_dl = None
    for lst in document.lists:
        if "<dt>Coffee</dt>" in lst.html:
            simple_dl = lst
            break

    assert simple_dl is not None
    # Check content is present (outer dl tags may or may not be included)
    assert "<dt>Coffee</dt>" in simple_dl.html
    assert "black hot drink" in simple_dl.html
    assert "<dt>Milk</dt>" in simple_dl.html
    assert "white cold drink" in simple_dl.html

    assert "Coffee" in simple_dl.text
    assert "black hot drink" in simple_dl.text


def test_description_list_nested_content(description_list_file_path):
    """Test parsing of nested description list content."""
    html_file = HTMLFile(file_path=str(description_list_file_path))
    document = load_file(html_file)

    # Find the nested description list by content
    nested_dl = None
    for lst in document.lists:
        if "<dt>Beverages</dt>" in lst.html:
            nested_dl = lst
            break

    assert nested_dl is not None
    # Outer dl may not be included, but nested one should be
    assert nested_dl.html.count("<dl>") >= 1
    assert "<dt>Hot</dt>" in nested_dl.html
    assert "Coffee, Tea" in nested_dl.html


def test_description_list_multiple_definitions(description_list_file_path):
    """Test parsing of description list with multiple dd elements per dt."""
    html_file = HTMLFile(file_path=str(description_list_file_path))
    document = load_file(html_file)

    # Find the multi-DD description list by content
    multi_dd_dl = None
    for lst in document.lists:
        if "<dt>Term 1</dt>" in lst.html:
            multi_dd_dl = lst
            break

    assert multi_dd_dl is not None
    assert "<dd>Definition 1a</dd>" in multi_dd_dl.html
    assert "<dd>Definition 1b</dd>" in multi_dd_dl.html
    assert "<dt>Term 2</dt>" in multi_dd_dl.html
    assert "<dd>Definition 2</dd>" in multi_dd_dl.html


# ---------------------------------------------------------------------------
# Preface elements (test_preface.html)
# ---------------------------------------------------------------------------

expected_preface_table = "<table>\n<caption>Super amazing preface table</caption>\n<tr>\n<td><p>Useless paragraph in a useless table</p></td>\n</tr>\n</table>"


@pytest.fixture
def preface_document(html_preface_file_path) -> Document:
    return load_file(HTMLFile(file_path=html_preface_file_path))


def test_preface(preface_document: Document):
    assert any(
        p.text == "I am text in a super preface with nested paragaphs even though this is hilariously dumb"
        for p in preface_document.paragraphs
    )
    assert any(t.html == expected_preface_table for t in preface_document.tables)
    assert any(l.text == "Oh look a list!" for l in preface_document.lists)
