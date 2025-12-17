"""Tests for split_document_naive function."""
import pytest
from pathlib import Path

from ragdoc.parsing import load_file, HTMLFile
from ragdoc.splitting import split_by_headings
from ragdoc.document import Document


@pytest.fixture
def html_documents(html_file_path) -> list[Document]:
    """Load HTML file and split into documents by heading."""
    html_file = HTMLFile(file_path=html_file_path)
    return split_by_headings(load_file(html_file))


def test_split_document_naive_count(html_documents):
    """Test that split_document_naive creates correct number of documents."""
    assert len(html_documents) == 6


def test_split_document_naive_titles(html_documents):
    """Test that split documents have correct names from headings."""
    document_titles = [s.name for s in html_documents]
    assert document_titles == [
        "New super amazing Document",
        "Second layer",
        "Second second layer",
        "Third second layer",
        "Numbered items",
        "We can also do images!"
    ]


def test_split_document_naive_headings(html_documents):
    """Test that each split document has its own heading."""
    all_headings = [h.text for s in html_documents for h in s.headings]
    assert all_headings == [
        "New super amazing Document",
        "Second layer",
        "Second second layer",
        "Third second layer",
        "Numbered items",
        "We can also do images!"
    ]


def test_split_preserves_document_structure(html_documents):
    """Test that split documents preserve content structure."""
    # First document (h1) should have paragraphs but no tables or lists
    h1_doc = html_documents[0]
    assert h1_doc.name == "New super amazing Document"
    assert h1_doc.level == 1
    assert len(h1_doc.paragraphs) == 3
    assert len(h1_doc.tables) == 0
    assert len(h1_doc.lists) == 0


def test_split_document_with_table(html_documents):
    """Test that table content is preserved in split document."""
    table_doc = html_documents[2]
    assert table_doc.name == "Second second layer"
    assert len(table_doc.tables) == 1
    assert len(table_doc.paragraphs) == 1


def test_split_document_with_list(html_documents):
    """Test that list content is preserved in split document."""
    list_doc = html_documents[3]
    assert list_doc.name == "Third second layer"
    assert len(list_doc.lists) == 1


def test_split_document_with_image(html_documents):
    """Test that image content is preserved in split document."""
    image_doc = html_documents[5]
    assert image_doc.name == "We can also do images!"
    assert len(image_doc.images) == 1
    assert len(image_doc.paragraphs) == 1


# Tests for description list splitting
@pytest.fixture
def description_list_documents(description_list_file_path):
    """Load description list HTML file and split into documents."""
    html_file = HTMLFile(file_path=str(description_list_file_path))
    return split_by_headings(load_file(html_file))


def test_split_description_list_sections(description_list_documents):
    """Test that description list document splits into correct sections."""
    # Should have 4 sections: title + 3 h2 sections
    assert len(description_list_documents) == 4
    
    section_names = [d.name for d in description_list_documents]
    assert section_names == [
        "Description List Examples",
        "Simple Description List",
        "Nested Description List",
        "Description List with Multiple DDs"
    ]


def test_split_preserves_document_title(description_list_file_path):
    """Test that split preserves the original document title."""
    html_file = HTMLFile(file_path=str(description_list_file_path))
    document = load_file(html_file)
    documents = split_by_headings(document)
    
    assert document.title == "Description List Test"
    assert all(d.title == document.title for d in documents)


def test_split_simple_description_list(description_list_documents):
    """Test split document containing simple description list."""
    simple_dl_doc = description_list_documents[1]
    assert simple_dl_doc.name == "Simple Description List"
    assert len(simple_dl_doc.lists) == 1
    
    dl_html = simple_dl_doc.lists[0].html
    # Outer dl tags may not be included in html_content
    assert "<dt>Coffee</dt>" in dl_html
    assert "black hot drink" in dl_html
    assert "<dt>Milk</dt>" in dl_html
    assert "white cold drink" in dl_html
    
    dl_text = simple_dl_doc.lists[0].text
    assert "Coffee" in dl_text
    assert "black hot drink" in dl_text
    assert "Milk" in dl_text
    assert "white cold drink" in dl_text


def test_split_nested_description_list(description_list_documents):
    """Test split document containing nested description list."""
    nested_dl_doc = description_list_documents[2]
    assert nested_dl_doc.name == "Nested Description List"
    assert len(nested_dl_doc.lists) == 1
    
    dl_html = nested_dl_doc.lists[0].html
    # Outer dl may not be included, but nested one should be
    assert dl_html.count("<dl>") >= 1
    assert "<dt>Beverages</dt>" in dl_html
    assert "<dt>Hot</dt>" in dl_html
    assert "Coffee, Tea" in dl_html


def test_split_description_list_multiple_definitions(description_list_documents):
    """Test split document with description list having multiple dd per dt."""
    multi_dd_doc = description_list_documents[3]
    assert multi_dd_doc.name == "Description List with Multiple DDs"
    assert len(multi_dd_doc.lists) == 1
    
    dl_html = multi_dd_doc.lists[0].html
    assert "<dt>Term 1</dt>" in dl_html
    assert "<dd>Definition 1a</dd>" in dl_html
    assert "<dd>Definition 1b</dd>" in dl_html
    assert "<dt>Term 2</dt>" in dl_html
    assert "<dd>Definition 2</dd>" in dl_html
