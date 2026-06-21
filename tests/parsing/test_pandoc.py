import re
from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.parsing import PandocFile, load_file
from ragdoc.splitting import split_by_headings


def replace_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@pytest.fixture
def pandoc_document(docx_file_path: Path) -> Document:
    return load_file(PandocFile(file_path=str(docx_file_path)))


def test_document_headings(pandoc_document: Document):
    assert len(pandoc_document.headings) == 6
    heading_names = [h.text for h in pandoc_document.headings]
    assert heading_names == [
        "New super amazing Document",
        "Second layer",
        "Second second layer",
        "Third second layer",
        "Numbered items",
        "We can also do images!",
    ]


def test_h1_content(pandoc_document: Document):
    assert pandoc_document.name == "New super amazing Document"
    assert pandoc_document.level == 1
    paragraphs = pandoc_document.paragraphs
    assert len(paragraphs) >= 3
    assert "I am just text under the first paragraph" in paragraphs[0].text
    assert "What it should however provide " in paragraphs[1].text
    assert "And checking how it looks if certain text" in paragraphs[2].text
    assert Path(pandoc_document.source_path).match("*/tests/data/test.DOCX")
    assert pandoc_document.metadata["filename"] == "test.DOCX"


expected_table_markdown = """
A table containing mostly crap. Values of Amount Columns are in
EUR.

| **I am a super merged cell that serves as table heading** | | | | |
| --- | --- | --- | --- | --- |
| Nothing of relevance goes on in here | Year | Amount | Nothing useful | |
| 2023 | 2010344,5 |
| 2022 | 2012412,6 | Just | Some |
| 2021 | 12031212,3 |  | Cells |
""".strip()


def test_table_content(pandoc_document: Document):
    assert len(pandoc_document.tables) == 1
    assert replace_whitespace(pandoc_document.tables[0].markdown) == replace_whitespace(expected_table_markdown)


expected_list_markdown = """
* I am a list

  + I even contain sublists
  + With more than one element
* But nothing really useful here
* Apart from testing

  + Even

    - Deeper

      * Nesting
""".strip()

expected_numbered_list_markdown = """
1. There can also be numbered lists
   
   1. With nesting
      
      1. Even rather deep nesting
2. Especially given there is no content
""".strip()


def test_list_content(pandoc_document: Document):
    assert len(pandoc_document.lists) == 2
    assert replace_whitespace(pandoc_document.lists[0].markdown) == replace_whitespace(expected_list_markdown)
    assert replace_whitespace(pandoc_document.lists[1].markdown) == replace_whitespace(expected_numbered_list_markdown)


def test_image_content(pandoc_document: Document):
    assert len(pandoc_document.images) == 1
    image = pandoc_document.images[0]
    assert image.alt == "A person holding his glasses to his ear Description automatically generated"
    assert image.width is None
    assert image.height is None


# ---------------------------------------------------------------------------
# Nested images (test_nested_images.docx)
# ---------------------------------------------------------------------------

expected_nested_image_table = """
<table>
<colgroup>
<col style="width: 44%"/>
<col style="width: 55%"/>
</colgroup>
<thead>
<tr>
<th>Column Header 1</th>
<th>Column Header 2</th>
</tr>
</thead>
<tbody>
<tr>
<td>Nothing really</td>
<td><ref id="{key}" rel="image"></ref></td>
</tr>
</tbody>
</table>
""".strip()

expected_nested_image_list = """
<ul>
<li><p>And here goes a list</p>
<ul>
<li><p>That for some reason</p></li>
<li><p>Has an image in it\u2019s bullets</p>
<ul>
<li><p><ref id="{key}" rel="image"></ref></p></li>
</ul></li>
</ul></li>
</ul>
""".strip()


@pytest.fixture
def nested_image_document(nested_image_file_path) -> Document:
    return split_by_headings(load_file(PandocFile(file_path=str(nested_image_file_path))))[0]


def test_nested_images(nested_image_document):
    assert len(nested_image_document.images) == 3

    assert nested_image_document.images[0].alt == "A cartoon of a seal Description automatically generated"

    assert (
        nested_image_document.images[1].alt
        == "Cartoon a cartoon of a room with a table and a coffee cup Description automatically generated"
    )
    table_image_key = nested_image_document.images[1].id
    assert nested_image_document.tables[0].image_ids == [table_image_key]
    assert nested_image_document.tables[0].html == expected_nested_image_table.format(key=table_image_key)

    assert nested_image_document.images[2].alt == "A black and white logo Description automatically generated"
    list_image_key = nested_image_document.images[2].id
    assert nested_image_document.lists[0].image_ids == [list_image_key]
    assert nested_image_document.lists[0].html == expected_nested_image_list.format(key=list_image_key)
