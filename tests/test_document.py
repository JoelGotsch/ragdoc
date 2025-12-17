"""Comprehensive tests for document.py element classes and Document model."""

import pytest

from ragdoc.document import (
    Document,
    DocumentList,
    ElementTypeEnum,
    Footnote,
    Heading,
    Image,
    Paragraph,
    RawText,
    Table,
    join_documents,
    merge_documents,
)

# ---------------------------------------------------------------------------
# Heading
# ---------------------------------------------------------------------------


def test_heading_basic_roundtrip():
    h = Heading(html="<h1>Title</h1>")
    assert h.text == "Title"
    assert h.level == 1
    assert h.html == "<h1>Title</h1>"


@pytest.mark.parametrize("level", range(1, 7))
def test_heading_all_heading_levels(level):
    html = f"<h{level}>Level {level}</h{level}>"
    h = Heading(html=html)
    assert h.level == level
    assert h.text == f"Level {level}"
    assert h.html == html


def test_heading_with_whitespace():
    # bs4 re-serialization (tag.decode) preserves inner whitespace; text strips it.
    h = Heading(html="  <h2>  spaced out  </h2>  ")
    assert h.text == "spaced out"
    assert h.html == "<h2>  spaced out  </h2>"


def test_heading_with_nested_inline_tags():
    h = Heading(html="<h3><strong>Bold</strong> heading</h3>")
    assert h.text == "Bold heading"
    assert h.html == "<h3><strong>Bold</strong> heading</h3>"
    assert h.innerhtml == "<strong>Bold</strong> heading"
    assert h.level == 3


def test_heading_with_extra_wrapper():
    """from_html should find the heading tag even inside a wrapper div."""
    h = Heading(html="<div><h2>Wrapped</h2></div>")
    assert h.text == "Wrapped"
    assert h.level == 2


def test_heading_html_setter_innerhtml():
    h = Heading(html="<h1>Original</h1>")
    h.html = "<h2>Updated <b>this is bold</b></h2>"
    assert h.text == "Updated this is bold"
    assert h.level == 2
    assert h.html == "<h2>Updated <b>this is bold</b></h2>"
    assert h.innerhtml == "Updated <b>this is bold</b>"
    assert h.markdown == "Updated **this is bold**"


def test_heading_construct_with_fields():
    h = Heading(html="<h4>Direct</h4>")
    assert h.html == "<h4>Direct</h4>"


def test_heading_with_page_and_bounding_box():
    h = Heading(html="<h1>Title</h1>", page=3, bounding_box=(0.1, 0.2, 0.8, 0.9))
    assert h.page == 3
    assert h.bounding_box == (0.1, 0.2, 0.8, 0.9)


def test_heading_markdown():
    h = Heading(html="<h2>My Title</h2>")
    assert "My Title" in h.markdown


def test_heading_level_zero_raises():
    """Setting level=0 should raise ValueError (h0 not allowed)."""
    h = Heading(html="<h1>12-March-2024</h1>")
    assert h.level == 1
    with pytest.raises(ValueError):
        h.level = 0


def test_heading_h0_normalizes_to_h1():
    """<h0> is not a heading tag (pattern is h[1-6]) — content is text-wrapped into <h1>."""
    h = Heading(html="<h0>Metadata</h0>")
    assert h.html == "<h1>Metadata</h1>"
    assert h.level == 1


# ---------------------------------------------------------------------------
# Paragraph
# ---------------------------------------------------------------------------


def test_paragraph_basic_roundtrip():
    html = "<p>Hello world</p>"
    p = Paragraph(html=html)
    assert p.html == html


def test_paragraph_with_inline_formatting():
    html = "<p>This is <strong>bold</strong> and <em>italic</em></p>"
    p = Paragraph(html=html)
    assert p.html == html


def test_paragraph_with_links():
    html = '<p>Visit <a href="https://example.com">here</a></p>'
    p = Paragraph(html=html)
    assert p.html == html


def test_paragraph_with_nested_spans():
    html = "<p><span class='a'><span class='b'>deep</span></span></p>"
    p = Paragraph(html=html)
    assert p.html == html


def test_paragraph_with_line_breaks():
    html = "<p>line one<br/>line two</p>"
    p = Paragraph(html=html)
    assert "line one" in p.html
    assert "line two" in p.html


def test_paragraph_with_footnote_reference():
    html = '<p>Some text<ref id="abc123" rel="footnote"/>[1]</p>'
    p = Paragraph(html=html)
    assert p.footnote_ids == ["abc123"]


def test_paragraph_with_multiple_footnote_references():
    html = '<p>Text<ref id="id1" rel="footnote"/>[1] more<ref id="id2" rel="footnote"/>[2]</p>'
    p = Paragraph(html=html)
    assert p.footnote_ids == ["id1", "id2"]


def test_paragraph_no_footnotes():
    p = Paragraph(html="<p>No refs</p>")
    assert p.footnote_ids == []


def test_paragraph_with_image_ref():
    html = '<p>See <ref id="img-42" rel="image"/> below</p>'
    p = Paragraph(html=html)
    assert p.image_ids == ["img-42"]


def test_paragraph_html_tag_property():
    p = Paragraph(html="<p>hello</p>")
    tag = p.html_tag
    assert tag.name == "p"
    assert tag.get_text() == "hello"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

TABLE_SIMPLE_TABLE = (
    "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
)


def test_table_basic_roundtrip():
    t = Table(html=TABLE_SIMPLE_TABLE)
    assert t.html == TABLE_SIMPLE_TABLE


def test_table_without_thead():
    html = "<table><tr><td>x</td><td>y</td></tr></table>"
    t = Table(html=html)
    assert t.html == html


def test_table_with_colspan():
    html = '<table><tr><td colspan="2">wide</td></tr></table>'
    t = Table(html=html)
    assert 'colspan="2"' in t.html


def test_table_with_rowspan():
    html = '<table><tr><td rowspan="3">tall</td><td>right</td></tr></table>'
    t = Table(html=html)
    assert 'rowspan="3"' in t.html


def test_table_with_nested_formatting():
    html = "<table><tr><td><strong>bold cell</strong></td></tr></table>"
    t = Table(html=html)
    assert "<strong>bold cell</strong>" in t.html


# ---------------------------------------------------------------------------
# DocumentList
# ---------------------------------------------------------------------------


def test_document_list_unordered_list_roundtrip():
    html = "<ul><li>one</li><li>two</li></ul>"
    dl = DocumentList(html=html)
    assert dl.html == html


def test_document_list_ordered_list_roundtrip():
    html = "<ol><li>first</li><li>second</li></ol>"
    dl = DocumentList(html=html)
    assert dl.html == html


def test_document_list_nested_list():
    html = "<ul><li>a<ul><li>a1</li><li>a2</li></ul></li><li>b</li></ul>"
    dl = DocumentList(html=html)
    assert dl.html == html


def test_document_list_with_inline_formatting():
    html = "<ul><li><em>italic item</em></li></ul>"
    dl = DocumentList(html=html)
    assert "<em>italic item</em>" in dl.html


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

IMAGE_SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUg=="


def test_image_construct_and_html():
    img = Image(image=IMAGE_SAMPLE_B64, image_type="png")
    assert f"data:image/png;base64,{IMAGE_SAMPLE_B64}" in img.html
    assert img.html.endswith("/>")


def test_image_src_property():
    img = Image(image=IMAGE_SAMPLE_B64, image_type="jpeg")
    assert img.src == f"data:image/jpeg;base64,{IMAGE_SAMPLE_B64}"


def test_image_placeholder_html():
    img = Image(image=IMAGE_SAMPLE_B64)
    assert img.placeholder_html == f'<ref id="{img.id}" rel="image"/>'


def test_image_html_includes_alt():
    img = Image(image=IMAGE_SAMPLE_B64, alt="a photo")
    assert 'alt="a photo"' in img.html


def test_image_html_includes_dimensions():
    img = Image(image=IMAGE_SAMPLE_B64, width=100, height=200)
    assert 'width="100"' in img.html
    assert 'height="200"' in img.html


def test_image_html_omits_none_fields():
    img = Image(image=IMAGE_SAMPLE_B64)
    assert "alt=" not in img.html
    assert "width=" not in img.html
    assert "height=" not in img.html


def test_image_from_html_data_uri():
    html = f'<img src="data:image/png;base64,{IMAGE_SAMPLE_B64}" alt="pic" width="50" height="60"/>'
    img = Image(html=html, image=IMAGE_SAMPLE_B64)
    assert img.alt == "pic"
    assert img.width == 50
    assert img.height == 60


def test_image_from_html_no_img_tag():
    img = Image(html="<div>not an image</div>", image=IMAGE_SAMPLE_B64)
    assert img.image == IMAGE_SAMPLE_B64


def test_image_ids_on_paragraph_with_ref():
    img = Image(image=IMAGE_SAMPLE_B64)
    p = Paragraph(html=f"<p>{img.placeholder_html}</p>")
    assert img.id in p.image_ids


def test_image_html_setter_with_src():
    new_b64 = "YWJjZGVm"
    img = Image(image=IMAGE_SAMPLE_B64, image_type="jpeg")
    original_id = img.id
    # Update html with data URI src and new attributes
    img.html = f'<img src="data:image/png;base64,{new_b64}" alt="new alt" width="50" height="75"/>'
    assert img.alt == "new alt"
    assert img.width == 50
    assert img.height == 75
    assert img.image_type == "png"
    # id should remain unchanged (not in data URI html)
    assert img.id == original_id


# ---------------------------------------------------------------------------
# RawText
# ---------------------------------------------------------------------------


def test_rawtext_basic_roundtrip():
    rt = RawText(html="<div>Hello raw</div>")
    assert rt.text == "Hello raw"
    assert rt.html == "<div>Hello raw</div>"


def test_rawtext_nested_roundtrip():
    rt = RawText(html="<div><span>Nested <p>paragraph</p></span> raw</div>")
    assert rt.text == "Nested paragraph raw"
    assert rt.html == "<div><span>Nested <p>paragraph</p></span> raw</div>"


def test_rawtext_strips_tags():
    rt = RawText(html="<p>nested <strong>tags</strong></p>")
    assert rt.text == "nested tags"
    assert rt.html == "<div>nested <strong>tags</strong></div>"


def test_rawtext_html_wraps_in_div():
    rt = RawText(html="plain")
    assert rt.html == "<div>plain</div>"


def test_rawtext_whitespace_stripping():
    rt = RawText(html="<div>  spaces  </div>")
    assert rt.text == "spaces"


def test_rawtext_empty_content():
    rt = RawText(html="<div></div>")
    assert rt.text == ""


def test_rawtext_multiline_content():
    rt = RawText(html="<div>line1\nline2</div>")
    assert "line1" in rt.text
    assert "line2" in rt.text


# ---------------------------------------------------------------------------
# Footnote
# ---------------------------------------------------------------------------


def test_footnote_basic_roundtrip():
    fn = Footnote(number=1, innerhtml="A note")
    html = fn.html
    fn2 = Footnote(html=html, id=fn.id)
    assert fn2.number == 1
    assert fn2.text == "A note"


def test_footnote_from_html_with_bracket_format():
    fn = Footnote(html='<p id="footnote-abc">[3] Some footnote text</p>')
    assert fn.number == 3
    assert fn.text == "Some footnote text"


def test_footnote_html_includes_id():
    fn = Footnote(number=5, innerhtml="Info", id="fn-5")
    assert 'id="footnote-fn-5"' in fn.html


def test_footnote_html_includes_number_and_text():
    fn = Footnote(number=2, innerhtml="Details")
    assert 'data-number="2"' in fn.html
    assert "Details" in fn.html


def test_footnote_referenced_in_paragraph():
    fn = Footnote(number=1, innerhtml="A note")
    p = Paragraph(html=f'<p>See<ref id="{fn.id}" rel="footnote"/>[1]</p>')
    assert fn.id in p.footnote_ids


# ---------------------------------------------------------------------------
# Cross-element parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "element,expected_type",
    [
        (Heading(html="<h1>Test</h1>"), ElementTypeEnum.HEADING),
        (Paragraph(html="<p>test</p>"), ElementTypeEnum.PARAGRAPH),
        (Table(html="<table><tr><td>a</td></tr></table>"), ElementTypeEnum.TABLE),
        (DocumentList(html="<ul><li>x</li></ul>"), ElementTypeEnum.DOCUMENT_LIST),
        (Image(image="iVBORw0KGgoAAAANSUhEUg=="), ElementTypeEnum.IMAGE),
        (RawText(html="<div>x</div>"), ElementTypeEnum.RAW_TEXT),
        (Footnote(number=1, innerhtml="x"), ElementTypeEnum.FOOTNOTE),
    ],
    ids=["heading", "paragraph", "table", "list", "image", "raw_text", "footnote"],
)
def test_element_type(element, expected_type):
    assert element.element_type == expected_type


@pytest.mark.parametrize(
    "element,new_html,check_field,expected",
    [
        (Heading(html="<h1>Original</h1>"), "<h2>Updated</h2>", "text", "Updated"),
        (Paragraph(html="<p>old</p>"), "<p>new</p>", "html", "<p>new</p>"),
        (
            Table(html="<table><tr><td>old</td></tr></table>"),
            "<table><tr><td>new</td></tr></table>",
            "html",
            "new",
        ),
        (DocumentList(html="<ul><li>old</li></ul>"), "<ol><li>new</li></ol>", "html", "<ol><li>new</li></ol>"),
        (RawText(html="<div>old</div>"), "<div>new</div>", "text", "new"),
        (Footnote(number=1, innerhtml="old"), "<p>[7] replaced</p>", "number", 7),
    ],
    ids=["heading", "paragraph", "table", "list", "raw_text", "footnote"],
)
def test_html_setter(element, new_html, check_field, expected):
    element.html = new_html
    actual = getattr(element, check_field)
    if isinstance(expected, str) and isinstance(actual, str):
        assert expected in actual
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "from_html_cls,raw_html,expected_html",
    [
        (Paragraph, "   <p>trimmed</p>   ", "<p>trimmed</p>"),
        (Table, "  <table><tr><td>x</td></tr></table>  ", "<table><tr><td>x</td></tr></table>"),
        (DocumentList, "   <ul><li>x</li></ul>   ", "<ul><li>x</li></ul>"),
    ],
    ids=["paragraph", "table", "list"],
)
def test_whitespace_strip(from_html_cls, raw_html, expected_html):
    element = from_html_cls(html=raw_html)
    assert element.html == expected_html


# ---------------------------------------------------------------------------
# BaseElement common behaviour
# ---------------------------------------------------------------------------


def test_base_element_id_auto_generated():
    h = Heading(html="<h1>T</h1>")
    assert h.id is not None
    assert len(h.id) > 0


def test_base_element_id_is_unique():
    ids = {Heading(html="<h1>T</h1>").id for _ in range(20)}
    assert len(ids) == 20


def test_base_element_html_tag_property():
    h = Heading(html="<h1>Title</h1>")
    tag = h.html_tag
    assert tag.name == "h1"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def _make_doc(*elements):
    return Document(elements=list(elements))


def test_document_empty_document():
    doc = Document()
    assert doc.empty
    assert doc.headings == []
    assert doc.paragraphs == []
    assert doc.tables == []
    assert doc.lists == []
    assert doc.images == []
    assert doc.raw_texts == []
    assert doc.footnotes == []


@pytest.mark.parametrize(
    "element,attr",
    [
        (Heading(html="<h1>H</h1>"), "headings"),
        (Table(html="<table><tr><td>x</td></tr></table>"), "tables"),
        (DocumentList(html="<ul><li>a</li></ul>"), "lists"),
        (Image(image="abc"), "images"),
        (RawText(html="raw"), "raw_texts"),
        (Footnote(number=1, innerhtml="fn"), "footnotes"),
    ],
    ids=["heading", "table", "list", "image", "raw_text", "footnote"],
)
def test_document_filter_by_type(element, attr):
    doc = _make_doc(element)
    assert getattr(doc, attr) == [element]


def test_document_unnamed_with_no_headings():
    doc = Document(elements=[Paragraph(html="<p>x</p>")])
    assert doc.unnamed is True


def test_document_unnamed_false_with_heading():
    doc = _make_doc(Heading(html="<h1>H</h1>"))
    assert doc.unnamed is False


def test_document_unnamed_false_with_title():
    doc = Document(title="T")
    assert doc.unnamed is False


def test_document_name_from_heading():
    doc = _make_doc(Heading(html="<h1>Main</h1>"))
    assert doc.name == "Main"


def test_document_name_none_when_no_heading():
    doc = Document()
    assert doc.name is None


def test_document_level_from_heading():
    doc = _make_doc(Heading(html="<h2>H</h2>"))
    assert doc.level == 2


def test_document_level_none_when_no_heading():
    doc = Document()
    assert doc.level is None


def test_document_main_heading_picks_lowest_level():
    h1 = Heading(html="<h2>H2</h2>", bounding_box=(0.0, 0.0, 1.0, 0.5))
    h2 = Heading(html="<h1>H1</h1>", bounding_box=(0.0, 0.5, 1.0, 1.0))
    doc = _make_doc(h1, h2)
    assert doc.main_heading.text == "H1"


def test_document_image_idx():
    img = Image(image="abc")
    doc = _make_doc(img)
    assert doc.image_idx[img.id] is img


def test_document_footnote_idx():
    fn = Footnote(number=1, innerhtml="fn")
    doc = _make_doc(fn)
    assert doc.footnote_idx[fn.id] is fn


def test_document_empty_property():
    assert Document().empty is True
    assert Document(elements=[Heading(html="<h1>H</h1>")]).empty is False


# ---------------------------------------------------------------------------
# merge_documents / join_documents (field-policy suite)
# ---------------------------------------------------------------------------


def test_merge_elements_concatenated_fresh_list():
    d1 = Document(elements=[Heading(html="<h1>A</h1>")])
    d2 = Document(elements=[Paragraph(html="<p>B</p>")])
    merged = merge_documents(d1, d2)
    assert len(merged.elements) == 2
    assert merged.elements is not d1.elements and merged.elements is not d2.elements
    assert len(d1.elements) == 1  # inputs untouched


def test_merge_titles_bridge_heading_escaped():
    d1 = Document(title="First", elements=[])
    d2 = Document(title="Second <script>", elements=[])
    merged = merge_documents(d1, d2)
    assert merged.title == "First"
    bridge = [e for e in merged.elements if isinstance(e, Heading)]
    assert len(bridge) == 1
    assert bridge[0].html == "<h1>Second &lt;script&gt;</h1>"
    assert bridge[0].text == "Second <script>"


def test_merge_one_title_no_bridge():
    merged = merge_documents(Document(elements=[]), Document(title="Only", elements=[]))
    assert merged.title == "Only"
    assert merged.elements == []


def test_merge_metadata_policy_first_matches_old_or():
    d1 = Document(metadata={"a": 1, "shared": "first"})
    d2 = Document(metadata={"b": 2, "shared": "second"})
    merged = merge_documents(d1, d2)  # default policy="first"
    assert merged.metadata == {"a": 1, "b": 2, "shared": "first"}


def test_merge_metadata_policy_second():
    d1 = Document(metadata={"a": 1, "shared": "first"})
    d2 = Document(metadata={"b": 2, "shared": "second"})
    merged = merge_documents(d1, d2, metadata_policy="second")
    assert merged.metadata == {"a": 1, "b": 2, "shared": "second"}


def test_merge_metadata_policy_strict_raises_on_conflict():
    d1 = Document(metadata={"shared": "first", "same": 1})
    d2 = Document(metadata={"shared": "second", "same": 1})
    with pytest.raises(ValueError, match="shared"):
        merge_documents(d1, d2, metadata_policy="strict")
    # equal values are not conflicts
    ok = merge_documents(Document(metadata={"same": 1}), Document(metadata={"same": 1}), metadata_policy="strict")
    assert ok.metadata == {"same": 1}


def test_merge_drops_provenance_and_external_refs():
    from ragdoc.document import ExternalRef

    d1 = Document(source_path="/a.html", external_refs=[ExternalRef(target_id="x", rel_type="external-parent")])
    d1.source_id = "sid"
    d1.source_hash = "shash"
    d2 = Document()
    merged = merge_documents(d1, d2)
    assert merged.source_path == "/a.html"  # first or second
    assert merged.source_id is None
    assert merged.source_hash is None
    assert merged.external_refs == []
    assert merged.id not in (d1.id, d2.id)  # new identity


def test_merge_parser_kept_iff_equal():
    same = merge_documents(Document(parser="html"), Document(parser="html"))
    assert same.parser == "html"
    differs = merge_documents(Document(parser="html"), Document(parser="xlsx"))
    assert differs.parser is None


def test_merge_result_metadata_is_fresh_dict():
    d1 = Document(metadata={"a": 1})
    d2 = Document(metadata={"b": 2})
    merged = merge_documents(d1, d2)
    merged.metadata["c"] = 3
    assert "c" not in d1.metadata and "c" not in d2.metadata


def test_join_documents_empty_and_reduce_order():
    assert join_documents([]).empty
    single = join_documents([Document(elements=[Heading(html="<h1>H</h1>")])])
    assert len(single.elements) == 1
    d1 = Document(elements=[Heading(html="<h1>A</h1>")], metadata={"shared": "first"})
    d2 = Document(elements=[Paragraph(html="<p>B</p>")], metadata={"shared": "second"})
    d3 = Document(elements=[Paragraph(html="<p>C</p>")], metadata={"shared": "third"})
    joined = join_documents([d1, d2, d3], metadata_policy="first")
    assert [e.text for e in joined.elements] == ["A", "B", "C"]
    assert joined.metadata == {"shared": "first"}  # left-to-right reduce, first wins


# ---------------------------------------------------------------------------
# Serialization roundtrip (model_dump / model_validate)
# ---------------------------------------------------------------------------


def test_serialization_heading_serialize_roundtrip():
    h = Heading(html="<h2>T</h2>", page=1)
    data = h.model_dump()
    h2 = Heading.model_validate(data)
    assert h2.text == h.text
    assert h2.level == h.level


def test_serialization_paragraph_serialize_roundtrip():
    p = Paragraph(html="<p>content</p>")
    data = p.model_dump()
    p2 = Paragraph.model_validate(data)
    assert p2.html == p.html


def test_serialization_document_serialize_roundtrip():
    doc = Document(
        title="Test",
        elements=[
            Heading(html="<h1>H</h1>"),
            Paragraph(html="<p>text</p>"),
            Table(html="<table><tr><td>x</td></tr></table>"),
            DocumentList(html="<ul><li>a</li></ul>"),
            RawText(html="raw"),
            Footnote(number=1, innerhtml="fn"),
        ],
    )
    data = doc.model_dump()
    doc2 = Document.model_validate(data)
    assert len(doc2.elements) == len(doc.elements)


# ---------------------------------------------------------------------------
# Document.orphaned_footnotes
# ---------------------------------------------------------------------------


def test_orphaned_footnotes_empty_document():
    doc = Document(elements=[])
    assert doc.orphaned_footnotes == []


def test_orphaned_footnotes_no_footnotes():
    doc = Document(elements=[Paragraph(html="<p>No references.</p>")])
    assert doc.orphaned_footnotes == []


def test_orphaned_footnotes_all_orphaned():
    doc = Document(
        elements=[
            Paragraph(html="<p>No ref tags here.</p>"),
            Footnote(id="fn-1", number=1, innerhtml="Orphan one."),
            Footnote(id="fn-2", number=2, innerhtml="Orphan two."),
        ]
    )
    orphans = doc.orphaned_footnotes
    assert len(orphans) == 2
    assert {f.id for f in orphans} == {"fn-1", "fn-2"}


def test_orphaned_footnotes_none_orphaned_all_referenced():
    doc = Document(
        elements=[
            Paragraph(html='<p>See <ref id="fn-1" rel="footnote"/> and <ref id="fn-2" rel="footnote"/>.</p>'),
            Footnote(id="fn-1", number=1, innerhtml="First footnote."),
            Footnote(id="fn-2", number=2, innerhtml="Second footnote."),
        ]
    )
    assert doc.orphaned_footnotes == []


def test_orphaned_footnotes_mixed_referenced_and_orphaned():
    doc = Document(
        elements=[
            Paragraph(html='<p>See <ref id="fn-1" rel="footnote"/> for details.</p>'),
            Footnote(id="fn-1", number=1, innerhtml="Referenced footnote."),
            Footnote(id="fn-2", number=2, innerhtml="Orphaned footnote."),
        ]
    )
    orphans = doc.orphaned_footnotes
    assert len(orphans) == 1
    assert orphans[0].id == "fn-2"


def test_orphaned_footnotes_ref_in_different_element_counts_as_referenced():
    """A footnote referenced in any element is not orphaned."""
    doc = Document(
        elements=[
            Paragraph(html="<p>First paragraph.</p>"),
            Paragraph(html='<p>Second paragraph <ref id="fn-1" rel="footnote"/>.</p>'),
            Footnote(id="fn-1", number=1, innerhtml="Somewhere else."),
        ]
    )
    assert doc.orphaned_footnotes == []
