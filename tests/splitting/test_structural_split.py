"""Oversized tables and lists split on row/item boundaries into well-formed fragments.

Every fragment keeps its wrapper element (``<table>``, ``<ul>``/``<ol>``/``<dl>``) and its
element type (Table / DocumentList), no fragment is ever sliced mid-tag, no content is lost
or duplicated, and every split respects the token budget.
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup, Tag

from ragdoc.document import (
    BaseElement,
    Document,
    DocumentList,
    ExternalRef,
    Footnote,
    Heading,
    Image,
    Paragraph,
    RawText,
    Table,
)
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
from ragdoc.splitting.groups import ElementGroup
from ragdoc.splitting.token import (
    build_fragment_html,
    plan_html_split,
    split_at_html_tags,
    split_document,
    split_oversized_element,
)
from ragdoc.utils import GPTTokenizer

FORMATS = [OutputFormat.HTML, OutputFormat.MARKDOWN, OutputFormat.GFM, OutputFormat.PLAIN]


def _renderer(fmt: OutputFormat) -> Renderer:
    return Renderer(fmt, render_for_prompt)


def _single_root(html: str) -> Tag:
    """Parse *html* and assert it is exactly one well-formed root element."""
    roots = [
        node for node in BeautifulSoup(html, "html.parser").contents if not (isinstance(node, str) and not node.strip())
    ]
    assert len(roots) == 1, f"expected one root element, got {len(roots)}: {html[:120]!r}"
    root = roots[0]
    assert isinstance(root, Tag)
    assert html.startswith(f"<{root.name}"), html[:60]
    assert html.endswith(f"</{root.name}>"), html[-60:]
    return root


def _content_elements(split: Document) -> list[BaseElement]:
    return [el for el in split.elements if not isinstance(el, Heading)]


def _rows(n: int) -> str:
    return "".join(f"<tr><td>row {i}</td><td>value number {i}</td></tr>" for i in range(n))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
def test_oversized_table_splits_into_well_formed_tables(fmt):
    """Reviewer repro: 120-row table, max_tokens=400."""
    tokenizer = GPTTokenizer()
    renderer = _renderer(fmt)
    doc = Document(elements=[Table(html=f"<table>{_rows(120)}</table>")])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    assert len(splits) > 1
    seen_rows: list[str] = []
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
        (element,) = _content_elements(split)
        assert isinstance(element, Table), f"table fragment degraded to {type(element).__name__}: {element.html[:80]!r}"
        table = _single_root(element.html)
        assert table.name == "table"
        rows = table.find_all("tr")
        assert rows, "fragment has no rows"
        for row in rows:
            cells = row.find_all("td")
            assert len(cells) == 2, f"row sliced mid-structure: {row}"
            seen_rows.append(cells[0].get_text())
    assert seen_rows == [f"row {i}" for i in range(120)]  # every row exactly once, in order


def test_table_fragments_repeat_header_and_keep_sections_and_attributes():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.HTML)
    html = (
        '<table class="data" style="width: 100%"><caption>Quarterly numbers</caption>'
        "<thead><tr><th>name</th><th>value</th></tr></thead>"
        f"<tbody>{_rows(80)}</tbody></table>"
    )
    table = Table(html=html, page=7, bounding_box=(0.1, 0.2, 0.3, 0.4), metadata={"origin": "t1"})

    splits = split_document(Document(elements=[table]), renderer, tokenizer, max_tokens=400)

    assert len(splits) > 1
    seen_rows: list[str] = []
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
        (element,) = _content_elements(split)
        assert isinstance(element, Table)
        assert element.page == 7
        assert element.bounding_box == (0.1, 0.2, 0.3, 0.4)
        assert element.metadata == {"origin": "t1"}
        root = _single_root(element.html)
        assert root.get("class") == ["data"]
        assert root.get("style") == "width: 100%"
        caption = root.find("caption")
        thead = root.find("thead")
        body = root.find("tbody")
        assert isinstance(caption, Tag) and caption.get_text() == "Quarterly numbers"
        assert isinstance(thead, Tag) and [th.get_text() for th in thead.find_all("th")] == ["name", "value"]
        assert isinstance(body, Tag)
        seen_rows.extend(tr.find("td").get_text() for tr in body.find_all("tr", recursive=False))
    assert seen_rows == [f"row {i}" for i in range(80)]


def test_table_fragment_metadata_is_not_shared_with_the_source_table():
    table = Table(html=f"<table>{_rows(60)}</table>", metadata={"k": "v"})
    splits = split_document(Document(elements=[table]), _renderer(OutputFormat.HTML), GPTTokenizer(), max_tokens=400)
    fragment = _content_elements(splits[0])[0]
    fragment.metadata["k"] = "changed"
    assert table.metadata == {"k": "v"}


def test_table_with_a_single_oversized_row_still_respects_the_budget():
    """One row larger than the budget cannot be kept structural — it falls back to the text tiers."""
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.MARKDOWN)
    long_cell = " ".join(f"Sentence number {i} ends here." for i in range(200))
    html = (
        "<table><tr><td>a</td><td>b</td></tr>"
        f"<tr><td>big</td><td>{long_cell}</td></tr>"
        "<tr><td>c</td><td>d</td></tr></table>"
    )

    splits = split_document(
        Document(elements=[Table(html=html)]), renderer, tokenizer, max_tokens=400, overlap_tokens=20
    )

    assert len(splits) > 2
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
    rendered_all = " ".join(" ".join(renderer.render(split).split()) for split in splits)
    for i in (0, 57, 199):
        assert f"Sentence number {i} ends here." in rendered_all


def test_table_row_refs_travel_with_their_row():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.HTML)
    note = Footnote(number=1, innerhtml="the note text")
    rows = _rows(60) + f"<tr><td>row with note</td><td>see<ref id='{note.id}' rel='footnote'/></td></tr>" + _rows(5)
    doc = Document(elements=[Table(html=f"<table>{rows}</table>"), note])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    holders = [s for s in splits if any(isinstance(el, Footnote) for el in s.elements)]
    assert len(holders) == 1
    assert "row with note" in _content_elements(holders[0])[0].html
    assert "the note text" in renderer.render(holders[0])


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("list_tag", ["ul", "ol"])
@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.value)
def test_oversized_list_splits_into_well_formed_lists(list_tag, fmt):
    tokenizer = GPTTokenizer()
    renderer = _renderer(fmt)
    items = "".join(f"<li>item {i} with <em>some</em> descriptive text attached</li>" for i in range(150))
    doc = Document(elements=[DocumentList(html=f'<{list_tag} class="steps">{items}</{list_tag}>')])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    assert len(splits) > 1
    seen: list[str] = []
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
        (element,) = _content_elements(split)
        assert isinstance(element, DocumentList), f"list fragment degraded to {type(element).__name__}"
        root = _single_root(element.html)
        assert root.name == list_tag
        assert root.get("class") == ["steps"]
        fragment_items = root.find_all("li", recursive=False)
        assert fragment_items
        if list_tag == "ol":
            assert int(str(root.get("start", "1"))) == len(seen) + 1  # numbering continues across fragments
        seen.extend(li.get_text().split(" with")[0] for li in fragment_items)
    assert seen == [f"item {i}" for i in range(150)]


def test_definition_list_keeps_terms_with_their_definitions():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.HTML)
    pairs = "".join(f"<dt>term {i}</dt><dd>definition of term {i} goes here</dd>" for i in range(120))
    doc = Document(elements=[DocumentList(html=f"<dl>{pairs}</dl>")])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    assert len(splits) > 1
    terms: list[str] = []
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
        (element,) = _content_elements(split)
        assert isinstance(element, DocumentList)
        root = _single_root(element.html)
        assert root.name == "dl"
        children = [c for c in root.children if isinstance(c, Tag)]
        assert children[0].name == "dt"  # never split between a dt and its dd
        assert children[-1].name == "dd"
        terms.extend(dt.get_text() for dt in root.find_all("dt", recursive=False))
    assert terms == [f"term {i}" for i in range(120)]


def test_image_refs_in_list_items_travel_with_their_item():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.MARKDOWN)
    image = Image(image="AAAA", image_type="png", alt="the chart")
    items = "".join(f"<li>item {i} with some descriptive text attached</li>" for i in range(100))
    items += f"<li>pictured <ref id='{image.id}' rel='image'/></li>"
    doc = Document(elements=[DocumentList(html=f"<ul>{items}</ul>"), image])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    holders = [s for s in splits if any(isinstance(el, Image) for el in s.elements)]
    assert len(holders) == 1
    assert "the chart" in renderer.render(holders[0])


# ---------------------------------------------------------------------------
# Heading context and generic containers
# ---------------------------------------------------------------------------


def test_table_fragments_carry_heading_context():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.MARKDOWN)
    heading = Heading(html="<h1>Results</h1>")
    doc = Document(elements=[heading, Table(html=f"<table>{_rows(120)}</table>")])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    assert len(splits) > 1
    for split in splits:
        assert split.elements[0] is heading
        assert isinstance(split.elements[1], Table)
        assert tokenizer.count(renderer.render(split)) <= 400


def test_generic_container_fragments_are_wrapped_raw_text():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.HTML)
    blocks = "".join(f"<p>block {i} of the container with some words</p>" for i in range(150))
    doc = Document(elements=[RawText(html=f'<div class="box">{blocks}</div>')])

    splits = split_document(doc, renderer, tokenizer, max_tokens=400)

    assert len(splits) > 1
    seen: list[str] = []
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
        (element,) = _content_elements(split)
        assert isinstance(element, RawText)
        root = _single_root(element.html)
        assert root.name == "div"
        assert root.get("class") == ["box"]
        seen.extend(p.get_text().split(" of")[0] for p in root.find_all("p", recursive=False))
    assert seen == [f"block {i}" for i in range(150)]


def test_structural_split_declines_when_bare_text_sits_between_children():
    """A paragraph of text with inline tags must NOT be split at its inline tags (text would be dropped)."""
    para = Paragraph(html="<p>" + " ".join(f"plain words {i} <em>emph {i}</em>." for i in range(80)) + "</p>")
    result = split_at_html_tags(
        group=ElementGroup(root=para),
        heading_ctx=[],
        renderer=_renderer(OutputFormat.MARKDOWN),
        tokenizer=GPTTokenizer(),
        max_tokens=400,
        doc_title=None,
        doc_metadata={},
        parent_ref=ExternalRef(target_id="parent", rel_type="external-parent"),
    )
    assert result is None


def test_oversized_paragraph_with_inline_tags_loses_no_text():
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.PLAIN)
    para = Paragraph(html="<p>" + " ".join(f"plain words {i} <em>emph {i}</em>." for i in range(120)) + "</p>")

    splits = split_oversized_element(Document(elements=[para]), renderer, tokenizer, max_tokens=400, overlap_tokens=20)

    assert len(splits) > 1
    joined = " ".join(" ".join(renderer.render(s).split()) for s in splits)
    for i in range(120):
        assert f"plain words {i} " in joined
        assert f"emph {i}" in joined


# ---------------------------------------------------------------------------
# plan_html_split / build_fragment_html
# ---------------------------------------------------------------------------


def test_plan_html_split_table_units_and_header():
    plan = plan_html_split(
        "<table border='1'><thead><tr><th>h</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody>"
        "<tfoot><tr><td>sum</td></tr></tfoot></table>"
    )
    assert plan is not None
    assert plan.outer_name == "table"
    assert plan.header_html == "<thead><tr><th>h</th></tr></thead>"
    assert [unit.html for unit in plan.units] == [
        "<tr><td>1</td></tr>",
        "<tr><td>2</td></tr>",
        "<tr><td>sum</td></tr>",
    ]
    assert build_fragment_html(plan, 0, 2) == (
        '<table border="1"><thead><tr><th>h</th></tr></thead>'
        "<tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>"
    )
    assert build_fragment_html(plan, 1, 3) == (
        '<table border="1"><thead><tr><th>h</th></tr></thead><tbody><tr><td>2</td></tr></tbody>'
        "<tfoot><tr><td>sum</td></tr></tfoot></table>"
    )


def test_plan_html_split_ordered_list_continues_numbering():
    plan = plan_html_split('<ol start="5"><li>a</li><li>b</li><li>c</li></ol>')
    assert plan is not None
    assert build_fragment_html(plan, 0, 1) == '<ol start="5"><li>a</li></ol>'
    assert build_fragment_html(plan, 1, 3) == '<ol start="6"><li>b</li><li>c</li></ol>'


@pytest.mark.parametrize(
    "html",
    [
        "plain text, no tags",
        "<p>just text</p>",
        "<table><tr><td>only one row</td></tr></table>",
        "<ul><li>only one item</li></ul>",
        "<p>text <b>bold</b> more text <i>italic</i></p>",
    ],
)
def test_plan_html_split_returns_none_when_nothing_to_split(html):
    assert plan_html_split(html) is None


# ---------------------------------------------------------------------------
# Text tiers verify the *rendered* chunk against the budget
# ---------------------------------------------------------------------------


def test_token_slice_chunks_fit_the_budget_when_rerendering_inflates_them():
    """Markdown-special characters are escaped when a sliced chunk is rendered again.

    The slice window must shrink until the rendered chunk fits, not just the raw slice.
    """
    tokenizer = GPTTokenizer()
    renderer = _renderer(OutputFormat.MARKDOWN)
    para = Paragraph(html="<p>" + " ".join(f"a*b_c[{i}]--x" for i in range(900)) + "</p>")  # no sentence ends

    splits = split_oversized_element(Document(elements=[para]), renderer, tokenizer, max_tokens=400, overlap_tokens=20)

    assert len(splits) > 1
    for split in splits:
        assert tokenizer.count(renderer.render(split)) <= 400
