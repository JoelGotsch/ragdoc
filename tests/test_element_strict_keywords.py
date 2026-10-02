"""Unknown constructor keywords are errors — nothing is silently dropped.

``html_content=`` (Heading/Paragraph/Table/DocumentList) and ``innerhtml=`` (RawText) were the
canonical spellings before the stored-``html`` refactor. They are **not** accepted as aliases:
passing them raises a ``ValidationError`` that names the new field. The same applies to data
loaded with ``model_validate`` — JSON written by the old model fails loudly instead of
producing empty elements.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field, ValidationError

from ragdoc.document import (
    BaseElement,
    Document,
    DocumentList,
    ElementTypeEnum,
    ExternalRef,
    Footnote,
    Heading,
    Image,
    Paragraph,
    RawText,
    Table,
)

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjkB6QAAAABJRU5ErkJggg=="

# ---------------------------------------------------------------------------
# Renamed keywords: clear error naming the new field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "content"),
    [
        (Heading, "<h1>lost</h1>"),
        (Paragraph, "<p>lost</p>"),
        (Table, "<table><tr><td>lost</td></tr></table>"),
        (DocumentList, "<ul><li>lost</li></ul>"),
    ],
)
def test_html_content_keyword_raises_and_names_html(cls, content):
    with pytest.raises(ValidationError) as excinfo:
        cls(html_content=content)
    message = str(excinfo.value)
    assert "html_content" in message
    assert "html=" in message
    assert cls.__name__ in message


def test_rawtext_innerhtml_keyword_raises_and_names_html():
    with pytest.raises(ValidationError) as excinfo:
        RawText(innerhtml="lost")  # pyright: ignore[reportCallIssue]
    message = str(excinfo.value)
    assert "innerhtml" in message
    assert "html=" in message
    assert "RawText" in message


def test_renamed_keyword_raises_even_alongside_the_new_one():
    with pytest.raises(ValidationError, match="html_content"):
        Paragraph(html="<p>kept</p>", html_content="<p>lost</p>")  # pyright: ignore[reportCallIssue]


def test_footnote_innerhtml_is_still_a_real_field():
    footnote = Footnote(number=1, innerhtml="still <em>valid</em>")
    assert footnote.innerhtml == "still <em>valid</em>"


# ---------------------------------------------------------------------------
# Any other unknown keyword
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: Heading(html="<h1>x</h1>", bogus=1),
        lambda: Paragraph(html="<p>x</p>", bogus=1),
        lambda: Table(html="<table></table>", bogus=1),
        lambda: DocumentList(html="<ul></ul>", bogus=1),
        lambda: RawText(html="x", bogus=1),
        lambda: Image(image=_PNG_B64, bogus=1),
        lambda: Footnote(number=1, innerhtml="x", bogus=1),
        lambda: Document(bogus=1),
        lambda: Document(filename="old-field.pdf"),
    ],
    ids=["Heading", "Paragraph", "Table", "DocumentList", "RawText", "Image", "Footnote", "Document", "Document-old"],
)
def test_unknown_keyword_raises(build):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        build()


def test_html_keyword_still_accepted_on_derived_html_types():
    """Image/Footnote have no stored ``html`` field; the keyword is consumed by their before-validators."""
    image = Image(html=f'<img src="data:image/png;base64,{_PNG_B64}" alt="a" width="1" height="1"/>')
    assert image.image == _PNG_B64
    assert image.image_type == "png"
    footnote = Footnote(html='<aside class="footnote" data-number="3">note</aside>')
    assert footnote.number == 3
    assert footnote.innerhtml == "note"


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def _full_document() -> Document:
    footnote = Footnote(number=1, innerhtml="note with <em>markup</em>", page=2)
    image = Image(
        image=_PNG_B64, image_type="png", alt='a "quoted" <alt>', width=1, height=1, text_representation="a pixel"
    )
    return Document(
        title="Round trip",
        source_path="/data/report.pdf",
        source_id="report.pdf",
        source_hash="abc",
        parser="html",
        parser_version="1",
        metadata={"filename": "report.pdf", "custom": {"nested": [1, 2]}},
        external_refs=[ExternalRef(target_id="parent", rel_type="external-parent")],
        elements=[
            Heading(html='<h2 style="text-align: center">Title</h2>', page=1, bounding_box=(0.1, 0.2, 0.3, 0.4)),
            Paragraph(
                html=f'<p>Text<ref id="{footnote.id}" rel="footnote"/> and <ref id="{image.id}" rel="image"/></p>'
            ),
            Table(html="<table><tr><td>a &amp; b</td></tr></table>", metadata={"k": "v"}),
            DocumentList(html="<ol><li>one</li><li>two</li></ol>"),
            RawText(html="<div><pre><code>x &lt; y</code></pre></div>"),
            RawText(html="<config>\n <a>1</a>\n</config>"),
            image,
            footnote,
        ],
    )


def test_dump_validate_round_trip_for_every_element_type():
    doc = _full_document()

    from_python = Document.model_validate(doc.model_dump())
    from_json_mode = Document.model_validate(doc.model_dump(mode="json"))
    from_json = Document.model_validate_json(doc.model_dump_json())

    for restored in (from_python, from_json_mode, from_json):
        assert [type(el) for el in restored.elements] == [type(el) for el in doc.elements]
        assert restored.elements == doc.elements
        assert [el.html for el in restored.elements] == [el.html for el in doc.elements]
        assert restored.model_dump() == doc.model_dump()
        assert restored.content_hash() == doc.content_hash()


@pytest.mark.parametrize("element", _full_document().elements, ids=lambda el: type(el).__name__)
def test_each_element_round_trips_on_its_own(element):
    cls = type(element)
    assert cls.model_validate(element.model_dump()) == element
    assert cls.model_validate_json(element.model_dump_json()) == element
    assert "html_content" not in element.model_dump()


def test_dump_of_derived_html_types_has_no_html_key():
    """Image/Footnote derive html — it must not be serialized (it would be rejected on load)."""
    assert "html" not in Image(image=_PNG_B64).model_dump()
    assert "html" not in Footnote(number=1, innerhtml="x").model_dump()


def test_old_format_json_fails_loudly_instead_of_losing_content():
    """A document serialized by the pre-refactor model (``html_content`` / RawText ``innerhtml``)."""
    old_paragraph = {"element_type": "paragraph", "id": "p1", "html_content": "<p>important</p>"}
    old_rawtext = {"element_type": "raw_text", "id": "r1", "innerhtml": "important"}

    with pytest.raises(ValidationError, match="html_content"):
        Document.model_validate({"elements": [old_paragraph]})
    with pytest.raises(ValidationError, match="innerhtml"):
        Document.model_validate({"elements": [old_rawtext]})


# ---------------------------------------------------------------------------
# Custom element subclasses
# ---------------------------------------------------------------------------


class Callout(BaseElement):
    """A custom element in the style of docs/notebooks/custom_elements.py."""

    element_type: Literal["callout"] = "callout"  # pyright: ignore[reportIncompatibleVariableOverride]
    kind: str = Field(default="note", description="Callout kind.")
    body: str = Field(default="", description="Callout body text.")

    @property
    def html(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return f'<div class="callout {self.kind}">{self.body}</div>'


def test_custom_subclass_fields_work_and_unknown_keywords_raise():
    callout = Callout(kind="warning", body="careful")
    assert callout.html == '<div class="callout warning">careful</div>'
    assert callout.text == "careful"
    assert Callout.model_validate(callout.model_dump()) == callout
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Callout(kind="warning", bodyy="typo")  # pyright: ignore[reportCallIssue]


def test_element_type_enum_values_unchanged():
    assert Paragraph(html="<p>x</p>").element_type == ElementTypeEnum.PARAGRAPH
