"""Unit tests for Document.content_hash() — canonical-JSON, pandoc-free (v1).

Covers the field-participation contract (title + elements in; metadata, provenance,
page/bbox out), ref-ordinal normalization (re-parse stability), the fail-loud
exhaustiveness guard, and two pinned golden hashes.

No pandoc, no I/O — pure construction and hashing.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import pytest

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
    element_content_payload,
    normalize_ref_ids,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def build_full_document(
    img_id: str = "img-fixed",
    fn_id: str = "fn-fixed",
    dangling_id: str = "missing-fixed",
) -> Document:
    """A Document with all seven element types, a resolved image ref, a footnote ref,
    a dangling ref, a title, and inline CSS.

    Element ids are parameters so tests can prove that ids (and the ref tags embedding
    them) do not affect the hash.
    """
    return Document(
        title="Golden Fixture",
        elements=[
            Heading(html='<h1 style="font-size:24px">Golden Fixture</h1>'),
            Paragraph(
                html=(
                    '<p style="text-align:left">See <ref id="'
                    + img_id
                    + '" rel="image"/> and note<ref id="'
                    + fn_id
                    + '" rel="footnote"/> and <ref id="'
                    + dangling_id
                    + '" rel="figure"/>.</p>'
                )
            ),
            Table(html="<table><tr><td>cell</td></tr></table>"),
            DocumentList(html="<ul><li>item one</li></ul>"),
            Image(
                id=img_id,
                image="aGVsbG8=",
                image_type="png",
                width=100,
                height=50,
                alt="A chart",
                text_representation="<table><tr><td>42</td></tr></table>",
            ),
            RawText(html="raw <i>text</i>"),
            Footnote(id=fn_id, number=1, innerhtml="Footnote text with <b>bold</b>."),
        ],
    )


# ---------------------------------------------------------------------------
# Shape + golden hashes
# ---------------------------------------------------------------------------


def test_hash_is_64_char_lowercase_hex():
    h = build_full_document().content_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_golden_hash_full_fixture():
    # Changing this constant re-embeds every deployed corpus — bump
    # _CONTENT_HASH_DOMAIN deliberately and add a CHANGELOG migration note.
    assert build_full_document().content_hash() == "d69b8a53a4b360e94400f7f7ca2a7643d94bc1fa3c695d25117e6669dc6ad7bf"


def test_golden_hash_empty_document():
    # Changing this constant re-embeds every deployed corpus — bump
    # _CONTENT_HASH_DOMAIN deliberately and add a CHANGELOG migration note.
    assert Document().content_hash() == "3d34da2725660143006466065d758ebef2a49a61c1b82be6e3b6865c5f690d4c"


# ---------------------------------------------------------------------------
# Stability (what must NOT change the hash)
# ---------------------------------------------------------------------------


def test_same_content_same_hash():
    assert build_full_document().content_hash() == build_full_document().content_hash()


def test_element_ids_do_not_affect_hash():
    """Re-parse stability: fresh element uuids — including rewritten <ref id=...>
    targets — must not change the hash (the ref-ordinal normalization at work)."""
    doc_a = build_full_document(img_id="11111111-aaaa", fn_id="22222222-bbbb")
    doc_b = build_full_document(img_id="33333333-cccc", fn_id="44444444-dddd")
    assert doc_a.content_hash() == doc_b.content_hash()


def test_dangling_ref_id_does_not_affect_hash():
    doc_a = build_full_document(dangling_id="dead-beef-1")
    doc_b = build_full_document(dangling_id="dead-beef-2")
    assert doc_a.content_hash() == doc_b.content_hash()


def test_document_metadata_does_not_affect_hash():
    doc = build_full_document()
    base = doc.content_hash()
    doc.metadata = {"filename": "report.pdf", "split_sequence": 2, "split_total": 5, "custom": "x"}
    assert doc.content_hash() == base


def test_element_metadata_page_bbox_do_not_affect_hash():
    doc = build_full_document()
    base = doc.content_hash()
    element = doc.elements[1]
    element.metadata = {"parser_note": "volatile"}
    element.page = 7
    element.bounding_box = (0.1, 0.2, 0.3, 0.4)
    assert doc.content_hash() == base


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: setattr(d, "id", "other-doc-id"), id="id"),
        pytest.param(lambda d: setattr(d, "source_path", "elsewhere/report.pdf"), id="source_path"),
        pytest.param(lambda d: setattr(d, "source_id", "sync-id"), id="source_id"),
        pytest.param(lambda d: setattr(d, "source_hash", "deadbeef"), id="source_hash"),
        pytest.param(lambda d: setattr(d, "parser", "mineru"), id="parser"),
        pytest.param(lambda d: setattr(d, "parser_version", "2.0"), id="parser_version"),
        pytest.param(
            lambda d: setattr(d, "external_refs", [ExternalRef(target_id="t", rel_type="external-parent")]),
            id="external_refs",
        ),
    ],
)
def test_provenance_fields_do_not_affect_hash(mutate):
    doc = build_full_document()
    base = doc.content_hash()
    mutate(doc)
    assert doc.content_hash() == base


def test_footnote_own_id_does_not_affect_hash():
    """Footnote hashes innerhtml, not html — the volatile id="footnote-{uuid}" is excluded."""
    fn_a = Footnote(id="fn-one", number=1, innerhtml="Same text.")
    fn_b = Footnote(id="fn-two", number=1, innerhtml="Same text.")
    assert element_content_payload(fn_a, {}) == element_content_payload(fn_b, {})


def test_hash_stable_across_serialization_roundtrip():
    """The LocalDocumentStore path: model_dump_json -> model_validate_json hashes equal."""
    doc = build_full_document()
    reloaded = Document.model_validate_json(doc.model_dump_json())
    assert reloaded.content_hash() == doc.content_hash()


# ---------------------------------------------------------------------------
# Sensitivity (what MUST change the hash)
# ---------------------------------------------------------------------------


def test_title_change_changes_hash():
    doc_a = build_full_document()
    doc_b = build_full_document()
    doc_b.title = "Another Title"
    assert doc_a.content_hash() != doc_b.content_hash()


def test_element_text_edit_changes_hash():
    doc_a = build_full_document()
    doc_b = build_full_document()
    doc_b.elements[2] = Table(html="<table><tr><td>edited</td></tr></table>")
    assert doc_a.content_hash() != doc_b.content_hash()


def test_element_reorder_changes_hash():
    para_a = Paragraph(html="<p>Alpha</p>")
    para_b = Paragraph(html="<p>Beta</p>")
    doc_ab = Document(elements=[para_a, para_b])
    doc_ba = Document(elements=[para_b, para_a])
    assert doc_ab.content_hash() != doc_ba.content_hash()


def test_element_type_change_changes_hash():
    """Same html string, different element type -> different hash (structure sensitivity)."""
    raw = RawText(html="x")  # html == "<div>x</div>"
    para = Paragraph(html="<div>x</div>")
    assert raw.html == para.html
    assert Document(elements=[raw]).content_hash() != Document(elements=[para]).content_hash()


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"image": "b3RoZXI="}, id="image"),
        pytest.param({"image_type": "jpeg"}, id="image_type"),
        pytest.param({"alt": "Different alt"}, id="alt"),
        pytest.param({"text_representation": "different text"}, id="text_representation"),
        pytest.param({"width": 101}, id="width"),
        pytest.param({"height": 51}, id="height"),
    ],
)
def test_image_content_changes_hash(override: dict[str, str | int]):
    base_kwargs: dict[str, str | int] = {
        "image": "aGVsbG8=",
        "image_type": "png",
        "alt": "A chart",
        "text_representation": "text repr",
        "width": 100,
        "height": 50,
    }
    img_a = Image(**base_kwargs)  # pyright: ignore[reportArgumentType]
    img_b = Image(**{**base_kwargs, **override})  # pyright: ignore[reportArgumentType]
    assert Document(elements=[img_a]).content_hash() != Document(elements=[img_b]).content_hash()


def test_footnote_number_and_text_change_hash():
    base = Document(elements=[Footnote(number=1, innerhtml="Same text.")]).content_hash()
    renumbered = Document(elements=[Footnote(number=2, innerhtml="Same text.")]).content_hash()
    rewritten = Document(elements=[Footnote(number=1, innerhtml="Other text.")]).content_hash()
    assert base != renumbered
    assert base != rewritten


def test_ref_rel_type_participates():
    """Same target ordinal, different rel -> different hash."""

    def doc_with_rel(rel: str) -> Document:
        img = Image(id="img-1", image="aGVsbG8=")
        para = Paragraph(html=f'<p>See <ref id="img-1" rel="{rel}"/>.</p>')
        return Document(elements=[para, img])

    assert doc_with_rel("image").content_hash() != doc_with_rel("footnote").content_hash()


# ---------------------------------------------------------------------------
# normalize_ref_ids + exhaustiveness guard + subclass override
# ---------------------------------------------------------------------------


def test_normalize_ref_ids_only_touches_ref_tags():
    html = '<p id="keep-me">See <a id="also-keep" href="#x">link</a> and <ref id="uuid-1" rel="image"/>.</p>'
    normalized = normalize_ref_ids(html, {"uuid-1": 3})
    assert 'id="keep-me"' in normalized
    assert 'id="also-keep"' in normalized
    assert '<ref id="#3" rel="image"/>' in normalized


def test_normalize_ref_ids_dangling_ref_becomes_unresolved():
    assert normalize_ref_ids('<ref id="nope" rel="figure"/>', {}) == '<ref id="unresolved" rel="figure"/>'


def test_element_payload_raises_on_unknown_element_type():
    class MysteryElement(BaseElement):
        element_type: Literal["mystery"] = "mystery"

    with pytest.raises(TypeError, match="MysteryElement"):
        element_content_payload(MysteryElement(), {})


def test_subclass_override_composes():
    """The docstring's customization example: compose over super().content_hash()."""

    class SourceDocument(Document):
        def content_hash(self) -> str:
            base = super().content_hash()
            return hashlib.sha256(f"{self.source_path}:{base}".encode()).hexdigest()

    doc = SourceDocument(title="T", source_path="a/b.pdf")
    plain = Document(title="T", source_path="a/b.pdf")
    h = doc.content_hash()
    assert len(h) == 64
    assert h != plain.content_hash()
    assert doc.content_hash() == h  # deterministic
