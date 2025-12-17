"""Unit tests for chunking/provenance.py: resolve_chunk_provenance and mint_chunk_id."""

from __future__ import annotations

import re

import pytest

from ragdoc.chunking.provenance import mint_chunk_id, resolve_chunk_provenance
from ragdoc.document import Document, Paragraph

# ---------------------------------------------------------------------------
# mint_chunk_id
# ---------------------------------------------------------------------------


def test_mint_chunk_id_deterministic():
    a = mint_chunk_id("src", 1, 0, "hash")
    b = mint_chunk_id("src", 1, 0, "hash")
    assert a == b
    assert re.fullmatch(r"[0-9a-f]{64}", a)


@pytest.mark.parametrize(
    "other",
    [
        ("other-src", 1, 0, "hash"),
        ("src", 2, 0, "hash"),
        ("src", 1, 1, "hash"),
        ("src", 1, 0, "other-hash"),
    ],
)
def test_mint_chunk_id_each_component_changes_id(other):
    assert mint_chunk_id("src", 1, 0, "hash") != mint_chunk_id(*other)


def test_mint_chunk_id_unit_separator_prevents_concatenation_ambiguity():
    # "a" + seq 11 vs "a1" + seq 1 must not collide.
    assert mint_chunk_id("a", 11, 0, "h") != mint_chunk_id("a1", 1, 0, "h")


# ---------------------------------------------------------------------------
# resolve_chunk_provenance
# ---------------------------------------------------------------------------


def test_resolve_prefers_source_id():
    doc = Document(elements=[Paragraph(html="<p>x</p>")], source_path="/tmp/a.html")
    doc.source_id = "sid"
    assert resolve_chunk_provenance(doc).source_id == "sid"


def test_resolve_falls_back_to_source_path_then_id():
    doc = Document(elements=[Paragraph(html="<p>x</p>")], source_path="/tmp/a.html")
    assert resolve_chunk_provenance(doc).source_id == "/tmp/a.html"
    bare = Document(elements=[Paragraph(html="<p>x</p>")])
    assert resolve_chunk_provenance(bare).source_id == bare.id


def test_resolve_source_hash_passthrough_never_faked():
    doc = Document(elements=[Paragraph(html="<p>x</p>")])
    assert resolve_chunk_provenance(doc).source_hash is None  # no content-hash masquerade
    doc.source_hash = "file-bytes-hash"
    assert resolve_chunk_provenance(doc).source_hash == "file-bytes-hash"


def test_resolve_content_hash_populated():
    doc = Document(elements=[Paragraph(html="<p>x</p>")])
    assert resolve_chunk_provenance(doc).content_hash == doc.content_hash()
