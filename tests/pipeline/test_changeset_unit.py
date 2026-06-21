"""Unit tests for the ChangeSet reviewable artifact (Phase 4).

Covers:
- ChangeSet[Chunk] save -> load round-trip PRESERVES chunk.id and created_at
  (idempotent upsert depends on this).
- ChangeSet[Document] round-trip.
- Editing the saved JSON between save and load is reflected after load.
- to_delete is preserved.

Note: ChangeSet is a generic model — load() must be called on the CONCRETE
parametrization (ChangeSet[Chunk].load(...)) so Pydantic can resolve list[T].
"""

from __future__ import annotations

import json
from pathlib import Path

from ragdoc.chunking import Chunk
from ragdoc.document import Document, Paragraph
from ragdoc.pipeline.changeset import ChangeSet, SourceChange


def _chunk(source_id: str, body: str = "x") -> Chunk:
    return Chunk(
        prompt_content=body,
        embedding_content=body,
        source_id=source_id,
        source_hash="h",
        content_hash="ch",
    )


def test_changeset_chunk_round_trip_preserves_id_and_created_at(tmp_path: Path):
    chunk = _chunk("a.pdf")
    cs: ChangeSet[Chunk] = ChangeSet(
        to_add=[SourceChange(source_id="a.pdf", source_hash="h", content_hash="ch", items=[chunk])]
    )
    p = tmp_path / "cs.json"
    cs.save(p)

    loaded = ChangeSet[Chunk].load(p)
    reloaded_chunk = loaded.to_add[0].items[0]
    assert reloaded_chunk.id == chunk.id
    assert reloaded_chunk.created_at == chunk.created_at
    assert reloaded_chunk.source_id == "a.pdf"
    assert reloaded_chunk.content_hash == "ch"


def test_changeset_document_round_trip(tmp_path: Path):
    doc = Document(elements=[Paragraph(html_content="<p>hi</p>")], source_id="d.pdf")
    cs: ChangeSet[Document] = ChangeSet(to_update=[SourceChange(source_id="d.pdf", source_hash="h", items=[doc])])
    p = tmp_path / "cs.json"
    cs.save(p)

    loaded = ChangeSet[Document].load(p)
    reloaded_doc = loaded.to_update[0].items[0]
    assert reloaded_doc.id == doc.id
    assert reloaded_doc.source_id == "d.pdf"
    assert len(reloaded_doc.elements) == 1


def test_changeset_to_delete_preserved(tmp_path: Path):
    cs: ChangeSet[Chunk] = ChangeSet(to_delete=["gone-1.pdf", "gone-2.pdf"])
    p = tmp_path / "cs.json"
    cs.save(p)
    assert ChangeSet[Chunk].load(p).to_delete == ["gone-1.pdf", "gone-2.pdf"]


def test_changeset_manual_edit_between_save_and_load(tmp_path: Path):
    """A human edits the JSON (drops a chunk); load reflects the edit."""
    cs: ChangeSet[Chunk] = ChangeSet(
        to_add=[
            SourceChange(
                source_id="a.pdf",
                source_hash="h",
                items=[_chunk("a.pdf", "keep"), _chunk("a.pdf", "drop")],
            )
        ]
    )
    p = tmp_path / "cs.json"
    cs.save(p)

    data = json.loads(p.read_text())
    data["to_add"][0]["items"] = data["to_add"][0]["items"][:1]  # drop second chunk
    p.write_text(json.dumps(data))

    loaded = ChangeSet[Chunk].load(p)
    assert len(loaded.to_add[0].items) == 1
    assert loaded.to_add[0].items[0].prompt_content == "keep"


def test_changeset_defaults_are_empty(tmp_path: Path):
    cs: ChangeSet[Chunk] = ChangeSet()
    assert cs.to_add == [] and cs.to_update == [] and cs.to_delete == []
