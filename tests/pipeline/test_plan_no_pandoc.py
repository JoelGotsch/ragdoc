"""Phase-5 exit criterion: change detection is pandoc-free.

A Boundary-2 ``VectorStorePipeline.plan()`` over an unchanged corpus recomputes every
document's ``content_hash`` (LocalDocumentStore is index-free) but must invoke
``pypandoc.convert_text`` **zero** times — the canonical-JSON hash needs no rendering,
and unchanged sources are never re-chunked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.pipeline import ChunkPipeline, LocalDocumentStore, VectorStorePipeline

from .conftest import MemoryVectorStore, make_document


@pytest.mark.anyio
async def test_plan_unchanged_corpus_invokes_no_pandoc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    doc_store = LocalDocumentStore(tmp_path)
    for sid, body in [("a.pdf", "Alpha body"), ("b.pdf", "Beta body")]:
        doc = make_document(title=sid, body=body)
        doc.source_id = sid
        doc.source_hash = f"filehash::{sid}"
        await doc_store.upsert([doc])

    vec = MemoryVectorStore()
    pipeline = VectorStorePipeline.from_document_store(
        chunk=ChunkPipeline(),
        vector_store=vec,
        document_store=doc_store,
    )
    # Bring the vector store in sync (chunk rendering here may legitimately use pandoc).
    result = await pipeline.run()
    assert set(result.processed) == {"a.pdf", "b.pdf"}

    # Spy on every route to pandoc: the pypandoc module itself plus the names already
    # bound via `from pypandoc import convert_text` in ragdoc modules.
    import pypandoc

    import ragdoc.document as document_module
    import ragdoc.rendering.base as rendering_base

    calls: list[tuple[object, ...]] = []

    def _spy(*args: object, **kwargs: object) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(pypandoc, "convert_text", _spy)
    monkeypatch.setattr(document_module, "convert_text", _spy)
    monkeypatch.setattr(rendering_base, "convert_text", _spy)

    changeset = await pipeline.plan()  # unchanged corpus

    assert changeset.to_add == []
    assert changeset.to_update == []
    assert changeset.to_delete == []
    assert calls == [], f"plan() over an unchanged corpus made {len(calls)} pypandoc.convert_text call(s)"
