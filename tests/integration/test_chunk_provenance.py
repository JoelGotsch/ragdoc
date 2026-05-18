"""Integration test: verify source provenance propagates through the pipeline.

Covers the full flow from MinerU parser → DocumentPipeline → chunks and
VectorStorePipeline → source_id / source_hash on every chunk.

Bug regression: parsers must set document.source_path and metadata["filename"]
so they propagate into chunks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

from ragdoc.chunking.chunk import Chunk
from ragdoc.parsing.registry import get_parser
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline

_DATA_DIR = Path(__file__).parent.parent / "parsing" / "data" / "mineru"
_MINERU_FILE = _DATA_DIR / "attention-is-all-you-need_middle.json"

_DATA_PRESENT = _MINERU_FILE.exists()


# ---------------------------------------------------------------------------
# Minimal in-memory VectorStore
# ---------------------------------------------------------------------------


class _MemoryVectorStore:
    def __init__(self) -> None:
        self.stored: dict[str, Chunk] = {}

    async def upsert(self, chunks: list[Chunk]) -> list[str]:
        for c in chunks:
            self.stored[c.id] = c
        return [c.id for c in chunks]

    async def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self.stored.pop(id_, None)

    async def get_source_hash(self, source_id: str) -> str | None:
        for chunk in self.stored.values():
            if chunk.source_id == source_id:
                return chunk.source_hash
        return None

    async def delete_by_source(self, source_id: str) -> None:
        to_delete = [cid for cid, c in self.stored.items() if c.source_id == source_id]
        for cid in to_delete:
            del self.stored[cid]

    async def list_source_ids(self) -> set[str]:
        return {c.source_id for c in self.stored.values() if c.source_id}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_PRESENT, reason="MinerU test data missing")
@pytest.mark.anyio
async def test_chunks_have_source_path_and_metadata_filename():
    """Chunks must carry source_path and metadata["filename"] from the parsed document."""
    chunks = await DocumentPipeline(parser=get_parser("mineru")).run(_MINERU_FILE)

    assert chunks, "Pipeline produced no chunks"
    for chunk in chunks:
        assert chunk.source_path == str(_MINERU_FILE), (
            f"chunk.source_path={chunk.source_path!r}, expected {str(_MINERU_FILE)!r}"
        )
        assert chunk.metadata.get("filename") == _MINERU_FILE.name, (
            f"chunk.metadata['filename']={chunk.metadata.get('filename')!r}, expected {_MINERU_FILE.name!r}"
        )


@pytest.mark.skipif(not _DATA_PRESENT, reason="MinerU test data missing")
@pytest.mark.anyio
async def test_vector_store_pipeline_sets_source_id_and_source_hash():
    """VectorStorePipeline must set source_id and source_hash on every chunk."""
    vstore = _MemoryVectorStore()
    result = await VectorStorePipeline(
        pipeline=DocumentPipeline(parser=get_parser("mineru")),
        vector_store=vstore,
    ).run([_MINERU_FILE])

    assert not result.errors, f"Pipeline errors: {result.errors}"
    chunks = list(vstore.stored.values())
    assert chunks, "No chunks produced"

    for chunk in chunks:
        assert chunk.source_id is not None, f"chunk.source_id is None (id={chunk.id})"
        assert chunk.source_hash is not None, f"chunk.source_hash is None (id={chunk.id})"
        assert chunk.source_path == str(_MINERU_FILE)
        assert chunk.metadata.get("filename") == _MINERU_FILE.name
