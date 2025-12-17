"""Unit tests for the Chunk model provenance contract (Phase 1 of sync architecture).

Covers:
- source_id / source_hash are REQUIRED (no default) — construction without them raises.
- content_hash is optional (new field).
- A fully-specified Chunk constructs and round-trips through model_dump_json.

See plans/DESIGN_SYNC_ARCHITECTURE.md (Option A: model-level required, chunker fallbacks).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragdoc.chunking import Chunk


def test_chunk_requires_source_id_and_source_hash():
    with pytest.raises(ValidationError):
        Chunk(prompt_content="p", embedding_content="e")  # type: ignore[call-arg]


def test_chunk_requires_source_id_even_when_source_hash_given():
    with pytest.raises(ValidationError):
        Chunk(prompt_content="p", embedding_content="e", source_hash="h")  # type: ignore[call-arg]


def test_chunk_source_id_may_not_be_none():
    with pytest.raises(ValidationError):
        Chunk(
            prompt_content="p",
            embedding_content="e",
            source_id=None,  # type: ignore[arg-type]
            source_hash="h",
        )


def test_chunk_constructs_with_required_provenance():
    chunk = Chunk(
        prompt_content="p",
        embedding_content="e",
        source_id="doc.pdf",
        source_hash="abc",
    )
    assert chunk.source_id == "doc.pdf"
    assert chunk.source_hash == "abc"
    assert chunk.content_hash is None  # optional, defaults to None


def test_chunk_content_hash_round_trips():
    chunk = Chunk(
        prompt_content="p",
        embedding_content="e",
        source_id="doc.pdf",
        source_hash="abc",
        content_hash="deadbeef",
    )
    reloaded = Chunk.model_validate_json(chunk.model_dump_json())
    assert reloaded.content_hash == "deadbeef"
    assert reloaded.id == chunk.id
