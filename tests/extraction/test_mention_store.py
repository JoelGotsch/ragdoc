"""Tests for LocalMentionStore — round-trip, per-source state, delete-by-source."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.extraction.mention import Mention
from ragdoc.extraction.stores import LocalMentionStore

from .conftest import Event


def mention(source_id: str, ordinal: int, *, source_hash: str = "h", content_hash: str = "c") -> Mention[Event]:
    return Mention[Event](
        mention_id=f"{source_id}-{ordinal}",
        source_id=source_id,
        source_hash=source_hash,
        content_hash=content_hash,
        ordinal=ordinal,
        metadata={"split_sequence": ordinal + 1},
        payload=Event(title=f"E{ordinal}"),
    )


@pytest.mark.anyio
async def test_round_trip(tmp_path: Path):
    store = LocalMentionStore(tmp_path, Event)
    await store.upsert([mention("a", 0), mention("a", 1), mention("b", 0)])

    assert await store.list_source_ids() == {"a", "b"}
    all_mentions = await store.list_mentions()
    assert len(all_mentions) == 3
    assert all(isinstance(m, Mention) for m in all_mentions)
    # payload rehydrates into the user model
    assert {m.payload.title for m in all_mentions} == {"E0", "E1"}


@pytest.mark.anyio
async def test_list_source_state(tmp_path: Path):
    store = LocalMentionStore(tmp_path, Event)
    await store.upsert([mention("a", 0, source_hash="h_a", content_hash="c_a"), mention("b", 0, source_hash="h_b")])
    state = await store.list_source_state()
    assert state["a"].source_hash == "h_a" and state["a"].content_hash == "c_a"
    assert state["b"].source_hash == "h_b"


@pytest.mark.anyio
async def test_delete_by_source(tmp_path: Path):
    store = LocalMentionStore(tmp_path, Event)
    await store.upsert([mention("a", 0), mention("b", 0)])
    await store.delete_by_source("a")
    assert await store.list_source_ids() == {"b"}


@pytest.mark.anyio
async def test_upsert_merges_by_mention_id(tmp_path: Path):
    store = LocalMentionStore(tmp_path, Event)
    await store.upsert([mention("a", 0)])
    await store.upsert([mention("a", 1)])  # same source, different id → accumulate
    assert len(store_by_source := await store.list_mentions()) == 2
    assert {m.mention_id for m in store_by_source} == {"a-0", "a-1"}


# ---------------------------------------------------------------------------
# Phase 4: list_mentions(payload_type=...) filter API
# ---------------------------------------------------------------------------


from typing import Literal

from pydantic import BaseModel

from ragdoc.extraction.schema import build_node_union


class _Person(BaseModel):
    kind: Literal["Person"] = "Person"
    full_name: str


class _Company(BaseModel):
    kind: Literal["Company"] = "Company"
    name: str


class _NotRegistered(BaseModel):
    kind: Literal["NotRegistered"] = "NotRegistered"
    x: int = 0


def _hetero_mention(payload, source_id: str, mention_id: str) -> Mention:
    return Mention(
        mention_id=mention_id,
        source_id=source_id,
        source_hash="h",
        content_hash="c",
        payload=payload,
    )


@pytest.mark.anyio
async def test_list_mentions_none_returns_all(tmp_path: Path):
    """Default behaviour — payload_type=None returns every stored mention."""
    node_union = build_node_union((_Person, _Company))
    store = LocalMentionStore(tmp_path, node_union)
    await store.upsert(
        [
            _hetero_mention(_Person(full_name="Alice"), "s1", "p-1"),
            _hetero_mention(_Company(name="Acme"), "s1", "c-1"),
        ]
    )
    all_mentions = await store.list_mentions()
    assert len(all_mentions) == 2


@pytest.mark.anyio
async def test_list_mentions_filter_by_node_type(tmp_path: Path):
    node_union = build_node_union((_Person, _Company))
    store = LocalMentionStore(tmp_path, node_union)
    await store.upsert(
        [
            _hetero_mention(_Person(full_name="Alice"), "s1", "p-1"),
            _hetero_mention(_Person(full_name="Bob"), "s1", "p-2"),
            _hetero_mention(_Company(name="Acme"), "s1", "c-1"),
        ]
    )
    persons = await store.list_mentions(_Person)
    companies = await store.list_mentions(_Company)

    assert {m.payload.full_name for m in persons} == {"Alice", "Bob"}
    assert {m.payload.name for m in companies} == {"Acme"}
    # Disjoint and exhaustive
    assert len(persons) + len(companies) == 3


@pytest.mark.anyio
async def test_list_mentions_unknown_type_returns_empty(tmp_path: Path):
    """Asking for a type not in the store returns []; no error (store doesn't track schema)."""
    node_union = build_node_union((_Person, _Company))
    store = LocalMentionStore(tmp_path, node_union)
    await store.upsert([_hetero_mention(_Person(full_name="Alice"), "s1", "p-1")])

    assert await store.list_mentions(_NotRegistered) == []


@pytest.mark.anyio
async def test_memory_mention_store_filter_api():
    """The in-memory MentionStore used by tests honours the same payload_type filter."""
    from .conftest import MemoryMentionStore

    store = MemoryMentionStore()
    await store.upsert(
        [
            _hetero_mention(_Person(full_name="Alice"), "s1", "p-1"),
            _hetero_mention(_Company(name="Acme"), "s1", "c-1"),
        ]
    )
    assert len(await store.list_mentions()) == 2
    assert len(await store.list_mentions(_Person)) == 1
    assert (await store.list_mentions(_Person))[0].payload.full_name == "Alice"
    assert await store.list_mentions(_NotRegistered) == []
