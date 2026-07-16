"""Unit tests for the generic SyncEngine (spec §3a-1 step 2).

Exercised directly against a stub store and scripted producers (no pipelines):

- streaming: run() writes each source as its event is yielded (not buffered);
- task cleanup: an aborted stream cancels in-flight producer tasks (no leaks);
- pre_write scheduling: per source in run(), once over the whole corpus in apply();
- orphan emission: plan() buffers orphans into to_delete, run() deletes them;
- build_current_map: collisions raise before any hashing;
- protocol conformance: the concrete store protocols satisfy SourceSyncStore.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ragdoc.chunking import Chunk
from ragdoc.document import Document
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.stores import MentionStore
from ragdoc.pipeline.changeset import SourceChange
from ragdoc.pipeline.stores import DocumentStore, SourceState, VectorStore
from ragdoc.pipeline.sync import (
    SourceSyncStore,
    SyncEngine,
    SyncPlanInput,
    SyncSource,
    build_current_map,
    source_hash_token,
)

from .conftest import MemoryVectorStore

# ---------------------------------------------------------------------------
# Stub payload + store
# ---------------------------------------------------------------------------


class Item(BaseModel):
    """Minimal sync payload for engine unit tests."""

    source_id: str = Field(description="Sync identity key.")
    source_hash: str = Field(description="Change token stored for this item's source.")
    marker: str = Field(description="Content marker to tell versions apart.")


class StubStore:
    """In-memory SourceSyncStore[Item] recording the order of write operations."""

    def __init__(self) -> None:
        self.items: dict[str, list[Item]] = {}
        self.events: list[str] = []

    async def upsert(self, items: list[Item]) -> list[str]:
        for item in items:
            self.items.setdefault(item.source_id, []).append(item)
        self.events.extend(f"upsert:{item.source_id}" for item in items)
        return [item.marker for item in items]

    async def delete_by_source(self, source_id: str) -> None:
        self.items.pop(source_id, None)
        self.events.append(f"delete:{source_id}")

    async def list_source_state(self) -> dict[str, SourceState]:
        return {
            sid: SourceState(source_hash=items[0].source_hash, content_hash=None)
            for sid, items in self.items.items()
            if items
        }


def change_for(src: SyncSource, marker: str) -> SourceChange[Item]:
    return SourceChange(
        source_id=src.source_id,
        source_hash=src.change_token,
        content_hash=None,
        items=[Item(source_id=src.source_id, source_hash=src.change_token, marker=marker)],
    )


def request_for(*sids: str) -> SyncPlanInput:
    return SyncPlanInput(
        sources=[SyncSource(source_id=sid, change_token=f"tok::{sid}") for sid in sids],
        live_source_ids=frozenset(sids),
    )


# ---------------------------------------------------------------------------
# streaming: run() writes as events are yielded
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_writes_fast_source_before_slow_source_finishes():
    """The fast source's upsert lands while the slow producer is still running."""
    fast_written = asyncio.Event()

    class SignallingStore(StubStore):
        async def upsert(self, items: list[Item]) -> list[str]:
            ids = await super().upsert(items)
            if "upsert:fast" in self.events:
                fast_written.set()
            return ids

    store = SignallingStore()

    async def producer(src: SyncSource) -> SourceChange[Item] | None:
        if src.source_id == "slow":
            await asyncio.wait_for(fast_written.wait(), timeout=5)
        return change_for(src, marker=f"v1-{src.source_id}")

    engine: SyncEngine[Item] = SyncEngine(store=store, producer=producer, token_of=source_hash_token)
    result = await engine.run(request_for("fast", "slow"))

    assert set(result.processed) == {"fast", "slow"}
    # fast was written before slow's production completed — the stream is not buffered
    assert store.events.index("upsert:fast") < store.events.index("upsert:slow")


@pytest.mark.anyio
async def test_run_second_pass_skips_via_token():
    store = StubStore()
    calls: list[str] = []

    async def producer(src: SyncSource) -> SourceChange[Item] | None:
        calls.append(src.source_id)
        return change_for(src, marker="v1")

    engine: SyncEngine[Item] = SyncEngine(store=store, producer=producer, token_of=source_hash_token)
    await engine.run(request_for("a"))
    result = await engine.run(request_for("a"))

    assert result.skipped == ["a"] and result.processed == []
    assert calls == ["a"]  # no second production call


# ---------------------------------------------------------------------------
# task cleanup on abort
# ---------------------------------------------------------------------------


class _WeirdBaseError(BaseException):
    """BaseException subclass that is neither Exception nor CancelledError."""


@pytest.mark.anyio
async def test_aborted_run_cancels_inflight_producers():
    """A BaseException aborting the stream cancels (and awaits) the in-flight tasks."""
    store = StubStore()
    hang_started = asyncio.Event()
    hang_cancelled = asyncio.Event()

    async def producer(src: SyncSource) -> SourceChange[Item] | None:
        if src.source_id == "hang":
            hang_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                hang_cancelled.set()
                raise
            return None
        await hang_started.wait()  # guarantee "hang" is in flight before we blow up
        raise _WeirdBaseError()

    engine: SyncEngine[Item] = SyncEngine(store=store, producer=producer, token_of=source_hash_token)
    before = asyncio.all_tasks()
    with pytest.raises(_WeirdBaseError):
        await engine.run(request_for("hang", "boom"))

    assert hang_cancelled.is_set(), "in-flight producer task must be cancelled on abort"
    leftovers = {t for t in asyncio.all_tasks() - before if not t.done()}
    assert not leftovers, f"aborted run left pending tasks: {leftovers}"


# ---------------------------------------------------------------------------
# pre_write scheduling: per source in run(), whole corpus in apply()
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pre_write_per_source_in_run_and_once_in_apply():
    async def producer(src: SyncSource) -> SourceChange[Item] | None:
        return change_for(src, marker="v1")

    pre_write_calls: list[int] = []

    async def pre_write(items: list[Item]) -> None:
        pre_write_calls.append(len(items))

    def make_engine(store: StubStore) -> SyncEngine[Item]:
        return SyncEngine(store=store, producer=producer, token_of=source_hash_token, pre_write=pre_write)

    await make_engine(StubStore()).run(request_for("a", "b"))
    assert sorted(pre_write_calls) == [1, 1]  # once per source

    pre_write_calls.clear()
    plan_engine = make_engine(StubStore())
    changeset = await plan_engine.plan(request_for("a", "b"))
    assert pre_write_calls == []  # plan never pre-writes
    await plan_engine.apply(changeset)
    assert pre_write_calls == [2]  # once, whole corpus


@pytest.mark.anyio
async def test_apply_pre_write_failure_leaves_store_untouched():
    async def producer(src: SyncSource) -> SourceChange[Item] | None:
        return change_for(src, marker="v1")

    async def failing_pre_write(items: list[Item]) -> None:
        raise RuntimeError("pre-write boom")

    store = StubStore()
    engine: SyncEngine[Item] = SyncEngine(
        store=store, producer=producer, token_of=source_hash_token, pre_write=failing_pre_write
    )
    changeset = await engine.plan(request_for("a", "b"))
    result = await engine.apply(changeset)

    assert store.items == {} and store.events == []
    assert sorted(sid for sid, _ in result.errors) == ["a", "b"]
    assert result.processed == []


# ---------------------------------------------------------------------------
# orphan emission
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orphans_planned_and_deleted():
    async def producer(src: SyncSource) -> SourceChange[Item] | None:
        return change_for(src, marker="v1")

    store = StubStore()
    store.items["old"] = [Item(source_id="old", source_hash="tok::old", marker="stale")]
    engine: SyncEngine[Item] = SyncEngine(store=store, producer=producer, token_of=source_hash_token)

    changeset = await engine.plan(request_for("a"), delete_orphans=True)
    assert changeset.to_delete == ["old"]
    assert "old" in store.items  # plan never writes

    result = await engine.run(request_for("a"), delete_orphans=True)
    assert result.deleted == ["old"]
    assert "old" not in store.items


# ---------------------------------------------------------------------------
# build_current_map
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_build_current_map_collision_raises_before_hashing(tmp_path: Path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    p1, p2 = tmp_path / "d1" / "s.txt", tmp_path / "d2" / "s.txt"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")
    hash_calls: list[Path] = []

    def spy_hash(path: Path) -> str:
        hash_calls.append(path)
        return "h"

    with pytest.raises(ValueError, match="duplicate source_ids"):
        await build_current_map([p1, p2], lambda p: p.name, spy_hash, concurrency=4)
    assert hash_calls == []  # strictly before any hashing


@pytest.mark.anyio
async def test_build_current_map_hashes_all_sources(tmp_path: Path):
    paths = []
    for name in ("a.txt", "b.txt", "c.txt"):
        p = tmp_path / name
        p.write_text(name, encoding="utf-8")
        paths.append(p)

    current = await build_current_map(paths, lambda p: p.name, lambda p: f"h::{p.name}", concurrency=2)
    assert current == {p.name: (p, f"h::{p.name}") for p in paths}
    assert current.errors == {}


@pytest.mark.anyio
async def test_build_current_map_isolates_per_path_hash_failures(tmp_path: Path):
    """One unreadable file must not abort hashing; it is excluded and its error captured."""
    good = tmp_path / "good.txt"
    good.write_text("g", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("b", encoding="utf-8")

    def flaky_hash(path: Path) -> str:
        if path.name == "bad.txt":
            raise PermissionError("unreadable")
        return f"h::{path.name}"

    current = await build_current_map([good, bad], lambda p: p.name, flaky_hash, concurrency=2)
    assert dict(current) == {"good.txt": (good, "h::good.txt")}
    assert set(current.errors) == {"bad.txt"}
    assert isinstance(current.errors["bad.txt"], PermissionError)


# ---------------------------------------------------------------------------
# protocol conformance
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    """Test payload for the mention conformance check."""

    title: str = Field(description="Short event title.")


def _static_protocol_conformance(
    vector_store: VectorStore, document_store: DocumentStore, mention_store: MentionStore
) -> tuple[SourceSyncStore[Chunk], SourceSyncStore[Document], SourceSyncStore[Mention[_Event]]]:
    """Static-only check (never called): the concrete store protocols satisfy SourceSyncStore.

    basedpyright verifies these returns; drift in any store protocol breaks type-check.
    """
    return vector_store, document_store, mention_store


def test_memory_vector_store_satisfies_source_sync_store():
    assert isinstance(MemoryVectorStore(), SourceSyncStore)
