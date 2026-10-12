"""Robustness tests for the generic SyncEngine: memory, back-pressure, and failure injection.

- ``run()`` releases a source's items after its write and back-pressures producers
  (peak memory is bounded by ``concurrency``, not by corpus size);
- the engine's concurrency cap;
- a failed write: delete-then-upsert stores are left empty (documented), stores offering
  ``replace_source`` keep the old items;
- ``pre_write`` (embed) failure inside ``run()``, orphan-delete failure;
- a stored ``content_hash`` of ``None`` is treated as changed.
"""

from __future__ import annotations

import asyncio
import gc
import weakref

import pytest
from pydantic import BaseModel, Field

from ragdoc.pipeline.changeset import ChangeSet, SourceChange
from ragdoc.pipeline.stores import SourceState
from ragdoc.pipeline.sync import (
    SourceReplaceStore,
    SyncEngine,
    SyncPlanInput,
    SyncSource,
    content_hash_token,
    source_hash_token,
)


class Item(BaseModel):
    """Minimal sync payload."""

    source_id: str = Field(description="Sync identity key.")
    source_hash: str = Field(description="Change token stored for this item's source.")
    marker: str = Field(description="Content marker to tell versions apart.")


class StubStore:
    """In-memory delete-then-upsert sink (no ``replace_source``) with failure injection."""

    def __init__(self) -> None:
        self.items: dict[str, list[Item]] = {}
        self.events: list[str] = []
        self.fail_upsert: set[str] = set()
        self.fail_delete: set[str] = set()
        self.content_hashes: dict[str, str | None] = {}

    async def upsert(self, items: list[Item]) -> list[str]:
        if items and items[0].source_id in self.fail_upsert:
            raise ConnectionError(f"upsert down for {items[0].source_id}")
        for item in items:
            self.items.setdefault(item.source_id, []).append(item)
        self.events.append(f"upsert:{items[0].source_id}")
        return [item.marker for item in items]

    async def delete_by_source(self, source_id: str) -> None:
        if source_id in self.fail_delete:
            raise ConnectionError(f"delete down for {source_id}")
        self.items.pop(source_id, None)
        self.events.append(f"delete:{source_id}")

    async def list_source_state(self) -> dict[str, SourceState]:
        return {
            sid: SourceState(source_hash=items[0].source_hash, content_hash=self.content_hashes.get(sid))
            for sid, items in self.items.items()
            if items
        }

    def markers(self, source_id: str) -> list[str]:
        return [item.marker for item in self.items.get(source_id, [])]


class ReplacingStore(StubStore):
    """A sink that can swap a source's items new-before-old (implements SourceReplaceStore)."""

    async def replace_source(self, source_id: str, items: list[Item]) -> list[str]:
        if source_id in self.fail_upsert:
            raise ConnectionError(f"replace down for {source_id}")
        self.items[source_id] = list(items)
        self.events.append(f"replace:{source_id}")
        return [item.marker for item in items]


def request_for(*sids: str, token: str = "v1") -> SyncPlanInput:
    return SyncPlanInput(
        sources=[SyncSource(source_id=sid, change_token=token) for sid in sids],
        live_source_ids=frozenset(sids),
    )


async def produce(src: SyncSource) -> SourceChange[Item]:
    item = Item(source_id=src.source_id, source_hash=src.change_token, marker=f"{src.source_id}@{src.change_token}")
    return SourceChange(source_id=src.source_id, source_hash=src.change_token, items=[item])


def engine_for(store: StubStore, concurrency: int = 10) -> SyncEngine[Item]:
    return SyncEngine(store=store, producer=produce, token_of=source_hash_token, concurrency=concurrency)


# ---------------------------------------------------------------------------
# Finding 4: memory + back-pressure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_releases_written_items_so_memory_does_not_grow_with_the_corpus():
    """The reviewer's repro: count the produced items still alive at each write."""
    refs: list[weakref.ref[Item]] = []
    alive_at_write: list[int] = []
    concurrency, n = 2, 50

    class CountingStore(StubStore):
        async def upsert(self, items: list[Item]) -> list[str]:
            gc.collect()
            alive_at_write.append(sum(1 for r in refs if r() is not None))
            return []  # store nothing: only the engine could be keeping items alive

    async def producer(src: SyncSource) -> SourceChange[Item]:
        change = await produce(src)
        refs.append(weakref.ref(change.items[0]))
        return change

    engine: SyncEngine[Item] = SyncEngine(
        store=CountingStore(), producer=producer, token_of=source_hash_token, concurrency=concurrency
    )
    result = await engine.run(request_for(*(f"s{i}" for i in range(n))))

    assert len(result.processed) == n
    # one per producer slot + one queued + the one being written — never the whole corpus
    assert max(alive_at_write) <= concurrency + 2


@pytest.mark.anyio
async def test_run_back_pressures_producers_while_the_writer_is_busy():
    """With the writer stuck on source #1, production stalls instead of racing through the corpus."""
    concurrency, n = 2, 40
    started = 0
    writer_blocked = asyncio.Event()
    release_writer = asyncio.Event()

    class SlowStore(StubStore):
        async def upsert(self, items: list[Item]) -> list[str]:
            writer_blocked.set()
            await release_writer.wait()
            return await super().upsert(items)

    async def producer(src: SyncSource) -> SourceChange[Item]:
        nonlocal started
        started += 1
        return await produce(src)

    store = SlowStore()
    engine: SyncEngine[Item] = SyncEngine(
        store=store, producer=producer, token_of=source_hash_token, concurrency=concurrency
    )
    run = asyncio.create_task(engine.run(request_for(*(f"s{i}" for i in range(n)))))
    await writer_blocked.wait()
    for _ in range(50):  # let every runnable producer make all the progress it can
        await asyncio.sleep(0)

    assert started <= concurrency + 2, f"{started} of {n} sources were produced while the writer was blocked"

    release_writer.set()
    result = await run
    assert len(result.processed) == n and started == n


@pytest.mark.anyio
async def test_engine_never_exceeds_its_concurrency_cap():
    concurrency, n = 3, 20
    running = 0
    peak = 0

    async def producer(src: SyncSource) -> SourceChange[Item]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.001)
        running -= 1
        return await produce(src)

    engine: SyncEngine[Item] = SyncEngine(
        store=StubStore(), producer=producer, token_of=source_hash_token, concurrency=concurrency
    )
    result = await engine.run(request_for(*(f"s{i}" for i in range(n))))

    assert len(result.processed) == n
    assert peak == concurrency


@pytest.mark.anyio
async def test_shared_semaphore_caps_concurrency_across_two_engines():
    shared = asyncio.Semaphore(2)
    running = 0
    peak = 0

    async def producer(src: SyncSource) -> SourceChange[Item]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.001)
        running -= 1
        return await produce(src)

    def _engine() -> SyncEngine[Item]:
        return SyncEngine(store=StubStore(), producer=producer, token_of=source_hash_token, concurrency=shared)

    await asyncio.gather(_engine().run(request_for("a", "b", "c", "d")), _engine().run(request_for("e", "f", "g")))

    assert peak == 2


@pytest.mark.anyio
async def test_writer_failure_mid_stream_cancels_blocked_producers():
    """A fatal error in the consumer must not leave producers parked on the hand-off queue."""

    class _Fatal(BaseException):
        pass

    class DyingStore(StubStore):
        async def upsert(self, items: list[Item]) -> list[str]:
            raise _Fatal

    before = len(asyncio.all_tasks())
    with pytest.raises(_Fatal):
        await engine_for(DyingStore(), concurrency=2).run(request_for(*(f"s{i}" for i in range(10))))
    await asyncio.sleep(0)

    assert len(asyncio.all_tasks()) == before


# ---------------------------------------------------------------------------
# Finding 5: an upsert that fails after the stale delete
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_then_upsert_store_is_left_empty_when_upsert_fails_then_recovers():
    """Documented limitation of stores without replace_source: the failed update leaves the
    source EMPTY (not 'unapplied'); the next run sees it as new and re-processes it."""
    store = StubStore()
    engine = engine_for(store)
    await engine.run(request_for("a", token="v1"))
    store.fail_upsert.add("a")

    failed = await engine.run(request_for("a", token="v2"))

    assert [sid for sid, _ in failed.errors] == ["a"] and failed.processed == []
    assert store.markers("a") == []  # the delete happened, the upsert did not

    store.fail_upsert.clear()
    recovered = await engine.run(request_for("a", token="v2"))
    assert recovered.processed == ["a"] and store.markers("a") == ["a@v2"]


@pytest.mark.anyio
async def test_replace_capable_store_keeps_old_items_when_the_update_fails():
    store = ReplacingStore()
    assert isinstance(store, SourceReplaceStore)
    engine = engine_for(store)
    await engine.run(request_for("a", token="v1"))
    store.fail_upsert.add("a")

    failed = await engine.run(request_for("a", token="v2"))

    assert [sid for sid, _ in failed.errors] == ["a"]
    assert store.markers("a") == ["a@v1"]  # the write really is unapplied
    assert "delete:a" not in store.events

    store.fail_upsert.clear()
    assert (await engine.run(request_for("a", token="v2"))).processed == ["a"]
    assert store.markers("a") == ["a@v2"]


@pytest.mark.anyio
async def test_replace_capable_store_new_source_uses_upsert_and_empty_update_deletes():
    store = ReplacingStore()
    await engine_for(store).run(request_for("a", token="v1"))
    assert store.events == ["upsert:a"]

    async def nothing(src: SyncSource) -> SourceChange[Item]:
        return SourceChange(source_id=src.source_id, source_hash=src.change_token, items=[])

    engine: SyncEngine[Item] = SyncEngine(store=store, producer=nothing, token_of=source_hash_token)
    result = await engine.run(request_for("a", token="v2"))

    assert result.processed == ["a"] and store.markers("a") == []
    assert store.events == ["upsert:a", "delete:a"]


@pytest.mark.anyio
@pytest.mark.parametrize("store_type", [StubStore, ReplacingStore])
async def test_apply_upsert_failure_is_isolated_per_source(store_type: type[StubStore]):
    store = store_type()
    engine = engine_for(store)
    await engine.run(request_for("a", "b", token="v1"))
    changeset: ChangeSet[Item] = await engine.plan(request_for("a", "b", token="v2"))
    store.fail_upsert.add("a")

    result = await engine.apply(changeset)

    assert [sid for sid, _ in result.errors] == ["a"] and result.processed == ["b"]
    assert store.markers("b") == ["b@v2"]
    expected_a = ["a@v1"] if store_type is ReplacingStore else []
    assert store.markers("a") == expected_a


# ---------------------------------------------------------------------------
# pre_write (embed) failure, orphan-delete failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_embed_failure_leaves_that_source_untouched_and_others_proceed():
    store = StubStore()
    await engine_for(store).run(request_for("a", "b", token="v1"))
    store.events.clear()

    async def embed(items: list[Item]) -> None:
        if items[0].source_id == "a":
            raise RuntimeError("embedding API down")

    engine: SyncEngine[Item] = SyncEngine(store=store, producer=produce, token_of=source_hash_token, pre_write=embed)
    result = await engine.run(request_for("a", "b", token="v2"))

    assert [sid for sid, _ in result.errors] == ["a"] and isinstance(result.errors[0][1], RuntimeError)
    assert result.processed == ["b"]
    assert store.markers("a") == ["a@v1"], "embed runs before any store mutation for the source"
    assert "delete:a" not in store.events
    assert store.markers("b") == ["b@v2"]


@pytest.mark.anyio
async def test_run_orphan_delete_failure_is_recorded_and_other_orphans_still_go():
    store = StubStore()
    await engine_for(store).run(request_for("a", "gone-1", "gone-2", token="v1"))
    store.fail_delete.add("gone-1")

    result = await engine_for(store).run(request_for("a", token="v1"), delete_orphans=True)

    assert result.skipped == ["a"]
    assert [sid for sid, _ in result.errors] == ["gone-1"] and result.deleted == ["gone-2"]
    assert store.markers("gone-1") == ["gone-1@v1"] and store.markers("gone-2") == []


@pytest.mark.anyio
async def test_apply_orphan_delete_failure_is_recorded():
    store = StubStore()
    await engine_for(store).run(request_for("gone", token="v1"))
    store.fail_delete.add("gone")

    result = await engine_for(store).apply(ChangeSet[Item](to_delete=["gone"]))

    assert [sid for sid, _ in result.errors] == ["gone"] and result.deleted == []


# ---------------------------------------------------------------------------
# stored content_hash of None ⇒ changed
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stored", "current", "expect_processed"),
    [
        pytest.param(None, "c1", True, id="stored-None-is-always-changed"),
        pytest.param(None, "", True, id="stored-None-vs-blank-current-token"),
        pytest.param("", "", True, id="blank-stored-token-never-matches"),
        pytest.param("c1", "c1", False, id="equal-tokens-skip"),
        pytest.param("c1", "c2", True, id="different-tokens-change"),
    ],
)
async def test_stored_content_hash_none_or_blank_is_treated_as_changed(
    stored: str | None, current: str, expect_processed: bool
):
    store = StubStore()
    await engine_for(store).run(request_for("a", token="v1"))
    store.content_hashes["a"] = stored
    engine: SyncEngine[Item] = SyncEngine(store=store, producer=produce, token_of=content_hash_token)

    result = await engine.run(request_for("a", token=current))

    assert (result.processed == ["a"]) is expect_processed
    assert (result.skipped == ["a"]) is not expect_processed
