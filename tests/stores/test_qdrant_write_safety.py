"""Write-safety tests against a real in-process Qdrant (``AsyncQdrantClient(":memory:")``).

No MagicMock call-shape assertions: every test checks what is actually **stored** after a
failed, cancelled, or crashed write, and what the next sync then does. Covers

- rollback never deletes pre-existing points (single- and multi-batch);
- an interrupted multi-batch write (error / cancellation / hard crash) never reads as
  "unchanged" — the next ``run()`` re-processes the source;
- ``replace_source`` keeps the old points when the new write fails and drops only stale ones;
- the shared base behaves the same for the vector, mention, and entity stores;
- one sync pipeline end-to-end: add → unchanged ⇒ skipped → changed ⇒ updated → orphan
  deletion (opt-in).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("qdrant_client", reason="qdrant extra not installed")

from qdrant_client import AsyncQdrantClient

from ragdoc.chunking import Chunk
from ragdoc.document import Document, Paragraph
from ragdoc.extraction.entity import Entity
from ragdoc.extraction.mention import Mention
from ragdoc.integrations.entity_stores.qdrant import QdrantEntityStore
from ragdoc.integrations.mention_stores.qdrant import QdrantMentionStore
from ragdoc.integrations.vector_stores.qdrant import QdrantVectorStore
from ragdoc.pipeline import DocumentPipeline, EmbedderConfig, VectorStorePipeline
from ragdoc.pipeline.changeset import SourceChange
from ragdoc.pipeline.sync import SourceReplaceStore, SyncEngine, SyncPlanInput, SyncSource, source_hash_token

BATCH = QdrantVectorStore._UPSERT_BATCH
BIG = 2 * BATCH + 50  # three batches


def mk_chunk(i: int, sid: str, source_hash: str, *, version: str | None = None) -> Chunk:
    """A chunk whose id depends on ``version`` (defaults to the hash ⇒ new ids per version)."""
    return Chunk(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sid}-{i}-{version or source_hash}")),
        source_id=sid,
        source_hash=source_hash,
        content_hash="c-" + source_hash,
        prompt_content=f"p{i}-{source_hash}",
        embedding_content=f"e{i}",
        named_embeddings={"dense": [0.1, 0.2, 0.3]},
    )


async def count(client: AsyncQdrantClient, collection: str) -> int:
    return (await client.count(collection)).count


async def stored_hashes(client: AsyncQdrantClient, collection: str) -> set[tuple[str | None, str | None]]:
    """Distinct ``(source_hash, content_hash)`` pairs actually stored in *collection*."""
    points, _ = await client.scroll(collection, limit=10_000, with_payload=["source_hash", "content_hash"])
    return {((p.payload or {}).get("source_hash"), (p.payload or {}).get("content_hash")) for p in points}


class _HardCrash(BaseException):
    """Stands in for a process death: not an ``Exception``, so no cleanup code may rely on it."""


def fail_on_call(
    client: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch, nth: int, make_exc: Callable[[], BaseException]
) -> None:
    """Make the *nth* ``client.upsert`` call raise (1-based); earlier/later calls hit the real store."""
    real_upsert = client.upsert
    calls = 0

    async def _upsert(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == nth:
            raise make_exc()
        return await real_upsert(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client, "upsert", _upsert)


@pytest.fixture
async def client() -> AsyncQdrantClient:
    return AsyncQdrantClient(":memory:")


@pytest.fixture
async def store(client: AsyncQdrantClient) -> QdrantVectorStore:
    return await QdrantVectorStore.create(client, "chunks", vector_size=3)


# ---------------------------------------------------------------------------
# Finding 2: rollback must never delete pre-existing data
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_failed_single_upsert_keeps_the_sources_existing_chunks(store, client, monkeypatch):
    """The reviewer's repro: 5 stored chunks, one more upsert fails before writing anything."""
    await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(5)])
    fail_on_call(client, monkeypatch, 1, lambda: ConnectionError("transient blip; nothing was written"))

    with pytest.raises(ConnectionError):
        await store.upsert([mk_chunk(99, "src-a", "h1")])

    assert await count(client, "chunks") == 5
    assert (await store.list_source_state())["src-a"].source_hash == "h1"


@pytest.mark.anyio
async def test_failed_multi_batch_upsert_removes_only_what_it_wrote(store, client, monkeypatch):
    old = [mk_chunk(i, "src-a", "h1") for i in range(5)]
    await store.upsert(old)
    fail_on_call(client, monkeypatch, 2, lambda: ConnectionError("batch 2 down"))

    with pytest.raises(ConnectionError):
        await store.upsert([mk_chunk(i, "src-a", "h2") for i in range(BIG)])

    points, _ = await client.scroll("chunks", limit=1000, with_payload=False)
    assert {str(p.id) for p in points} == {c.id for c in old}  # batch 1 undone, old chunks untouched
    assert (await store.list_source_state())["src-a"].source_hash == "h1"


@pytest.mark.anyio
async def test_rollback_does_not_delete_overwritten_preexisting_points(store, client, monkeypatch):
    """Ids that existed before the call survive the rollback; they read as 'changed' (documented limit:
    the overwritten points keep the new content, they are not restored)."""
    await store.upsert([mk_chunk(i, "src-a", "h1", version="stable") for i in range(5)])
    fail_on_call(client, monkeypatch, 2, lambda: ConnectionError("batch 2 down"))

    with pytest.raises(ConnectionError):
        await store.upsert([mk_chunk(i, "src-a", "h2", version="stable") for i in range(BIG)])

    assert await count(client, "chunks") == 5  # same five ids: overwritten, not deleted
    state = (await store.list_source_state())["src-a"]
    assert source_hash_token(state) is None  # blank ⇒ always changed, never "unchanged at h2"


@pytest.mark.anyio
async def test_failed_upsert_leaves_other_sources_alone(store, client, monkeypatch):
    await store.upsert([mk_chunk(i, "src-b", "hb") for i in range(3)])
    fail_on_call(client, monkeypatch, 3, lambda: RuntimeError("batch 3 down"))

    with pytest.raises(RuntimeError):
        await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(BIG)])

    assert await store.list_source_ids() == {"src-b"}
    assert await count(client, "chunks") == 3


# ---------------------------------------------------------------------------
# Finding 1: an interrupted multi-batch write never looks "unchanged"
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_multi_batch_upsert_stamps_real_hashes_on_every_point(store, client):
    await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(BIG)])

    assert await count(client, "chunks") == BIG
    assert await stored_hashes(client, "chunks") == {("h1", "c-h1")}
    state = (await store.list_source_state())["src-a"]
    assert (state.source_hash, state.content_hash) == ("h1", "c-h1")


@pytest.mark.anyio
async def test_multi_batch_upsert_spanning_sources_stamps_each_source_its_own_hashes(store, client):
    chunks = [mk_chunk(i, "src-a", "ha") for i in range(BATCH)] + [mk_chunk(i, "src-b", "hb") for i in range(BATCH)]
    await store.upsert(chunks)

    state = await store.list_source_state()
    assert state["src-a"].source_hash == "ha" and state["src-b"].source_hash == "hb"


def _big_engine(store: QdrantVectorStore, produced: list[str]) -> tuple[SyncEngine[Chunk], SyncPlanInput]:
    async def producer(src: SyncSource) -> SourceChange[Chunk]:
        produced.append(src.source_id)
        chunks = [mk_chunk(i, src.source_id, src.change_token) for i in range(BIG)]
        return SourceChange(source_id=src.source_id, source_hash=src.change_token, items=chunks)

    engine: SyncEngine[Chunk] = SyncEngine(store=store, producer=producer, token_of=source_hash_token)
    request = SyncPlanInput(sources=[SyncSource("big.pdf", "HASH-V1")], live_source_ids=frozenset({"big.pdf"}))
    return engine, request


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("make_exc", "left_behind"),
    [
        pytest.param(asyncio.CancelledError, 0, id="cancelled-cleanup-runs"),
        pytest.param(_HardCrash, BATCH, id="hard-crash-no-cleanup"),
    ],
)
async def test_interrupted_multi_batch_write_is_reprocessed_by_the_next_run(
    store, client, monkeypatch, make_exc, left_behind
):
    """The reviewer's repro: stop after batch 1 of 3; the next run() must NOT skip the source."""
    produced: list[str] = []
    engine, request = _big_engine(store, produced)
    fail_on_call(client, monkeypatch, 2, make_exc)

    with pytest.raises(make_exc):
        await engine.run(request)
    monkeypatch.undo()

    assert await count(client, "chunks") == left_behind
    # Whatever is left carries blank hashes — nothing claims to be the new version.
    assert await stored_hashes(client, "chunks") <= {(None, None)}

    result = await engine.run(request)

    assert result.processed == ["big.pdf"] and result.skipped == []
    assert await count(client, "chunks") == BIG
    assert await stored_hashes(client, "chunks") == {("HASH-V1", "c-HASH-V1")}
    # ...and only now is it unchanged.
    assert (await engine.run(request)).skipped == ["big.pdf"]
    assert produced == ["big.pdf", "big.pdf"]


@pytest.mark.anyio
async def test_crash_before_the_hash_stamp_is_reprocessed(store, client, monkeypatch):
    """All batches landed but the process died before the commit marker: still 'changed'."""
    produced: list[str] = []
    engine, request = _big_engine(store, produced)

    async def _crash(**kwargs: object) -> None:
        raise _HardCrash

    monkeypatch.setattr(client, "set_payload", _crash)
    with pytest.raises(_HardCrash):
        await engine.run(request)
    monkeypatch.undo()

    assert await count(client, "chunks") == BIG
    assert source_hash_token((await store.list_source_state())["big.pdf"]) is None
    assert (await engine.run(request)).processed == ["big.pdf"]
    assert await count(client, "chunks") == BIG


@pytest.mark.anyio
async def test_real_task_cancellation_mid_write_cleans_up_and_propagates(store, client, monkeypatch):
    """Cancel the running task (Ctrl-C / timeout shape) while batch 2 is in flight."""
    in_batch_two = asyncio.Event()
    real_upsert = client.upsert
    calls = 0

    async def _upsert(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            in_batch_two.set()
            await asyncio.sleep(3600)
        return await real_upsert(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client, "upsert", _upsert)
    task = asyncio.create_task(store.upsert([mk_chunk(i, "src-a", "h1") for i in range(BIG)]))
    await in_batch_two.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert await count(client, "chunks") == 0


# ---------------------------------------------------------------------------
# list_source_state: mixed sources read as changed
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_source_with_mixed_old_and_new_points_reads_as_changed(store):
    await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(3)])
    await store.upsert([mk_chunk(i, "src-a", "h2") for i in range(3)])  # direct upsert, old ones still there

    state = (await store.list_source_state())["src-a"]

    assert state.source_hash == "" and state.content_hash is None


# ---------------------------------------------------------------------------
# Finding 5: replace_source — new before old
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_vector_store_offers_replace_source(store):
    assert isinstance(store, SourceReplaceStore)


@pytest.mark.anyio
@pytest.mark.parametrize("n_new", [4, BIG], ids=["single-batch", "multi-batch"])
async def test_replace_source_swaps_in_new_chunks_and_drops_only_stale_ones(store, client, n_new):
    await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(7)])
    await store.upsert([mk_chunk(i, "src-b", "hb") for i in range(2)])
    new = [mk_chunk(i, "src-a", "h2") for i in range(n_new)]

    assert await store.replace_source("src-a", new) == [c.id for c in new]

    assert await count(client, "chunks") == n_new + 2
    state = await store.list_source_state()
    assert state["src-a"].source_hash == "h2" and state["src-b"].source_hash == "hb"


@pytest.mark.anyio
@pytest.mark.parametrize(("n_new", "failing_call"), [(4, 1), (BIG, 2)], ids=["single-batch", "multi-batch"])
async def test_replace_source_failure_keeps_the_old_chunks(store, client, monkeypatch, n_new, failing_call):
    old = [mk_chunk(i, "src-a", "h1") for i in range(7)]
    await store.upsert(old)
    fail_on_call(client, monkeypatch, failing_call, lambda: ConnectionError("down"))

    with pytest.raises(ConnectionError):
        await store.replace_source("src-a", [mk_chunk(i, "src-a", "h2") for i in range(n_new)])

    points, _ = await client.scroll("chunks", limit=1000, with_payload=False)
    assert {str(p.id) for p in points} == {c.id for c in old}
    assert (await store.list_source_state())["src-a"].source_hash == "h1"


@pytest.mark.anyio
async def test_replace_source_crash_between_write_and_stale_delete_reads_as_changed(store, client, monkeypatch):
    await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(3)])

    async def _crash(**kwargs: object) -> None:
        raise _HardCrash

    monkeypatch.setattr(client, "delete", _crash)
    with pytest.raises(_HardCrash):
        await store.replace_source("src-a", [mk_chunk(i, "src-a", "h2") for i in range(3)])
    monkeypatch.undo()

    assert await count(client, "chunks") == 6  # old + new, mixed
    assert source_hash_token((await store.list_source_state())["src-a"]) is None


@pytest.mark.anyio
async def test_replace_source_rejects_chunks_of_another_source(store):
    with pytest.raises(ValueError, match="other sources"):
        await store.replace_source("src-a", [mk_chunk(0, "src-b", "h")])


@pytest.mark.anyio
async def test_replace_source_with_no_chunks_deletes_the_source(store, client):
    await store.upsert([mk_chunk(i, "src-a", "h1") for i in range(3)])
    assert await store.replace_source("src-a", []) == []
    assert await count(client, "chunks") == 0


# ---------------------------------------------------------------------------
# The same guarantees on the other stores built on the shared base
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    """Minimal payload for mention / entity stores."""

    title: str = Field(description="Event title.")


async def _embed(payloads: list[BaseModel]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3] for _ in payloads]


def mk_mention(i: int, sid: str, content_hash: str) -> Mention[_Event]:
    return Mention[_Event](
        mention_id=f"{sid}-{content_hash}-{i}",
        source_id=sid,
        source_hash="f-" + content_hash,
        content_hash=content_hash,
        ordinal=i,
        payload=_Event(title=f"E{i}"),
    )


@pytest.fixture
async def mention_store(client: AsyncQdrantClient) -> QdrantMentionStore:
    return await QdrantMentionStore.create(client, "mentions", _Event, _embed, vector_size=3)


@pytest.mark.anyio
async def test_mention_store_multi_batch_roundtrip_keeps_hashes(mention_store, client):
    await mention_store.upsert([mk_mention(i, "s1", "c1") for i in range(BIG)])

    state = (await mention_store.list_source_state())["s1"]
    assert (state.source_hash, state.content_hash) == ("f-c1", "c1")
    mentions = await mention_store.list_mentions()
    assert len(mentions) == BIG and {m.content_hash for m in mentions} == {"c1"}


@pytest.mark.anyio
async def test_mention_store_interrupted_write_reads_as_changed(mention_store, client, monkeypatch):
    fail_on_call(client, monkeypatch, 2, _HardCrash)
    with pytest.raises(_HardCrash):
        await mention_store.upsert([mk_mention(i, "s1", "c1") for i in range(BIG)])
    monkeypatch.undo()

    assert await count(client, "mentions") == BATCH
    state = (await mention_store.list_source_state())["s1"]
    assert state.source_hash == "" and state.content_hash is None


@pytest.mark.anyio
async def test_mention_store_replace_source_failure_keeps_old_mentions(mention_store, client, monkeypatch):
    assert isinstance(mention_store, SourceReplaceStore)
    await mention_store.upsert([mk_mention(i, "s1", "c1") for i in range(4)])
    fail_on_call(client, monkeypatch, 2, lambda: ConnectionError("down"))

    with pytest.raises(ConnectionError):
        await mention_store.replace_source("s1", [mk_mention(i, "s1", "c2") for i in range(BIG)])
    monkeypatch.undo()

    assert {m.content_hash for m in await mention_store.list_mentions()} == {"c1"}
    await mention_store.replace_source("s1", [mk_mention(i, "s1", "c2") for i in range(BIG)])
    mentions = await mention_store.list_mentions()
    assert len(mentions) == BIG and {m.content_hash for m in mentions} == {"c2"}


def mk_entity(i: int, title: str) -> Entity[_Event]:
    return Entity[_Event](
        entity_id=f"ent-{i}", payload=_Event(title=title), member_mention_ids=[f"m{i}"], source_ids=["s1"]
    )


@pytest.mark.anyio
async def test_entity_store_failed_multi_batch_upsert_keeps_existing_entities(client, monkeypatch):
    """Entity points carry no change-detection hashes: written as-is, and a failed write
    must not delete entities that existed before it (the old delete-by-source bug class)."""
    entities = await QdrantEntityStore.create(client, "entities", _Event)
    await entities.upsert([mk_entity(i, "old") for i in range(10)])
    fail_on_call(client, monkeypatch, 2, lambda: ConnectionError("down"))

    with pytest.raises(ConnectionError):
        await entities.upsert([mk_entity(i, "new") for i in range(BIG)])
    monkeypatch.undo()

    assert await count(client, "entities") == 10  # the ten pre-existing ids; new ones rolled back
    await entities.upsert([mk_entity(i, "new") for i in range(BIG)])
    assert await count(client, "entities") == BIG


# ---------------------------------------------------------------------------
# End-to-end: one sync pipeline against the real in-memory store
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


async def _parse(path: Path) -> Document:
    return Document(elements=[Paragraph(html=f"<p>{path.read_text(encoding='utf-8')}</p>")])


def _pipeline(store: QdrantVectorStore) -> VectorStorePipeline:
    return VectorStorePipeline(
        DocumentPipeline(parser=_parse), store, embedders={"dense": EmbedderConfig(_FakeEmbedder())}
    )


async def _prompts(client: AsyncQdrantClient) -> dict[str, str]:
    points, _ = await client.scroll("chunks", limit=100, with_payload=True)
    return {str((p.payload or {})["source_id"]): str((p.payload or {})["prompt_content"]) for p in points}


@pytest.mark.anyio
async def test_vector_pipeline_end_to_end_add_skip_update_orphan(store, client, tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("alpha one", encoding="utf-8")
    b.write_text("beta one", encoding="utf-8")
    vsp = _pipeline(store)

    first = await vsp.run([a, b])
    assert sorted(first.processed) == ["a.txt", "b.txt"] and first.errors == []
    assert await store.list_source_ids() == {"a.txt", "b.txt"}

    second = await vsp.run([a, b])
    assert sorted(second.skipped) == ["a.txt", "b.txt"] and second.processed == []

    a.write_text("alpha two", encoding="utf-8")
    third = await vsp.run([a, b])
    assert third.processed == ["a.txt"] and third.skipped == ["b.txt"]
    prompts = await _prompts(client)
    assert "alpha two" in prompts["a.txt"] and "beta one" in prompts["b.txt"]
    assert await count(client, "chunks") == 2  # the stale a.txt chunk is gone

    kept = await vsp.run([a])  # b.txt absent, orphan deletion is opt-in
    assert kept.deleted == [] and await store.list_source_ids() == {"a.txt", "b.txt"}

    pruned = await vsp.run([a], delete_orphans=True)
    assert pruned.deleted == ["b.txt"] and pruned.skipped == ["a.txt"]
    assert await store.list_source_ids() == {"a.txt"}


@pytest.mark.anyio
async def test_vector_pipeline_plan_apply_matches_run_on_the_real_store(store, client, tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("alpha one", encoding="utf-8")
    vsp = _pipeline(store)
    await vsp.run([a])
    a.write_text("alpha two", encoding="utf-8")

    changeset = await vsp.plan([a])
    assert [sc.source_id for sc in changeset.to_update] == ["a.txt"] and changeset.to_add == []
    assert "alpha one" in (await _prompts(client))["a.txt"]  # plan() wrote nothing

    result = await vsp.apply(changeset)

    assert result.processed == ["a.txt"]
    assert "alpha two" in (await _prompts(client))["a.txt"] and await count(client, "chunks") == 1
    assert (await vsp.run([a])).skipped == ["a.txt"]


@pytest.mark.anyio
async def test_vector_pipeline_failed_update_keeps_serving_the_old_chunks(store, client, tmp_path, monkeypatch):
    """A changed source whose write fails keeps its previous chunks and is retried next run."""
    a = tmp_path / "a.txt"
    a.write_text("alpha one", encoding="utf-8")
    vsp = _pipeline(store)
    await vsp.run([a])
    a.write_text("alpha two", encoding="utf-8")
    fail_on_call(client, monkeypatch, 1, lambda: ConnectionError("down"))

    failed = await vsp.run([a])
    monkeypatch.undo()

    assert [sid for sid, _ in failed.errors] == ["a.txt"] and failed.processed == []
    assert "alpha one" in (await _prompts(client))["a.txt"]
    assert (await vsp.run([a])).processed == ["a.txt"]
    assert "alpha two" in (await _prompts(client))["a.txt"]
