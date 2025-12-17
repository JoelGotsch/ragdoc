"""Tests for MentionStorePipeline — per-source sync, hash gate, orphans, durability, plan/apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.extraction.structured import StructuredExtractor
from ragdoc.pipeline.changeset import ChangeSet
from ragdoc.pipeline.linear import DocumentPipeline

from .conftest import Event, MemoryMentionStore, make_event_client, make_extractor, make_parser


def make_pipeline(extractor, store) -> MentionStorePipeline[Event]:
    return MentionStorePipeline(
        pipeline=DocumentPipeline(parser=make_parser()),
        extractor=extractor,
        mention_store=store,
    )


@pytest.mark.anyio
async def test_run_extracts_and_stores_mentions(make_files, mstore):
    paths = make_files({"a.txt": "alpha", "b.txt": "beta"})
    result = await make_pipeline(make_extractor([Event(title="Ev")]), mstore).run(paths)

    assert set(result.processed) == {"a.txt", "b.txt"}
    assert {m.payload.title for m in await mstore.list_mentions()} == {"Ev"}
    # emits mentions, not chunks
    assert all(isinstance(m, Mention) for m in await mstore.list_mentions())


@pytest.mark.anyio
async def test_unchanged_source_skipped_no_llm_call(make_files, mstore):
    paths = make_files({"a.txt": "alpha"})
    client = make_event_client([[Event(title="Ev")]])
    extractor = StructuredExtractor(Event, client=client, model="m")
    pipeline = make_pipeline(extractor, mstore)

    await pipeline.run(paths)
    calls_after_first = client.beta.chat.completions.parse.call_count
    assert calls_after_first >= 1

    result = await pipeline.run(paths)  # nothing changed
    assert result.skipped == ["a.txt"] and result.processed == []
    # the hash gate prevented any further extraction call
    assert client.beta.chat.completions.parse.call_count == calls_after_first


@pytest.mark.anyio
async def test_changed_source_replaces_mentions(tmp_path, mstore):
    p = tmp_path / "a.txt"
    p.write_text("v1", encoding="utf-8")
    # first run yields one event; second run (file changed) yields a different one
    client = make_event_client([[Event(title="First")], [Event(title="Second")]])
    extractor = StructuredExtractor(Event, client=client, model="m")
    pipeline = make_pipeline(extractor, mstore)

    await pipeline.run([p])
    p.write_text("v2-changed", encoding="utf-8")
    result = await pipeline.run([p])

    assert result.processed == ["a.txt"]
    titles = [m.payload.title for m in mstore.by_source("a.txt")]
    assert titles == ["Second"]  # old mention replaced, no duplicate


@pytest.mark.anyio
async def test_uniform_content_hash_per_source(make_files, mstore):
    paths = make_files({"a.txt": "alpha alpha alpha"})
    await make_pipeline(make_extractor([Event(title="X"), Event(title="Y")]), mstore).run(paths)
    hashes = {m.content_hash for m in mstore.by_source("a.txt")}
    assert len(hashes) == 1  # all mentions of one source share one content_hash


@pytest.mark.anyio
async def test_orphan_handling(make_files, mstore):
    a = make_files({"a.txt": "alpha"})
    b = make_files({"b.txt": "beta"})
    pipeline = make_pipeline(make_extractor([Event(title="Ev")]), mstore)
    await pipeline.run(a + b)

    # default: orphans untouched
    res_default = await pipeline.run(a)
    assert "b.txt" in await mstore.list_source_ids()
    assert res_default.deleted == []

    # delete_orphans with the complete corpus = {a} removes b
    res = await pipeline.run(a, delete_orphans=True)
    assert res.deleted == ["b.txt"]
    assert await mstore.list_source_ids() == {"a.txt"}


@pytest.mark.anyio
async def test_collision_raises(tmp_path, mstore):
    # two files in different dirs with the same name collide under the default source_id_fn (p.name)
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "x.txt").write_text("a", encoding="utf-8")
    (d2 / "x.txt").write_text("b", encoding="utf-8")
    pipeline = make_pipeline(make_extractor([Event(title="Ev")]), mstore)
    with pytest.raises(ValueError, match="duplicate source_ids"):
        await pipeline.run([d1 / "x.txt", d2 / "x.txt"])


@pytest.mark.anyio
async def test_per_source_durability(make_files):
    """One bad source is collected in errors; the good source is still written."""

    class FlakyStore(MemoryMentionStore):
        async def upsert(self, mentions):  # type: ignore[no-untyped-def]
            if any(m.source_id == "bad.txt" for m in mentions):
                raise RuntimeError("boom")
            return await super().upsert(mentions)

    store = FlakyStore()
    paths = make_files({"good.txt": "g", "bad.txt": "b"})
    result = await make_pipeline(make_extractor([Event(title="Ev")]), store).run(paths)

    assert result.processed == ["good.txt"] or "good.txt" in result.processed
    assert [sid for sid, _ in result.errors] == ["bad.txt"]
    assert await store.list_source_ids() == {"good.txt"}


@pytest.mark.anyio
async def test_chunker_rejected_at_construction(mstore):
    from ragdoc.chunking import LLMChunker

    with pytest.raises(ValueError, match="not chunks"):
        MentionStorePipeline(
            pipeline=DocumentPipeline(parser=make_parser(), chunker=LLMChunker()),
            extractor=make_extractor([Event(title="Ev")]),
            mention_store=mstore,
        )


@pytest.mark.anyio
async def test_plan_apply_round_trip(make_files, mstore, tmp_path):
    paths = make_files({"a.txt": "alpha"})
    pipeline = make_pipeline(make_extractor([Event(title="Ev")]), mstore)

    changeset = await pipeline.plan(paths)
    assert [sc.source_id for sc in changeset.to_add] == ["a.txt"]
    assert mstore.stored == {}  # plan did not touch the store

    # serialize → reload on the concrete type → apply
    cs_path = tmp_path / "cs.json"
    changeset.save(cs_path)
    reloaded = ChangeSet[Mention[Event]].load(cs_path)
    result = await pipeline.apply(reloaded)

    assert result.processed == ["a.txt"]
    assert {m.payload.title for m in await mstore.list_mentions()} == {"Ev"}


@pytest.mark.anyio
async def test_roundtrip_never_touches_document_metadata(make_files, mstore, monkeypatch):
    """The Phase-4 headline regression: extraction is a typed channel — no 'mentions' key ever
    appears in the parent's or any split's metadata during a full pipeline run."""
    paths = make_files({"a.txt": "alpha body"})
    doc_pipeline = DocumentPipeline(parser=make_parser())

    captured_parents: list[Document] = []
    original_parse_and_process = doc_pipeline.parse_and_process

    async def spy_parse_and_process(path: Path) -> Document | None:
        parent = await original_parse_and_process(path)
        if parent is not None:
            captured_parents.append(parent)
        return parent

    monkeypatch.setattr(doc_pipeline, "parse_and_process", spy_parse_and_process)

    captured_splits: list[Document] = []

    def spy_splitter(document: Document) -> list[Document]:
        splits = [document]
        captured_splits.extend(splits)
        return splits

    pipeline = MentionStorePipeline(
        pipeline=doc_pipeline,
        extractor=make_extractor([Event(title="Ev")]),
        mention_store=mstore,
        splitter=spy_splitter,
    )
    result = await pipeline.run(paths)
    assert result.processed == ["a.txt"]
    assert captured_parents and captured_splits

    for doc in captured_parents + captured_splits:
        assert "mentions" not in doc.metadata
        assert "kg_mentions" not in doc.metadata


@pytest.mark.anyio
async def test_mentions_arrive_typed_without_revalidation(make_files, mstore):
    """The store receives the same Mention objects the extractor returned — no
    serialize→model_validate round trip in between."""
    returned: list[Mention[Event]] = []

    class TrackingExtractor:
        payload_model = Event

        async def extract(self, document: Document) -> list[Mention[Event]]:
            mention = Mention[Event](
                mention_id="pending",
                source_id=document.source_id or "",
                source_hash=document.source_hash or "",
                payload=Event(title="Tracked"),
            )
            returned.append(mention)
            return [mention]

    paths = make_files({"a.txt": "alpha"})
    result = await make_pipeline(TrackingExtractor(), mstore).run(paths)

    assert result.processed == ["a.txt"]
    stored = list(mstore.stored.values())
    assert len(stored) == len(returned) == 1
    assert stored[0] is returned[0]  # identity, not a re-validated copy
