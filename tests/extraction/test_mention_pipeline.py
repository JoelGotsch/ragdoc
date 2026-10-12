"""Tests for MentionStorePipeline — per-source sync, hash gate, orphans, durability, plan/apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.document import Document
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.extraction.structured import StructuredExtractor
from ragdoc.pipeline.changeset import ChangeSet
from ragdoc.pipeline.linear import IngestPipeline

from .conftest import Event, MemoryMentionStore, make_event_client, make_extractor, make_parser


def make_pipeline(extractor, store) -> MentionStorePipeline[Event]:
    return MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
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
    calls_after_first = client.chat.completions.parse.call_count
    assert calls_after_first >= 1

    result = await pipeline.run(paths)  # nothing changed
    assert result.skipped == ["a.txt"] and result.processed == []
    # the hash gate prevented any further extraction call
    assert client.chat.completions.parse.call_count == calls_after_first


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


def test_chunker_unexpressible_at_construction(mstore):
    """MentionStorePipeline extracts mentions, not chunks — an IngestPipeline has no chunker
    parameter, so the old runtime rejection is now a TypeError at construction."""
    from unittest.mock import MagicMock

    from ragdoc.chunking import LLMChunker

    with pytest.raises(TypeError):
        MentionStorePipeline(
            ingest=IngestPipeline(parser=make_parser(), chunker=LLMChunker(client=MagicMock())),  # type: ignore[call-arg]
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
    ingest = IngestPipeline(parser=make_parser())

    captured_parents: list[Document] = []
    original_ingest_run = ingest.run

    async def spy_ingest_run(path: Path) -> Document | None:
        parent = await original_ingest_run(path)
        if parent is not None:
            captured_parents.append(parent)
        return parent

    monkeypatch.setattr(ingest, "run", spy_ingest_run)

    captured_splits: list[Document] = []

    def spy_splitter(document: Document) -> list[Document]:
        splits = [document]
        captured_splits.extend(splits)
        return splits

    pipeline = MentionStorePipeline(
        ingest=ingest,
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


# ---------------------------------------------------------------------------
# Multi-split provenance: running ordinals (F2), edge-ref rewrite (F1), honest source_hash (F7)
# ---------------------------------------------------------------------------


def two_way_splitter(document: Document) -> list[Document]:
    """A deterministic splitter producing two distinct splits with content markers."""
    from ragdoc.document import Paragraph

    def _split(seq: int, marker: str) -> Document:
        return Document(
            title=f"{document.title}-{seq}",
            elements=[Paragraph(html=f"<p>{marker} body of split {seq}</p>")],
            metadata={**document.metadata, "split_sequence": seq, "split_total": 2},
        )

    return [_split(1, "SPLIT-ONE"), _split(2, "SPLIT-TWO")]


@pytest.mark.anyio
async def test_multisplit_edge_refs_resolve_to_finalized_node_ids(make_files, mstore):
    """F1 regression: the pipeline re-mints node mention_ids with the PARENT content_hash;
    every KG edge payload's refs must be rewritten through the old->new map, or every edge of a
    multi-split source dangles and kg_resolution orphans it."""
    from types import SimpleNamespace
    from typing import Literal
    from unittest.mock import AsyncMock, MagicMock

    from pydantic import BaseModel

    from ragdoc.extraction.kg import KnowledgeGraphExtractor, build_graph_batch_model
    from ragdoc.extraction.schema import EdgeRef, GraphSchema

    class Person(BaseModel):
        kind: Literal["Person"] = "Person"
        full_name: str

    class Company(BaseModel):
        kind: Literal["Company"] = "Company"
        name: str

    class Employment(BaseModel):
        kind: Literal["Employment"] = "Employment"
        refs: EdgeRef

    schema = GraphSchema(
        node_types=(Person, Company),
        edge_types=(Employment,),
        patterns=((Person, Employment, Company),),
    )
    batches_by_marker = {
        "SPLIT-ONE": (
            [
                {"local_id": "n0", "node": {"kind": "Person", "full_name": "Alice"}},
                {"local_id": "n1", "node": {"kind": "Company", "name": "Acme"}},
            ],
            [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
        ),
        "SPLIT-TWO": (
            [
                {"local_id": "n0", "node": {"kind": "Person", "full_name": "Bob"}},
                {"local_id": "n1", "node": {"kind": "Company", "name": "Beta"}},
            ],
            [{"kind": "Employment", "refs": {"source_mention_id": "n0", "target_mention_id": "n1"}}],
        ),
    }
    graph_batch_cls, _ = build_graph_batch_model(schema)

    # Keyed on the rendered split content (not call order) — splits are extracted concurrently.
    async def _parse(**kwargs):  # type: ignore[no-untyped-def]
        text = kwargs["messages"][1]["content"]
        for marker, (nodes, edges) in batches_by_marker.items():
            if marker in text:
                batch = graph_batch_cls.model_validate({"node_mentions": nodes, "edge_mentions": edges})
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])
        raise AssertionError(f"no batch registered for split text: {text!r}")

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_parse)

    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=KnowledgeGraphExtractor(schema, client=client, model="m"),
        mention_store=mstore,
        splitter=two_way_splitter,
    )
    result = await pipeline.run(make_files({"a.txt": "irrelevant raw body"}))
    assert result.processed == ["a.txt"]

    stored = await mstore.list_mentions()
    node_ids = {m.mention_id for m in stored if m.payload.kind in {"Person", "Company"}}  # type: ignore[attr-defined]
    edges = [m for m in stored if m.payload.kind == "Employment"]  # type: ignore[attr-defined]
    assert len(node_ids) == 4
    assert len(edges) == 2

    # Every edge ref resolves to a finalized node mention_id — nothing dangles.
    for e in edges:
        assert e.payload.refs.source_mention_id in node_ids  # type: ignore[attr-defined]
        assert e.payload.refs.target_mention_id in node_ids  # type: ignore[attr-defined]

    # And the wiring is per-split correct: Alice→Acme, Bob→Beta.
    by_name = {
        (getattr(m.payload, "full_name", None) or getattr(m.payload, "name", None)): m.mention_id
        for m in stored
        if m.payload.kind in {"Person", "Company"}  # type: ignore[attr-defined]
    }
    pairs = {(e.payload.refs.source_mention_id, e.payload.refs.target_mention_id) for e in edges}  # type: ignore[attr-defined]
    assert pairs == {(by_name["Alice"], by_name["Acme"]), (by_name["Bob"], by_name["Beta"])}


@pytest.mark.anyio
async def test_identical_payloads_across_splits_get_distinct_mention_ids(make_files, tmp_path):
    """F2 regression: ordinals run across the whole source, not per split — identical payloads
    at split-local ordinal 0 must finalize to distinct mention_ids and both survive the store."""
    from ragdoc.extraction.mention import mint_mention_id
    from ragdoc.extraction.stores import LocalMentionStore

    class OnePerSplitExtractor:
        payload_model = Event

        async def extract(self, document: Document) -> list[Mention[Event]]:
            payload = Event(title="Same")
            return [
                Mention[Event](
                    mention_id=mint_mention_id(document.source_id or "", document.content_hash(), 0, payload),
                    source_id=document.source_id or "",
                    payload=payload,
                )
            ]

    store = LocalMentionStore(tmp_path / "mentions", Event)
    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=OnePerSplitExtractor(),
        mention_store=store,
        splitter=two_way_splitter,
    )
    result = await pipeline.run(make_files({"a.txt": "body"}))
    assert result.processed == ["a.txt"]

    stored = await store.list_mentions()
    assert len(stored) == 2  # no silent overwrite in the mention_id-keyed store
    assert len({m.mention_id for m in stored}) == 2
    assert sorted(m.ordinal for m in stored) == [0, 1]


@pytest.mark.anyio
async def test_unreadable_file_surfaces_in_errors_and_is_not_orphaned(make_files, mstore):
    """A hash_fn failure is per-source isolated: it lands in result.errors, the other source
    still syncs, and the failed source's existing mentions are never orphan-deleted."""
    import hashlib

    paths = make_files({"good.txt": "g", "bad.txt": "b"})

    def flaky_hash(p: Path) -> str:
        if p.name == "bad.txt":
            raise OSError("unreadable")
        return hashlib.sha256(p.read_bytes()).hexdigest()

    # Seed the store with a mention for bad.txt so orphan deletion has something to (not) remove.
    await mstore.upsert(
        [Mention[Event](mention_id="pre", source_id="bad.txt", source_hash="old", payload=Event(title="Old"))]
    )

    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=make_extractor([Event(title="Ev")]),
        mention_store=mstore,
        hash_fn=flaky_hash,
    )
    result = await pipeline.run(paths, delete_orphans=True)

    assert result.processed == ["good.txt"]
    assert [sid for sid, _ in result.errors] == ["bad.txt"]
    assert result.deleted == []  # unreadable is not orphaned
    assert "bad.txt" in await mstore.list_source_ids()


@pytest.mark.anyio
async def test_direct_mode_stamps_real_file_hash(make_files, mstore):
    """F7: direct mode stamps the actual file-byte hash onto every mention."""
    import hashlib

    paths = make_files({"a.txt": "alpha"})
    await make_pipeline(make_extractor([Event(title="Ev")]), mstore).run(paths)

    expected = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    mentions = await mstore.list_mentions()
    assert mentions and all(m.source_hash == expected for m in mentions)


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
                source_hash=document.source_hash,
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


# ---------------------------------------------------------------------------
# Bounded per-source fan-out (split_concurrency)
# ---------------------------------------------------------------------------


def n_way_splitter(n: int):
    def _split(document: Document) -> list[Document]:
        from ragdoc.document import Paragraph

        return [
            Document(
                title=f"{document.title}-{seq}",
                elements=[Paragraph(html=f"<p>body of split {seq}</p>")],
                metadata={**document.metadata, "split_sequence": seq, "split_total": n},
            )
            for seq in range(1, n + 1)
        ]

    return _split


class InFlightExtractor:
    """Fake extractor recording how many extract() calls overlap (max_in_flight)."""

    payload_model = Event

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def extract(self, document: Document) -> list[Mention[Event]]:
        import asyncio

        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for _ in range(5):  # several loop turns, so unbounded siblings would all overlap
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        return [Mention[Event](mention_id="pending", source_id=document.source_id or "", payload=Event(title="Ev"))]


@pytest.mark.anyio
async def test_split_extraction_is_bounded_by_split_concurrency(make_files, mstore):
    """All splits of a source used to be gathered at once (unbounded LLM fan-out)."""
    extractor = InFlightExtractor()
    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=extractor,
        mention_store=mstore,
        splitter=n_way_splitter(12),
        split_concurrency=3,
    )
    result = await pipeline.run(make_files({"a.txt": "body"}))

    assert result.processed == ["a.txt"]
    assert extractor.calls == 12
    assert extractor.max_in_flight == 3  # bounded — and still concurrent up to the bound
    assert sorted(m.ordinal for m in mstore.stored.values()) == list(range(12))  # split order kept


@pytest.mark.anyio
async def test_split_concurrency_default_is_bounded(make_files, mstore):
    extractor = InFlightExtractor()
    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=extractor,
        mention_store=mstore,
        splitter=n_way_splitter(40),
    )
    await pipeline.run(make_files({"a.txt": "body"}))
    assert extractor.calls == 40
    assert extractor.max_in_flight == 4  # the documented default


@pytest.mark.anyio
async def test_int_split_concurrency_is_per_source(make_files, mstore):
    """An int bounds each source separately: worst case is concurrency x split_concurrency."""
    extractor = InFlightExtractor()
    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=extractor,
        mention_store=mstore,
        splitter=n_way_splitter(8),
        concurrency=2,
        split_concurrency=2,
    )
    await pipeline.run(make_files({"a.txt": "a", "b.txt": "b", "c.txt": "c"}))
    assert extractor.calls == 24
    assert 2 <= extractor.max_in_flight <= 4


@pytest.mark.anyio
async def test_shared_semaphore_is_a_global_extraction_budget(make_files, mstore):
    """A shared asyncio.Semaphore caps extract() calls across ALL sources of the run."""
    import asyncio

    extractor = InFlightExtractor()
    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=extractor,
        mention_store=mstore,
        splitter=n_way_splitter(8),
        concurrency=3,
        split_concurrency=asyncio.Semaphore(2),
    )
    await pipeline.run(make_files({"a.txt": "a", "b.txt": "b", "c.txt": "c"}))
    assert extractor.calls == 24
    assert extractor.max_in_flight == 2


def test_split_concurrency_must_be_positive(mstore):
    with pytest.raises(ValueError, match="split_concurrency"):
        MentionStorePipeline(
            ingest=IngestPipeline(parser=make_parser()),
            extractor=InFlightExtractor(),
            mention_store=mstore,
            split_concurrency=0,
        )


# ---------------------------------------------------------------------------
# extract() raising inside the sync path
# ---------------------------------------------------------------------------


class FailingOnExtractor:
    """extract() raises for any split whose rendered body contains *marker*."""

    payload_model = Event

    def __init__(self, marker: str, error: Exception) -> None:
        self._marker = marker
        self._error = error
        self.started: list[str] = []
        self.finished: list[str] = []

    async def extract(self, document: Document) -> list[Mention[Event]]:
        import asyncio

        text = " ".join(e.text for e in document.elements)
        self.started.append(text)
        if self._marker in text:
            await asyncio.sleep(0)  # let sibling splits start
            raise self._error
        await asyncio.sleep(0.05)
        self.finished.append(text)
        return [Mention[Event](mention_id="pending", source_id=document.source_id or "", payload=Event(title=text))]


@pytest.mark.anyio
async def test_extract_error_is_isolated_per_source_in_run(make_files, mstore):
    """One source whose extract() raises lands in result.errors; the others sync; the failing
    source's previously stored mentions are left untouched (no half-applied delete)."""
    paths = make_files({"good.txt": "fine", "bad.txt": "POISON"})
    await mstore.upsert(
        [Mention[Event](mention_id="pre", source_id="bad.txt", source_hash="old", payload=Event(title="Old"))]
    )
    boom = RuntimeError("extractor exploded")
    pipeline = make_pipeline(FailingOnExtractor("POISON", boom), mstore)

    result = await pipeline.run(paths)

    assert result.processed == ["good.txt"]
    assert result.errors == [("bad.txt", boom)]  # the original exception object, keyed by source_id
    assert [m.payload.title for m in mstore.by_source("bad.txt")] == ["Old"]
    assert len(mstore.by_source("good.txt")) == 1


@pytest.mark.anyio
async def test_extract_error_in_one_split_fails_whole_source_and_cancels_siblings(make_files, mstore):
    """A source is atomic: one failing split stores nothing for it, and the sibling splits'
    extract() calls are cancelled instead of running (and billing) to completion."""
    boom = RuntimeError("split 2 exploded")
    extractor = FailingOnExtractor("split 2", boom)
    pipeline = MentionStorePipeline(
        ingest=IngestPipeline(parser=make_parser()),
        extractor=extractor,
        mention_store=mstore,
        splitter=n_way_splitter(3),
    )

    result = await pipeline.run(make_files({"a.txt": "body"}))

    assert result.processed == []
    assert result.errors == [("a.txt", boom)]
    assert mstore.stored == {}
    assert len(extractor.started) == 3
    assert extractor.finished == []  # siblings cancelled, not awaited to completion


@pytest.mark.anyio
async def test_extract_error_omits_source_from_plan(make_files, mstore):
    """plan() never raises for a failing source: it is logged and left out of the ChangeSet."""
    paths = make_files({"good.txt": "fine", "bad.txt": "POISON"})
    pipeline = make_pipeline(FailingOnExtractor("POISON", RuntimeError("extractor exploded")), mstore)

    changeset = await pipeline.plan(paths)

    assert [c.source_id for c in changeset.to_add] == ["good.txt"]
    assert changeset.to_update == [] and changeset.to_delete == []
    assert mstore.stored == {}  # plan() writes nothing


@pytest.mark.anyio
async def test_malformed_llm_output_fails_the_source_without_retry(make_files, mstore):
    """A completion that does not validate against the schema (pydantic ValidationError out of
    ``chat.completions.parse``) is not retried by StructuredExtractor and fails only that source."""
    from unittest.mock import AsyncMock, MagicMock

    from pydantic import ValidationError

    from ragdoc.extraction.structured import _batch_model

    async def _parse(**kwargs):  # type: ignore[no-untyped-def]
        if "POISON" in kwargs["messages"][1]["content"]:
            _batch_model(Event).model_validate_json('{"mentions": [{"title": ')  # truncated JSON
        batch = _batch_model(Event)(mentions=[Event(title="Ev")])
        return MagicMock(choices=[MagicMock(message=MagicMock(parsed=batch))])

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_parse)
    pipeline = make_pipeline(StructuredExtractor(Event, client=client, model="m"), mstore)

    result = await pipeline.run(make_files({"good.txt": "fine", "bad.txt": "POISON"}))

    assert result.processed == ["good.txt"]
    assert [sid for sid, _ in result.errors] == ["bad.txt"]
    assert isinstance(result.errors[0][1], ValidationError)
    assert client.chat.completions.parse.await_count == 2  # one per source: no retry of malformed output
