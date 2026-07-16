"""Contract test suite for the three sync pipelines.

One suite, parametrized over five harnesses:

* ``vector-direct``  — VectorStorePipeline, direct path (files -> chunks).
* ``vector-b2``      — VectorStorePipeline, Boundary-2 mode (DocumentStore -> chunks).
* ``docstore``       — DocumentStorePipeline (files -> Documents, Boundary 1).
* ``mention-direct`` — MentionStorePipeline, direct path (files -> mentions).
* ``mention-b2``     — MentionStorePipeline, Boundary-2 mode (DocumentStore -> mentions).

Every case pins the **unified** behavior implemented by
:class:`~ragdoc.pipeline.sync.SyncEngine` (spec §2), including the former drift points:

* §2.4 — a custom ``BaseException`` propagates (``except Exception`` only).
* §2.5 — ``apply()`` skips the upsert after a failed stale-delete (no old/new mixing).
* §2.6 — a filtered source clears its stale **derived** entries (chunks/mentions, counted
  processed) but **preserves** the stored Document at Boundary 1 (skip-with-warning —
  a transient filter misclassification must not destroy durable state).
* §2.2 — a target sid missing from the doc store is warned + skip-counted.
* §2.3 — a document that vanishes between ``list_source_state`` and ``get_document`` is
  warned + skip-counted.

Stores are in-memory fakes implementing the store protocols (``MemoryVectorStore`` /
``MemoryDocumentStore`` reused from ``tests/pipeline/conftest.py``; ``MemoryMentionStore``
mirrors ``tests/extraction/conftest.py``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ragdoc.chunking import Chunk, SimpleChunker
from ragdoc.document import Document
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.pipeline import ChunkPipeline, DocumentPipeline, IngestPipeline, VectorStorePipeline
from ragdoc.pipeline.changeset import ChangeSet
from ragdoc.pipeline.document_store_pipeline import DocumentStorePipeline
from ragdoc.pipeline.stores import SourceState
from ragdoc.processing.base import DocumentProcessor

from .conftest import MemoryDocumentStore, MemoryVectorStore, make_document

# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------

ALL_CASES = ["vector-direct", "vector-b2", "docstore", "mention-direct", "mention-b2"]
DIRECT_CASES = ["vector-direct", "docstore", "mention-direct"]
B2_CASES = ["vector-b2", "mention-b2"]


def for_cases(names: Sequence[str], xfail: Sequence[str] = (), reason: str = "") -> pytest.MarkDecorator:
    """Parametrize ``case_name`` over *names*, strict-xfailing the *xfail* subset."""
    params = [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=reason)) if name in xfail else name
        for name in names
    ]
    return pytest.mark.parametrize("case_name", params)


def error_sids(result: object) -> list[str]:
    return [sid for sid, _ in result.errors]  # type: ignore[attr-defined]


def changed_sids(changeset: object) -> set[str]:
    cs = changeset
    return {sc.source_id for sc in [*cs.to_add, *cs.to_update]}  # type: ignore[attr-defined]


def _default_fail() -> BaseException:
    return RuntimeError("production boom")


# ---------------------------------------------------------------------------
# Payload model + fake extractor (mention cases)
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """An event mentioned in the document (test payload)."""

    title: str = Field(description="Short event title.")


class EchoExtractor:
    """Duck-typed stand-in for StructuredExtractor: no LLM, one deterministic
    ``Event`` per split whose title is derived from the split's renderer-stable content hash
    (so it changes with the content and is identical across re-parses of the same content)."""

    payload_model = Event

    async def extract(self, document: Document) -> list[Mention[Event]]:
        return [
            Mention[Event](
                mention_id="pending",
                source_id=document.source_id or "",
                source_hash=document.source_hash or "",
                payload=Event(title=f"ev-{document.content_hash()[:16]}"),
            )
        ]


# ---------------------------------------------------------------------------
# In-memory MentionStore (mirrors tests/extraction/conftest.py, kept local to
# avoid coupling test packages)
# ---------------------------------------------------------------------------


class MemoryMentionStore:
    """In-memory MentionStore implementing the store protocol used by the sync pipelines."""

    def __init__(self) -> None:
        self.stored: dict[str, Mention[Event]] = {}

    async def upsert(self, mentions: list[Mention[Event]]) -> list[str]:
        for m in mentions:
            self.stored[m.mention_id] = m
        return [m.mention_id for m in mentions]

    async def delete_by_source(self, source_id: str) -> None:
        for mid in [mid for mid, m in self.stored.items() if m.source_id == source_id]:
            del self.stored[mid]

    async def list_source_ids(self) -> set[str]:
        return {m.source_id for m in self.stored.values()}

    async def list_source_state(self) -> dict[str, SourceState]:
        state: dict[str, SourceState] = {}
        for m in self.stored.values():
            state[m.source_id] = SourceState(source_hash=m.source_hash, content_hash=m.content_hash)
        return state


class SpyDocumentStore(MemoryDocumentStore):
    """Upstream DocumentStore spy for the B2 cases.

    * ``get_calls`` counts ``get_document`` invocations (the B2 production entry point).
    * ``vanished`` simulates a document disappearing between ``list_source_state`` and
      ``get_document`` (spec §2.3): still listed, but ``get_document`` returns ``None``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0
        self.vanished: set[str] = set()

    async def get_document(self, source_id: str) -> Document | None:
        self.get_calls += 1
        if source_id in self.vanished:
            return None
        return await super().get_document(source_id)


# ---------------------------------------------------------------------------
# Gate stages: per-sid failure injection and filtering
# ---------------------------------------------------------------------------


class _GateProcessor(DocumentProcessor):
    """Drops documents whose source_id is in the (shared, mutable) *filtered* set."""

    def __init__(self, filtered: set[str]) -> None:
        self._filtered = filtered

    async def process(self, document: Document) -> Document | None:
        if document.source_id in self._filtered:
            return None
        return document


class _GateChunker(SimpleChunker):
    """SimpleChunker that raises for failing sids — B2 production-failure injection."""

    def __init__(self, harness: _HarnessBase) -> None:
        super().__init__()
        self._harness = harness

    async def chunk(self, document: Document) -> list[Chunk]:
        if document.source_id in self._harness.failing:
            raise self._harness.fail_exc()
        return await super().chunk(document)


# ---------------------------------------------------------------------------
# Harnesses
# ---------------------------------------------------------------------------


class _HarnessBase:
    """Uniform facade over one pipeline + its in-memory sink store.

    ``failing`` / ``fail_exc`` / ``filtered`` are mutable knobs read at production time, so
    tests can flip them between runs on one pipeline instance.
    """

    name: str
    is_b2: bool

    def __init__(self) -> None:
        self.failing: set[str] = set()
        self.fail_exc: Callable[[], BaseException] = _default_fail
        self.filtered: set[str] = set()
        self.pipeline: VectorStorePipeline | DocumentStorePipeline | MentionStorePipeline[Event]
        self.store: MemoryVectorStore | MemoryDocumentStore | MemoryMentionStore

    # -- source management (implemented per mode) --
    async def add(self, sid: str, content: str) -> None:
        raise NotImplementedError

    async def edit(self, sid: str, content: str) -> None:
        await self.add(sid, content)

    async def remove(self, sid: str) -> None:
        raise NotImplementedError

    def _sources_arg(self, sids: list[str] | None) -> object:
        raise NotImplementedError

    def production_calls(self) -> int:
        raise NotImplementedError

    # -- pipeline entry points --
    async def run(self, sids: list[str] | None = None, delete_orphans: bool = False) -> object:
        return await self.pipeline.run(self._sources_arg(sids), delete_orphans=delete_orphans)  # type: ignore[arg-type]

    async def plan(self, sids: list[str] | None = None, delete_orphans: bool = False) -> object:
        return await self.pipeline.plan(self._sources_arg(sids), delete_orphans=delete_orphans)  # type: ignore[arg-type]

    async def apply(self, changeset: object) -> object:
        return await self.pipeline.apply(changeset)  # type: ignore[arg-type]

    def load_changeset(self, path: Path) -> object:
        raise NotImplementedError

    # -- sink inspection --
    def items(self, sid: str) -> list[object]:
        raise NotImplementedError

    def markers(self, sid: str) -> set[str]:
        """Content-stable identity of the items currently stored for *sid*."""
        raise NotImplementedError

    async def store_ids(self) -> set[str]:
        return await self.store.list_source_ids()

    def store_dump(self) -> dict[str, object]:
        return dict(self.store.stored)


class _DirectBase(_HarnessBase):
    """Direct-mode base: sources are real files under tmp_path; production begins at parse."""

    is_b2 = False

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.tmp = tmp_path
        self.paths: dict[str, Path] = {}
        self.parse_calls = 0

    def _make_parser(self) -> Callable[[Path], object]:
        async def _parse(path: Path) -> Document:
            self.parse_calls += 1
            if path.name in self.failing:
                raise self.fail_exc()
            return make_document(title=path.stem, body=path.read_text(encoding="utf-8"))

        return _parse

    async def add(self, sid: str, content: str) -> None:
        p = self.tmp / sid
        p.write_text(content, encoding="utf-8")
        self.paths[sid] = p

    async def remove(self, sid: str) -> None:
        self.paths.pop(sid)

    def _sources_arg(self, sids: list[str] | None) -> list[Path]:
        names = sids if sids is not None else list(self.paths)
        return [self.paths[s] for s in names]

    def production_calls(self) -> int:
        return self.parse_calls


class _B2Base(_HarnessBase):
    """Boundary-2 base: sources are Documents seeded into an upstream DocumentStore;
    production begins at ``get_document``."""

    is_b2 = True

    def __init__(self) -> None:
        super().__init__()
        self.upstream = SpyDocumentStore()

    async def add(self, sid: str, content: str) -> None:
        doc = make_document(title=sid, body=content)
        doc.source_id = sid
        doc.source_hash = f"filehash::{sid}"
        await self.upstream.upsert([doc])

    async def remove(self, sid: str) -> None:
        await self.upstream.delete_by_source(sid)

    def _sources_arg(self, sids: list[str] | None) -> list[str] | None:
        return sids

    def production_calls(self) -> int:
        return self.upstream.get_calls


class _VectorSinkMixin(_HarnessBase):
    store: MemoryVectorStore

    def items(self, sid: str) -> list[Chunk]:
        return [c for c in self.store.stored.values() if c.source_id == sid]

    def markers(self, sid: str) -> set[str]:
        return {c.prompt_content for c in self.items(sid)}

    def load_changeset(self, path: Path) -> ChangeSet[Chunk]:
        return ChangeSet[Chunk].load(path)


class _MentionSinkMixin(_HarnessBase):
    store: MemoryMentionStore

    def items(self, sid: str) -> list[Mention[Event]]:
        return [m for m in self.store.stored.values() if m.source_id == sid]

    def markers(self, sid: str) -> set[str]:
        return {m.payload.title for m in self.items(sid)}

    def load_changeset(self, path: Path) -> ChangeSet[Mention[Event]]:
        return ChangeSet[Mention[Event]].load(path)


def _identity_splitter(harness: _HarnessBase) -> Callable[[Document], list[Document]]:
    def _split(document: Document) -> list[Document]:
        if document.source_id in harness.failing:
            raise harness.fail_exc()
        return [document]

    return _split


class VectorDirectHarness(_VectorSinkMixin, _DirectBase):
    name = "vector-direct"

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.store = MemoryVectorStore()
        self.pipeline = VectorStorePipeline(
            pipeline=DocumentPipeline(parser=self._make_parser(), processors=[_GateProcessor(self.filtered)]),
            vector_store=self.store,
        )


class VectorB2Harness(_VectorSinkMixin, _B2Base):
    name = "vector-b2"

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.store = MemoryVectorStore()
        self.pipeline = VectorStorePipeline.from_document_store(
            chunk=ChunkPipeline(chunker=_GateChunker(self)),
            vector_store=self.store,
            document_store=self.upstream,
        )


class DocStoreHarness(_DirectBase):
    name = "docstore"

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.store = MemoryDocumentStore()
        self.pipeline = DocumentStorePipeline(
            ingest=IngestPipeline(parser=self._make_parser(), processors=[_GateProcessor(self.filtered)]),
            document_store=self.store,
        )

    def items(self, sid: str) -> list[Document]:
        doc = self.store.stored.get(sid)
        return [doc] if doc is not None else []

    def markers(self, sid: str) -> set[str]:
        return {d.content_hash() for d in self.items(sid)}

    def load_changeset(self, path: Path) -> ChangeSet[Document]:
        return ChangeSet[Document].load(path)


class MentionDirectHarness(_MentionSinkMixin, _DirectBase):
    name = "mention-direct"

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.store = MemoryMentionStore()
        self.pipeline = MentionStorePipeline(
            ingest=IngestPipeline(parser=self._make_parser(), processors=[_GateProcessor(self.filtered)]),
            extractor=EchoExtractor(),  # type: ignore[arg-type]
            mention_store=self.store,
            splitter=_identity_splitter(self),
        )


class MentionB2Harness(_MentionSinkMixin, _B2Base):
    name = "mention-b2"

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.store = MemoryMentionStore()
        self.pipeline = MentionStorePipeline.from_document_store(
            extractor=EchoExtractor(),  # type: ignore[arg-type]
            mention_store=self.store,
            document_store=self.upstream,
            splitter=_identity_splitter(self),
        )


_BUILDERS: dict[str, type] = {
    "vector-direct": VectorDirectHarness,
    "vector-b2": VectorB2Harness,
    "docstore": DocStoreHarness,
    "mention-direct": MentionDirectHarness,
    "mention-b2": MentionB2Harness,
}


@pytest.fixture
def make_case(tmp_path: Path) -> Callable[[str], _HarnessBase]:
    def _make(name: str) -> _HarnessBase:
        return _BUILDERS[name](tmp_path)

    return _make


# ===========================================================================
# plan()
# ===========================================================================


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_plan_new_sources_in_to_add(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    await case.add("b.txt", "Beta")

    changeset = await case.plan()
    assert {sc.source_id for sc in changeset.to_add} == {"a.txt", "b.txt"}
    assert changeset.to_update == [] and changeset.to_delete == []


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_plan_unchanged_source_absent_from_changeset(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Stable")
    await case.run()

    changeset = await case.plan()
    assert changeset.to_add == [] and changeset.to_update == []


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_plan_changed_source_in_to_update(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "v1")
    await case.run()

    await case.edit("a.txt", "v2 entirely different")
    changeset = await case.plan()
    assert {sc.source_id for sc in changeset.to_update} == {"a.txt"}
    assert changeset.to_add == []


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_plan_never_writes_store(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")

    await case.plan()
    assert case.store_dump() == {}


@for_cases(DIRECT_CASES)
@pytest.mark.anyio
async def test_plan_collision_raises_before_production(case_name, make_case):
    case = make_case(case_name)
    (case.tmp / "d1").mkdir()
    (case.tmp / "d2").mkdir()
    p1, p2 = case.tmp / "d1" / "s.txt", case.tmp / "d2" / "s.txt"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source_ids"):
        await case.pipeline.plan([p1, p2])
    assert case.production_calls() == 0
    assert case.store_dump() == {}


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_plan_producer_error_source_omitted_and_logged(case_name, make_case, caplog):
    case = make_case(case_name)
    await case.add("good.txt", "Alpha")
    await case.add("bad.txt", "Beta")
    case.failing.add("bad.txt")

    with caplog.at_level(logging.WARNING):
        changeset = await case.plan()

    assert changed_sids(changeset) == {"good.txt"}
    assert any("failed" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_plan_skips_have_no_llm_or_parse_calls(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Stable")
    await case.run()
    calls_after_run = case.production_calls()
    assert calls_after_run >= 1

    changeset = await case.plan()
    assert changed_sids(changeset) == set()
    assert case.production_calls() == calls_after_run


# ===========================================================================
# run()
# ===========================================================================


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_processes_new_sources(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    await case.add("b.txt", "Beta")

    result = await case.run()
    assert set(result.processed) == {"a.txt", "b.txt"}
    assert result.skipped == [] and result.errors == [] and result.deleted == []
    assert await case.store_ids() == {"a.txt", "b.txt"}
    assert case.markers("a.txt") and case.markers("b.txt")


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_second_run_skips_all(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Stable")
    await case.run()
    calls_after_first = case.production_calls()

    result = await case.run()
    assert result.skipped == ["a.txt"] and result.processed == []
    assert case.production_calls() == calls_after_first


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_reprocesses_on_token_change(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "v1")
    await case.run()
    old_markers = case.markers("a.txt")

    await case.edit("a.txt", "v2 entirely different content")
    result = await case.run()
    assert result.processed == ["a.txt"] and result.skipped == []
    assert case.markers("a.txt") != old_markers


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_update_is_delete_then_upsert(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "v1")
    await case.run()
    old_markers = case.markers("a.txt")

    await case.edit("a.txt", "v2 entirely different content")
    await case.run()
    new_markers = case.markers("a.txt")
    assert new_markers, "update must leave the new items in the store"
    assert new_markers.isdisjoint(old_markers), "no stale items may survive an update"


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_one_bad_source_does_not_block_others(case_name, make_case):
    case = make_case(case_name)
    await case.add("good.txt", "Alpha")
    await case.add("bad.txt", "Beta")
    case.failing.add("bad.txt")

    result = await case.run()
    assert result.processed == ["good.txt"]
    assert error_sids(result) == ["bad.txt"]
    assert await case.store_ids() == {"good.txt"}


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_exception_collected_in_errors(case_name, make_case):
    case = make_case(case_name)
    await case.add("bad.txt", "Beta")
    case.failing.add("bad.txt")

    result = await case.run()
    assert error_sids(result) == ["bad.txt"]
    assert isinstance(result.errors[0][1], RuntimeError)
    assert result.processed == []
    assert case.items("bad.txt") == []


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_cancelled_error_propagates(case_name, make_case):
    case = make_case(case_name)
    await case.add("bad.txt", "Beta")
    case.failing.add("bad.txt")
    case.fail_exc = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await case.run()


class _WeirdBaseError(BaseException):
    """A BaseException subclass that is neither Exception nor CancelledError."""


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_custom_base_exception_propagates(case_name, make_case):
    case = make_case(case_name)
    await case.add("bad.txt", "Beta")
    case.failing.add("bad.txt")
    case.fail_exc = _WeirdBaseError

    with pytest.raises(_WeirdBaseError):
        await case.run()


# ===========================================================================
# delete_orphans
# ===========================================================================


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_delete_orphans_default_false_keeps_absent(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    await case.add("b.txt", "Beta")
    await case.run()

    await case.remove("b.txt")
    result = await case.run()
    assert result.deleted == []
    assert "b.txt" in await case.store_ids()


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_delete_orphans_true_removes_absent(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    await case.add("b.txt", "Beta")
    await case.run()

    await case.remove("b.txt")
    result = await case.run(delete_orphans=True)
    assert result.deleted == ["b.txt"]
    assert await case.store_ids() == {"a.txt"}


@for_cases(B2_CASES)
@pytest.mark.anyio
async def test_b2_orphans_compared_against_full_doc_store(case_name, make_case):
    """A subset run must never delete sinks whose documents still exist upstream."""
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    await case.add("b.txt", "Beta")
    await case.run()

    result = await case.run(["a.txt"], delete_orphans=True)
    assert result.deleted == []
    assert "b.txt" in await case.store_ids()


# ===========================================================================
# apply()
# ===========================================================================


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_apply_writes_buffered_changeset(case_name, make_case, tmp_path):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")

    changeset = await case.plan()
    cs_path = tmp_path / "changeset.json"
    changeset.save(cs_path)
    reloaded = case.load_changeset(cs_path)

    result = await case.apply(reloaded)
    assert result.processed == ["a.txt"] and result.errors == []
    assert await case.store_ids() == {"a.txt"}
    assert case.markers("a.txt")


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_apply_to_delete_processed_first(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    await case.add("b.txt", "Beta")
    await case.run()

    await case.remove("b.txt")
    changeset = await case.plan(delete_orphans=True)
    assert changeset.to_delete == ["b.txt"]

    result = await case.apply(changeset)
    assert result.deleted == ["b.txt"]
    assert await case.store_ids() == {"a.txt"}


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_apply_skips_upsert_after_failed_stale_delete(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "v1")
    await case.run()
    old_markers = case.markers("a.txt")

    await case.edit("a.txt", "v2 entirely different content")
    changeset = await case.plan()
    assert {sc.source_id for sc in changeset.to_update} == {"a.txt"}

    original_delete = case.store.delete_by_source

    async def _failing_delete(source_id: str) -> None:
        if source_id == "a.txt":
            raise RuntimeError("delete boom")
        await original_delete(source_id)

    case.store.delete_by_source = _failing_delete  # type: ignore[method-assign]

    result = await case.apply(changeset)
    assert "a.txt" in error_sids(result)
    assert "a.txt" not in result.processed
    assert case.markers("a.txt") == old_markers, "store must keep the old items only — no old/new mixing"


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_run_equals_apply_plan_end_state(case_name, make_case):
    run_case = make_case(case_name)
    apply_case = make_case(case_name)
    for case in (run_case, apply_case):
        await case.add("a.txt", "Alpha content")
        await case.add("b.txt", "Beta content")

    run_result = await run_case.run()
    apply_result = await apply_case.apply(await apply_case.plan())

    assert set(run_result.processed) == set(apply_result.processed) == {"a.txt", "b.txt"}
    assert await run_case.store_ids() == await apply_case.store_ids()
    for sid in ("a.txt", "b.txt"):
        assert run_case.markers(sid) == apply_case.markers(sid)


# ===========================================================================
# Filtered sources (spec §2.6)
# ===========================================================================


@for_cases(["vector-direct", "mention-direct"])
@pytest.mark.anyio
async def test_filtered_source_clears_stale_derived_entries(case_name, make_case):
    """Derived stores (chunks/mentions): a filtered source yields nothing — stale entries go."""
    case = make_case(case_name)
    await case.add("a.txt", "v1")
    await case.run()
    assert case.markers("a.txt")

    await case.edit("a.txt", "v2 which processing now filters out")
    case.filtered.add("a.txt")
    result = await case.run()

    assert result.processed == ["a.txt"]
    assert case.items("a.txt") == []
    assert "a.txt" not in await case.store_ids()


@for_cases(["docstore"])
@pytest.mark.anyio
async def test_filtered_source_preserves_stored_document(case_name, make_case, caplog):
    """Boundary 1 (durable Documents): a filtered source is skipped and its stored copy kept."""
    case = make_case(case_name)
    await case.add("a.txt", "v1")
    await case.run()
    old_markers = case.markers("a.txt")
    assert old_markers

    await case.edit("a.txt", "v2 which processing now filters out")
    case.filtered.add("a.txt")
    with caplog.at_level(logging.WARNING):
        result = await case.run()

    assert result.skipped == ["a.txt"] and result.processed == []
    assert case.markers("a.txt") == old_markers  # the v1 Document survives
    assert "a.txt" in await case.store_ids()
    assert any("filtered by processing" in r.getMessage() for r in caplog.records)


# ===========================================================================
# Boundary-2 missing / vanished documents (spec §2.2, §2.3)
# ===========================================================================


@for_cases(B2_CASES)
@pytest.mark.anyio
async def test_b2_missing_target_warned_and_skip_counted(case_name, make_case, caplog):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")

    with caplog.at_level(logging.WARNING):
        result = await case.run(["ghost.txt"])

    assert "ghost.txt" in result.skipped
    assert result.processed == [] and result.errors == []
    assert any("ghost.txt" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


@for_cases(B2_CASES)
@pytest.mark.anyio
async def test_b2_plan_missing_target_logs_warning(case_name, make_case, caplog):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")

    with caplog.at_level(logging.WARNING):
        changeset = await case.plan(["ghost.txt"])

    assert changed_sids(changeset) == set()
    assert any("ghost.txt" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


@for_cases(B2_CASES)
@pytest.mark.anyio
async def test_b2_vanished_document_skip_counted(case_name, make_case, caplog):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha")
    case.upstream.vanished.add("a.txt")

    with caplog.at_level(logging.WARNING):
        result = await case.run()

    assert result.skipped == ["a.txt"]
    assert result.processed == [] and result.errors == []
    assert any("a.txt" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


# ===========================================================================
# Provenance
# ===========================================================================


@for_cases(ALL_CASES)
@pytest.mark.anyio
async def test_items_carry_provenance(case_name, make_case):
    case = make_case(case_name)
    await case.add("a.txt", "Alpha content")
    await case.run()

    items = case.items("a.txt")
    assert items
    for item in items:
        assert item.source_id == "a.txt"
        assert item.source_hash
