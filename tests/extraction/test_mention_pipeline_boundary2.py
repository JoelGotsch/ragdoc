"""Tests for MentionStorePipeline Boundary-2 mode (DocumentStore -> MentionStore).

Mirrors `tests/pipeline/test_vectorstore_mode2.py` for the mention path. When a `document_store`
is given, the pipeline reads already-parsed-and-processed Documents from it (no parsing, no
re-running of pre-split processors), splits + extracts, and hash-gates on the Document's
`content_hash` instead of the file-byte `source_hash`.
"""

from __future__ import annotations

import pytest

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.extraction.structured import StructuredExtractor
from ragdoc.pipeline.linear import DocumentPipeline
from ragdoc.pipeline.stores import SourceState
from ragdoc.processing.base import DocumentProcessor

from .conftest import Event, MemoryMentionStore, make_event_client, make_extractor

# ---------------------------------------------------------------------------
# Local in-memory DocumentStore — kept here to avoid coupling test packages
# ---------------------------------------------------------------------------


class MemoryDocumentStore:
    def __init__(self) -> None:
        self.stored: dict[str, Document] = {}

    async def upsert(self, documents: list[Document]) -> list[str]:
        ids: list[str] = []
        for d in documents:
            if not d.source_id:
                raise ValueError("MemoryDocumentStore requires document.source_id")
            self.stored[d.source_id] = d
            ids.append(d.source_id)
        return ids

    async def delete_by_source(self, source_id: str) -> None:
        self.stored.pop(source_id, None)

    async def get_document(self, source_id: str) -> Document | None:
        return self.stored.get(source_id)

    async def list_source_ids(self) -> set[str]:
        return set(self.stored.keys())

    async def list_source_state(self) -> dict[str, SourceState]:
        return {
            sid: SourceState(source_hash=d.source_hash or "", content_hash=d.content_hash())
            for sid, d in self.stored.items()
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(sid: str, body: str) -> Document:
    doc = Document(
        title=sid,
        elements=[Heading(innerhtml=sid, level=1), Paragraph(html_content=f"<p>{body}</p>")],
    )
    doc.source_id = sid
    doc.source_hash = f"filehash::{sid}"
    return doc


async def _seed(docs: MemoryDocumentStore, sid: str, body: str) -> Document:
    doc = _make_doc(sid, body)
    await docs.upsert([doc])
    return doc


def make_b2_pipeline(
    docs: MemoryDocumentStore,
    extractor: StructuredExtractor[Event],
    mstore: MemoryMentionStore,
) -> MentionStorePipeline[Event]:
    return MentionStorePipeline(
        pipeline=DocumentPipeline(),  # no parser, no processors — Boundary 2
        extractor=extractor,
        mention_store=mstore,
        document_store=docs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docs() -> MemoryDocumentStore:
    return MemoryDocumentStore()


# ---------------------------------------------------------------------------
# 1. Boundary-2 read path: stored Docs in → extract runs → no file I/O
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_none_extracts_all_documents(docs, mstore):
    await _seed(docs, "a.pdf", "Alpha body")
    await _seed(docs, "b.pdf", "Beta body")

    extractor = make_extractor([Event(title="Ev")])
    result = await make_b2_pipeline(docs, extractor, mstore).run()  # source_ids=None → all

    assert set(result.processed) == {"a.pdf", "b.pdf"}
    assert {m.source_id for m in await mstore.list_mentions()} == {"a.pdf", "b.pdf"}


@pytest.mark.anyio
async def test_run_does_not_touch_files(docs, mstore, tmp_path):
    """Boundary-2 reads from doc store — assert no parsing happened by giving the
    pipeline no parser and confirming it still works (a direct-mode run would crash)."""
    await _seed(docs, "a.pdf", "x")
    result = await make_b2_pipeline(docs, make_extractor([Event(title="Ev")]), mstore).run()
    assert result.processed == ["a.pdf"]


# ---------------------------------------------------------------------------
# 2. content_hash gating: unchanged doc skipped, no .parse() call
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_unchanged_document_skipped_no_llm_call(docs, mstore):
    await _seed(docs, "a.pdf", "Stable content")
    client = make_event_client([[Event(title="Ev")]])
    extractor = StructuredExtractor(Event, client=client, model="m")
    pipeline = make_b2_pipeline(docs, extractor, mstore)

    await pipeline.run()
    calls_after_first = client.beta.chat.completions.parse.call_count
    assert calls_after_first >= 1

    result = await pipeline.run()  # nothing changed in the doc store
    assert result.skipped == ["a.pdf"] and result.processed == []
    # the content_hash gate prevented any further extraction call
    assert client.beta.chat.completions.parse.call_count == calls_after_first


@pytest.mark.anyio
async def test_edited_document_is_reextracted(docs, mstore):
    await _seed(docs, "a.pdf", "Original")
    client = make_event_client([[Event(title="First")], [Event(title="Second")]])
    extractor = StructuredExtractor(Event, client=client, model="m")
    pipeline = make_b2_pipeline(docs, extractor, mstore)

    await pipeline.run()
    await _seed(docs, "a.pdf", "Edited entirely")  # new content_hash
    result = await pipeline.run()

    assert result.processed == ["a.pdf"] and result.skipped == []
    titles = [m.payload.title for m in mstore.by_source("a.pdf")]
    assert titles == ["Second"]  # old mention replaced wholesale, no duplicate


# ---------------------------------------------------------------------------
# 3. run(source_ids=...) defaults / subset
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_source_ids_subset_limits_processing(docs, mstore):
    await _seed(docs, "a.pdf", "A")
    await _seed(docs, "b.pdf", "B")
    result = await make_b2_pipeline(docs, make_extractor([Event(title="Ev")]), mstore).run(["a.pdf"])
    assert result.processed == ["a.pdf"]
    assert {m.source_id for m in await mstore.list_mentions()} == {"a.pdf"}


# ---------------------------------------------------------------------------
# 4. Boundary discipline: reject misplaced stages at construction
# ---------------------------------------------------------------------------


class _SpyProcessor(DocumentProcessor):
    async def process(self, document: Document) -> Document:
        return document


def test_boundary2_rejects_processors_on_pipeline(docs, mstore):
    """Documents loaded from the store were already processed at Boundary 1; re-running the
    chain is destructive. Reject at construction."""
    with pytest.raises(ValueError, match="must not carry"):
        MentionStorePipeline(
            pipeline=DocumentPipeline(processors=[_SpyProcessor()]),
            extractor=make_extractor([Event(title="Ev")]),
            mention_store=mstore,
            document_store=docs,
        )


def test_boundary2_rejects_custom_parser_on_pipeline(docs, mstore):
    async def _custom_parser(path):  # pragma: no cover
        raise AssertionError("must not be used")

    with pytest.raises(ValueError, match="custom parser"):
        MentionStorePipeline(
            pipeline=DocumentPipeline(parser=_custom_parser),
            extractor=make_extractor([Event(title="Ev")]),
            mention_store=mstore,
            document_store=docs,
        )


# ---------------------------------------------------------------------------
# 5. Per-source durability: bad source isolated; good source written
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_per_source_durability(docs):
    class FlakyStore(MemoryMentionStore):
        async def upsert(self, mentions):  # type: ignore[no-untyped-def]
            if any(m.source_id == "bad.pdf" for m in mentions):
                raise RuntimeError("boom")
            return await super().upsert(mentions)

    store = FlakyStore()
    await _seed(docs, "good.pdf", "g")
    await _seed(docs, "bad.pdf", "b")

    result = await make_b2_pipeline(docs, make_extractor([Event(title="Ev")]), store).run()

    assert "good.pdf" in result.processed
    assert [sid for sid, _ in result.errors] == ["bad.pdf"]
    assert await store.list_source_ids() == {"good.pdf"}


# ---------------------------------------------------------------------------
# 6. plan() / apply() — reviewable path, mirrored
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_apply_round_trip(docs, mstore, tmp_path):
    from ragdoc.extraction.mention import Mention
    from ragdoc.pipeline.changeset import ChangeSet

    await _seed(docs, "a.pdf", "Alpha")
    pipeline = make_b2_pipeline(docs, make_extractor([Event(title="Ev")]), mstore)

    changeset = await pipeline.plan()
    assert [sc.source_id for sc in changeset.to_add] == ["a.pdf"]
    assert mstore.stored == {}  # plan did not touch the mention store

    cs_path = tmp_path / "cs.json"
    changeset.save(cs_path)
    reloaded = ChangeSet[Mention[Event]].load(cs_path)
    result = await pipeline.apply(reloaded)

    assert result.processed == ["a.pdf"]
    assert {m.payload.title for m in await mstore.list_mentions()} == {"Ev"}


# ---------------------------------------------------------------------------
# 7. delete_orphans semantics (mirrors direct mode)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orphan_handling_against_doc_store(docs, mstore):
    """In Boundary 2 mode, an orphan is a source_id present in the MentionStore but absent from
    the DocumentStore (the document was deleted upstream)."""
    await _seed(docs, "a.pdf", "A")
    await _seed(docs, "b.pdf", "B")
    pipeline = make_b2_pipeline(docs, make_extractor([Event(title="Ev")]), mstore)
    await pipeline.run()

    # remove b from upstream — it becomes an orphan
    await docs.delete_by_source("b.pdf")

    result_default = await pipeline.run()
    assert result_default.deleted == []
    assert "b.pdf" in await mstore.list_source_ids()

    result = await pipeline.run(delete_orphans=True)
    assert "b.pdf" in result.deleted
    assert await mstore.list_source_ids() == {"a.pdf"}
