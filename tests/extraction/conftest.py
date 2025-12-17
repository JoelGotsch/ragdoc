"""Shared fixtures for extraction tests: a sample payload, mock client, and in-memory stores."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.entity import Entity
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.processor import StructuredExtractionProcessor, _batch_model
from ragdoc.extraction.resolution import ReviewGroup, ReviewResult
from ragdoc.pipeline.stores import SourceState


class Event(BaseModel):
    """An event mentioned in the document."""

    title: str = Field(description="Short event title.")
    date: FuzzyDate | None = Field(default=None, description="When the event happened, if stated.")


def make_event_client(events_per_call: list[list[Event]]) -> MagicMock:
    """Mock client whose successive parse() calls return successive batches of events.

    One split → one parse() call. Pass a list-of-lists so multi-split docs return per-split events;
    the last batch is reused if more calls arrive than batches provided.
    """
    calls = {"i": 0}

    async def _parse(**_kwargs):  # type: ignore[no-untyped-def]
        i = min(calls["i"], len(events_per_call) - 1)
        calls["i"] += 1
        batch = _batch_model(Event)(mentions=events_per_call[i])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=batch))])

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(side_effect=_parse)
    return client


def make_extractor(events: list[Event]) -> StructuredExtractionProcessor[Event]:
    """An extractor whose every split yields *events* (mock client)."""
    return StructuredExtractionProcessor(Event, client=make_event_client([events]), model="test-model")


class MemoryMentionStore:
    """In-memory MentionStore for tests."""

    def __init__(self) -> None:
        self.stored: dict[str, Mention] = {}

    async def upsert(self, mentions: list[Mention]) -> list[str]:
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

    async def list_mentions(self, payload_type: type[BaseModel] | None = None) -> list[Mention]:
        all_mentions = list(self.stored.values())
        if payload_type is None:
            return all_mentions
        return [m for m in all_mentions if isinstance(m.payload, payload_type)]

    def by_source(self, source_id: str) -> list[Mention]:
        return [m for m in self.stored.values() if m.source_id == source_id]


class MemoryEntityStore:
    """In-memory EntityStore for tests."""

    def __init__(self) -> None:
        self.stored: dict[str, Entity] = {}

    async def upsert(self, entities: list[Entity]) -> list[str]:
        for e in entities:
            self.stored[e.entity_id] = e
        return [e.entity_id for e in entities]

    async def delete(self, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            self.stored.pop(eid, None)

    async def delete_by_source(self, source_id: str) -> None:
        for eid in [eid for eid, e in self.stored.items() if e.source_ids == [source_id]]:
            del self.stored[eid]

    async def list_source_ids(self) -> set[str]:
        return {s for e in self.stored.values() for s in e.source_ids}

    async def list_entities(self) -> list[Entity]:
        return list(self.stored.values())

    async def query(self, query):  # type: ignore[no-untyped-def]
        from ragdoc.extraction.query import filter_entities

        return filter_entities(list(self.stored.values()), query)


def make_mention(
    mention_id: str,
    title: str,
    source_id: str,
    *,
    date: FuzzyDate | None = None,
    content_hash: str = "c",
    source_hash: str = "h",
) -> Mention[Event]:
    return Mention[Event](
        mention_id=mention_id,
        source_id=source_id,
        source_hash=source_hash,
        content_hash=content_hash,
        payload=Event(title=title, date=date),
    )


def dict_embed(mapping: dict[str, list[float]], default: list[float] | None = None):
    """A fake text embedder backed by *mapping*; unknown texts get *default* (a distinct vector)."""
    fallback = default if default is not None else [0.0, 0.0, 1.0]

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [list(mapping.get(t, fallback)) for t in texts]

    return _embed


async def merge_all_reviewer(texts: list[str]) -> ReviewResult:
    """Reviewer that merges every candidate in the group (confidence 1.0)."""
    return ReviewResult(groups=[ReviewGroup(members=list(range(len(texts))), confidence=1.0)])


async def merge_first_pair_reviewer(texts: list[str]) -> ReviewResult:
    """Reviewer that merges only the first two candidates each round (forces many rounds)."""
    if len(texts) < 2:
        return ReviewResult(groups=[])
    return ReviewResult(groups=[ReviewGroup(members=[0, 1], confidence=1.0)])


async def no_merge_reviewer(texts: list[str]) -> ReviewResult:
    """Reviewer that never merges (each candidate is its own entity)."""
    return ReviewResult(groups=[ReviewGroup(members=[i], confidence=1.0) for i in range(len(texts))])


async def low_conf_reviewer(texts: list[str]) -> ReviewResult:
    """Reviewer that proposes a merge but below the auto-merge bar (→ pending)."""
    return ReviewResult(groups=[ReviewGroup(members=list(range(len(texts))), confidence=0.3)])


@pytest.fixture
def mstore() -> MemoryMentionStore:
    return MemoryMentionStore()


@pytest.fixture
def estore() -> MemoryEntityStore:
    return MemoryEntityStore()


@pytest.fixture
def make_files(tmp_path: Path):
    def _make(names_and_contents: dict[str, str]) -> list[Path]:
        paths = []
        for name, content in names_and_contents.items():
            p = tmp_path / name
            p.write_text(content, encoding="utf-8")
            paths.append(p)
        return paths

    return _make


def make_parser():
    """A parser that turns a file's text into a one-paragraph Document."""

    async def _parse(path: Path) -> Document:
        return Document(
            title=path.stem,
            elements=[Heading(innerhtml=path.stem, level=1), Paragraph(html_content=f"<p>{path.read_text()}</p>")],
        )

    return _parse
