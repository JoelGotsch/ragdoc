"""Tests for the backend-agnostic entity query (in-memory path over LocalEntityStore)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.entity import Entity, LocalEntityStore
from ragdoc.extraction.query import DateRange, EntityQuery, entity_matches

from .conftest import Event


def entity(eid: str, title: str, *, edtf: str | None = None, sources: list[str] | None = None) -> Entity[Event]:
    date = FuzzyDate(original_text=edtf or "", edtf=edtf, precision="YEAR") if edtf else None
    return Entity[Event](
        entity_id=eid,
        payload=Event(title=title),
        member_mention_ids=[f"{eid}-m"],
        source_ids=sources or ["s1"],
        date=date,
    )


D = dt.date


def test_date_overlap_matches():
    e = entity("e1", "shutdown", edtf="1994")  # [1994-01-01, 1995-01-01)
    assert entity_matches(e, EntityQuery(date=DateRange(D(1990, 1, 1), D(1996, 1, 1))))
    assert not entity_matches(e, EntityQuery(date=DateRange(D(1995, 1, 1), D(2000, 1, 1))))  # adjacent, no overlap


def test_undated_entity_excluded_by_date_query():
    assert not entity_matches(entity("e1", "x"), EntityQuery(date=DateRange(D(1990, 1, 1), D(1991, 1, 1))))


def test_where_field_match():
    e = entity("e1", "shutdown")
    assert entity_matches(e, EntityQuery(where={"title": "shutdown"}))
    assert not entity_matches(e, EntityQuery(where={"title": "inspection"}))


def test_combined_date_and_where():
    e = entity("e1", "shutdown", edtf="1994")
    assert entity_matches(e, EntityQuery(date=DateRange(D(1990, 1, 1), D(1996, 1, 1)), where={"title": "shutdown"}))
    assert not entity_matches(e, EntityQuery(date=DateRange(D(1990, 1, 1), D(1996, 1, 1)), where={"title": "other"}))


@pytest.mark.anyio
async def test_local_entity_store_query(tmp_path: Path):
    store = LocalEntityStore(tmp_path, Event)
    await store.upsert(
        [
            entity("e1", "1994 shutdown", edtf="1994"),
            entity("e2", "1980 startup", edtf="1980"),
            entity("e3", "undated note"),
        ]
    )

    in_range = await store.query(EntityQuery(date=DateRange(D(1990, 1, 1), D(2000, 1, 1))))
    assert {e.entity_id for e in in_range} == {"e1"}

    by_title = await store.query(EntityQuery(where={"title": "1980 startup"}))
    assert {e.entity_id for e in by_title} == {"e2"}


@pytest.mark.anyio
async def test_local_entity_store_query_limit(tmp_path: Path):
    store = LocalEntityStore(tmp_path, Event)
    await store.upsert([entity(f"e{i}", "shutdown", edtf="1994") for i in range(5)])
    assert len(await store.query(EntityQuery(where={"title": "shutdown"}, limit=2))) == 2


def test_fuzzy_date_bounds_are_persisted():
    # computed_field => derived bounds appear in the serialized JSON (so stores can index them)
    dumped = FuzzyDate(original_text="1994", edtf="1994", precision="YEAR").model_dump(mode="json")
    assert dumped["start"] == "1994-01-01" and dumped["end"] == "1995-01-01"
