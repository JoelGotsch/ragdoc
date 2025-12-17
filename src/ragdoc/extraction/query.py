"""A backend-agnostic query spec for filtering entities (for agent tools).

The same :class:`EntityQuery` drives both in-memory filtering over a
:class:`~ragdoc.extraction.entity.LocalEntityStore` and server-side filtering in
:class:`~ragdoc.integrations.entity_stores.qdrant.QdrantEntityStore`, so an agent tool's
code is the same regardless of backend::

    q = EntityQuery(date=DateRange(date(1990, 1, 1), date(1995, 1, 1)), where={"facility": "A"})
    hits = await entity_store.query(q)

Date filtering is a half-open interval **overlap** ([`FuzzyDate`][ragdoc.extraction.dates.FuzzyDate]
bounds vs the query range); ``where`` matches caller-payload fields by exact value.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragdoc.extraction.entity import Entity


@dataclass(frozen=True)
class DateRange:
    """A half-open ``[start, end)`` query window. ``None`` bounds are unbounded.

    Attributes:
        start: Inclusive earliest date (``None`` = unbounded past).
        end: Exclusive latest date (``None`` = unbounded future).
    """

    start: dt.date | None = None
    end: dt.date | None = None


@dataclass
class EntityQuery:
    """A filter over entities, evaluated identically by every store backend.

    Attributes:
        date: When set, keep only entities whose date interval overlaps this range.
        where: Exact-match constraints on canonical-payload fields (``{field: value}``).
        limit: Maximum number of results to return (``None`` = no limit).
    """

    date: DateRange | None = None
    where: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None


def entity_matches(entity: Entity, query: EntityQuery) -> bool:
    """Return whether *entity* satisfies *query* (the in-memory evaluation, shared by Local stores)."""
    if query.date is not None:
        if entity.date is None or not entity.date.overlaps(query.date.start, query.date.end):
            return False
    for key, value in query.where.items():
        if getattr(entity.payload, key, None) != value:
            return False
    return True


def filter_entities(entities: list[Entity], query: EntityQuery) -> list[Entity]:
    """Apply *query* to *entities* in memory, honouring ``limit``."""
    matched = [e for e in entities if entity_matches(e, query)]
    return matched[: query.limit] if query.limit is not None else matched
