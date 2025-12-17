""":class:`Entity` — a canonical, deduplicated record (Layer 2), plus its store protocol.

An :class:`Entity` is produced by the resolution pass from a cluster of
:class:`~ragdoc.extraction.mention.Mention`\\ s judged to refer to the same real-world
thing. It records the merged payload, the surface ``aliases`` seen, the contributing
``member_mention_ids``, **all** originating ``source_ids``, and (when the payload carries a date)
the union :class:`~ragdoc.extraction.dates.FuzzyDate`.

The :class:`EntityStore` is **derived**: it is fully re-creatable from the ``MentionStore`` by
re-running resolution, so ``entity_id`` is a content hash of the sorted member mention ids (stable
given the same membership).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, Protocol, runtime_checkable
from urllib.parse import quote, unquote

from pydantic import BaseModel, Field

from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.mention import PayloadT
from ragdoc.extraction.query import EntityQuery, filter_entities

_ENTITY_SUFFIX = ".entity.json"


class Entity(BaseModel, Generic[PayloadT]):
    """A canonical, deduplicated entity merged from one or more mentions."""

    entity_id: str = Field(description="Stable content hash of the sorted member mention ids.")
    payload: PayloadT = Field(description="The canonical (merged) payload instance.")
    aliases: list[str] = Field(default_factory=list, description="Distinct surface forms seen across members.")
    member_mention_ids: list[str] = Field(default_factory=list, description="Contributing mention ids (sorted).")
    source_ids: list[str] = Field(default_factory=list, description="All documents this entity originated from.")
    confidence: float = Field(default=1.0, description="Aggregate confidence in this entity.")
    date: FuzzyDate | None = Field(default=None, description="Union of member dates, when the payload has one.")


@runtime_checkable
class EntityStore(Protocol):
    """Async, source-aware sink for canonical entities (Layer 2), written by resolution."""

    async def upsert(self, entities: list[Entity]) -> list[str]:
        """Add or update *entities* (keyed by ``entity_id``); return their ids."""
        ...

    async def delete(self, entity_ids: list[str]) -> None:
        """Remove entities by id (used to drop entities no longer produced by re-resolution)."""
        ...

    async def delete_by_source(self, source_id: str) -> None:
        """Delete entities that originated solely from *source_id* (no-op if absent)."""
        ...

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values contributing to stored entities."""
        ...

    async def list_entities(self) -> list[Entity]:
        """Return every stored canonical entity."""
        ...

    async def query(self, query: EntityQuery) -> list[Entity]:
        """Return entities matching *query* (date-range overlap + exact payload-field matches)."""
        ...


class LocalEntityStore:
    """Filesystem-backed :class:`EntityStore` — one JSON file per entity.

    Args:
        directory: Root directory for the store. Created if it does not exist.
        payload_model: The payload model to rehydrate entities into (e.g. ``Event``).
    """

    def __init__(self, directory: Path | str, payload_model: type[BaseModel]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._payload_model = payload_model

    def _path(self, entity_id: str) -> Path:
        return self._dir / (quote(entity_id, safe="") + _ENTITY_SUFFIX)

    def _iter_files(self):
        return self._dir.glob("*" + _ENTITY_SUFFIX)

    def _entity_type(self) -> type[Entity]:
        payload_model = self._payload_model
        return Entity[payload_model]

    async def upsert(self, entities: list[Entity]) -> list[str]:
        for e in entities:
            self._path(e.entity_id).write_text(e.model_dump_json(indent=2), encoding="utf-8")
        return [e.entity_id for e in entities]

    async def delete(self, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            self._path(eid).unlink(missing_ok=True)

    async def delete_by_source(self, source_id: str) -> None:
        etype = self._entity_type()
        for path in list(self._iter_files()):
            entity = etype.model_validate_json(path.read_text(encoding="utf-8"))
            if entity.source_ids == [source_id]:
                path.unlink(missing_ok=True)

    async def list_source_ids(self) -> set[str]:
        etype = self._entity_type()
        ids: set[str] = set()
        for path in self._iter_files():
            ids.update(etype.model_validate_json(path.read_text(encoding="utf-8")).source_ids)
        return ids

    async def list_entities(self) -> list[Entity]:
        etype = self._entity_type()
        return [etype.model_validate_json(p.read_text(encoding="utf-8")) for p in self._iter_files()]

    async def query(self, query: EntityQuery) -> list[Entity]:
        """Filter all stored entities in memory (loads the store, then applies *query*)."""
        return filter_entities(await self.list_entities(), query)

    @staticmethod
    def _decode_id(path: Path) -> str:
        return unquote(path.name[: -len(_ENTITY_SUFFIX)])
