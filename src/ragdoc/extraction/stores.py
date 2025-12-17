"""Store protocols and a local backend for extraction layers.

* :class:`MentionStore` — source-aware sink for raw :class:`~ragdoc.extraction.mention.Mention`
  records (Layer 1). Mirrors the :class:`~ragdoc.pipeline.stores.VectorStore` shape so it
  reuses :class:`~ragdoc.pipeline.stores.SourceState` and the delete-then-upsert sync
  discipline.
* :class:`LocalMentionStore` — filesystem backend (one JSON file per source), index-free, for local
  development.

(The Layer-2 ``EntityStore`` protocol lives alongside the ``Entity`` model in
``ragdoc.extraction.entity``.)

Implementations index on the mention's first-class fields (``source_id`` / ``source_hash`` /
``content_hash``), never on ``metadata``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import quote, unquote

import aiofiles
from pydantic import BaseModel

from ragdoc.extraction.mention import Mention
from ragdoc.pipeline.stores import SourceState

_MENTIONS_SUFFIX = ".mentions.json"


@runtime_checkable
class MentionStore(Protocol):
    """Async, source-aware sink for raw mentions (Layer 1).

    A source's mentions share one ``source_hash`` / ``content_hash`` (uniform per source), so
    :meth:`list_source_state` can report change-detection state per source from any of them.
    """

    async def upsert(self, mentions: list[Mention]) -> list[str]:
        """Add or update *mentions* (keyed by ``mention_id``); return their ids."""
        ...

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all mentions whose ``source_id`` equals *source_id* (no-op if absent)."""
        ...

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values with mentions present."""
        ...

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` for every source with mentions."""
        ...

    async def list_mentions(self, payload_type: type[BaseModel] | None = None) -> list[Mention]:
        """Return stored mentions, optionally filtered to those whose payload is *payload_type*.

        ``payload_type=None`` (default) returns every stored mention — unchanged behaviour. Pass
        a registered node or edge type (e.g. ``Person``) to get only mentions whose payload is
        an instance of that type; filtering uses ``isinstance``, so a single-type store returns
        everything when the type matches, and a multi-type / union-typed store splits cleanly by
        the user's declared types. An unknown type returns ``[]`` (the store doesn't track the
        schema; user error is non-destructive).
        """
        ...


class LocalMentionStore:
    """Filesystem-backed :class:`MentionStore` — one JSON file (a list of mentions) per source.

    Args:
        directory: Root directory for the store. Created if it does not exist.
        payload_model: The payload model to rehydrate mentions into (e.g. ``Event``).
    """

    def __init__(self, directory: Path | str, payload_model: type[BaseModel]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._payload_model = payload_model

    def _path(self, source_id: str) -> Path:
        return self._dir / (quote(source_id, safe="") + _MENTIONS_SUFFIX)

    def _iter_files(self):
        return self._dir.glob("*" + _MENTIONS_SUFFIX)

    def _mention_type(self) -> type[Mention]:
        payload_model = self._payload_model
        return Mention[payload_model]

    async def _load(self, path: Path) -> list[Mention]:
        import json

        async with aiofiles.open(path, encoding="utf-8") as f:
            raw = json.loads(await f.read())
        mtype = self._mention_type()
        return [mtype.model_validate(d) for d in raw]

    async def _write(self, source_id: str, mentions: list[Mention]) -> None:
        import json

        payload = [m.model_dump(mode="json") for m in mentions]
        async with aiofiles.open(self._path(source_id), "w", encoding="utf-8") as f:
            await f.write(json.dumps(payload, indent=2))

    async def upsert(self, mentions: list[Mention]) -> list[str]:
        """Merge *mentions* into their per-source files (update existing ids, add new ones)."""
        by_source: dict[str, dict[str, Mention]] = {}
        for m in mentions:
            by_source.setdefault(m.source_id, {})[m.mention_id] = m
        written: list[str] = []
        for source_id, incoming in by_source.items():
            existing = (
                {m.mention_id: m for m in await self._load(self._path(source_id))}
                if self._path(source_id).exists()
                else {}
            )
            existing.update(incoming)
            await self._write(source_id, list(existing.values()))
            written.extend(m.mention_id for m in incoming.values())
        return written

    async def delete_by_source(self, source_id: str) -> None:
        self._path(source_id).unlink(missing_ok=True)

    async def list_source_ids(self) -> set[str]:
        return {unquote(p.name[: -len(_MENTIONS_SUFFIX)]) for p in self._iter_files()}

    async def list_source_state(self) -> dict[str, SourceState]:
        state: dict[str, SourceState] = {}
        for path in self._iter_files():
            mentions = await self._load(path)
            if mentions:
                first = mentions[0]
                state[first.source_id] = SourceState(source_hash=first.source_hash, content_hash=first.content_hash)
        return state

    async def list_mentions(self, payload_type: type[BaseModel] | None = None) -> list[Mention]:
        all_mentions = [m for path in self._iter_files() for m in await self._load(path)]
        if payload_type is None:
            return all_mentions
        return [m for m in all_mentions if isinstance(m.payload, payload_type)]
