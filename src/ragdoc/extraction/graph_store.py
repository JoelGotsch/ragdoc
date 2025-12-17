"""GraphStore Protocol + LocalGraphStore — backend-agnostic typed-KG sink.

The :class:`GraphStore` Protocol gives the resolved entities (nodes + edges) a uniform sink
shape. Implementations are free in their physical storage: :class:`LocalGraphStore` is the
default — JSON-per-entity on disk, two subdirectories (``nodes/`` and ``edges/``), mirrors
:class:`~ragdoc.extraction.entity.LocalEntityStore`. Backend adapters
(``Neo4jGraphStore``, ``PostgresGraphStore``) land as drop-in implementations under
``integrations/`` per the design's TODO entries — no churn in the layer above.

Edges are first-class entities (``Entity[EdgeT]``) with their own ``refs: EdgeRef`` payload; the
``refs`` carry the **entity ids** of the two endpoint nodes (rewritten at resolution time, see
:func:`~ragdoc.extraction.kg_resolution._rewrite_edge_endpoints`). Reverse adjacency
(``neighbors``) is derived at query time by walking the edge set — no write-time bidirectional
bookkeeping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import quote, unquote

from pydantic import BaseModel

from ragdoc.extraction.entity import Entity
from ragdoc.extraction.schema import GraphSchema, build_edge_union, build_node_union

if TYPE_CHECKING:
    pass


_ENTITY_SUFFIX = ".entity.json"


@runtime_checkable
class GraphStore(Protocol):
    """Async, source-aware sink for resolved node + edge entities."""

    async def upsert_nodes(self, entities: list[Entity]) -> list[str]:
        """Add or replace *entities* in the node store; return their ids."""
        ...

    async def upsert_edges(self, entities: list[Entity]) -> list[str]:
        """Add or replace *entities* in the edge store; return their ids."""
        ...

    async def get_node(self, entity_id: str) -> Entity | None:
        """Return the node entity for *entity_id*, or ``None`` if absent."""
        ...

    async def get_edge(self, entity_id: str) -> Entity | None:
        """Return the edge entity for *entity_id*, or ``None`` if absent."""
        ...

    async def delete(self, entity_ids: list[str]) -> None:
        """Remove entities (nodes or edges) by id (silent on unknown)."""
        ...

    async def delete_by_source(self, source_id: str) -> None:
        """Delete every node and edge whose ``source_ids == [source_id]`` (originated solely there).

        Mirrors :meth:`LocalEntityStore.delete_by_source` — multi-source entities are preserved.
        """
        ...

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values contributing to stored entities (nodes + edges)."""
        ...

    async def list_nodes(self, payload_type: type[BaseModel] | None = None) -> list[Entity]:
        """Return stored node entities, optionally filtered by ``isinstance(payload, payload_type)``."""
        ...

    async def list_edges(self, payload_type: type[BaseModel] | None = None) -> list[Entity]:
        """Return stored edge entities, optionally filtered by ``isinstance(payload, payload_type)``."""
        ...

    async def neighbors(self, entity_id: str, edge_types: list[str] | None = None) -> list[Entity]:
        """Return node entities reachable from *entity_id* via any (or *edge_types*-filtered) edge.

        Walks the edge set in-memory; both directions are followed (incoming + outgoing edges).
        ``edge_types`` filters by the edge payload's ``kind`` discriminator value.
        """
        ...


# ---------------------------------------------------------------------------
# LocalGraphStore — JSON-per-entity, two subdirectories
# ---------------------------------------------------------------------------


class LocalGraphStore:
    """Filesystem-backed :class:`GraphStore` — one JSON file per entity, split into
    ``nodes/`` and ``edges/`` subdirectories under *directory*.

    Args:
        directory: Root directory. Created (with subdirectories) if absent.
        schema: :class:`~ragdoc.extraction.schema.GraphSchema` used to rehydrate
            node + edge payloads from JSON into their typed Pydantic models.
    """

    def __init__(self, directory: Path | str, schema: GraphSchema) -> None:
        self._dir = Path(directory)
        self._nodes_dir = self._dir / "nodes"
        self._edges_dir = self._dir / "edges"
        self._nodes_dir.mkdir(parents=True, exist_ok=True)
        self._edges_dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema

    # ---- entity-type construction ----

    def _node_entity_type(self) -> type[Entity]:
        node_union = build_node_union(self._schema.node_types)
        return Entity[node_union]

    def _edge_entity_type(self) -> type[Entity]:
        edge_union = build_edge_union(self._schema.edge_types)
        return Entity[edge_union]

    # ---- path helpers ----

    def _node_path(self, entity_id: str) -> Path:
        return self._nodes_dir / (quote(entity_id, safe="") + _ENTITY_SUFFIX)

    def _edge_path(self, entity_id: str) -> Path:
        return self._edges_dir / (quote(entity_id, safe="") + _ENTITY_SUFFIX)

    def _iter_node_files(self):
        return self._nodes_dir.glob("*" + _ENTITY_SUFFIX)

    def _iter_edge_files(self):
        return self._edges_dir.glob("*" + _ENTITY_SUFFIX)

    # ---- node operations ----

    async def upsert_nodes(self, entities: list[Entity]) -> list[str]:
        for e in entities:
            self._node_path(e.entity_id).write_text(e.model_dump_json(indent=2), encoding="utf-8")
        return [e.entity_id for e in entities]

    async def get_node(self, entity_id: str) -> Entity | None:
        path = self._node_path(entity_id)
        if not path.exists():
            return None
        return self._node_entity_type().model_validate_json(path.read_text(encoding="utf-8"))

    async def list_nodes(self, payload_type: type[BaseModel] | None = None) -> list[Entity]:
        etype = self._node_entity_type()
        out = [etype.model_validate_json(p.read_text(encoding="utf-8")) for p in self._iter_node_files()]
        if payload_type is None:
            return out
        return [e for e in out if isinstance(e.payload, payload_type)]

    # ---- edge operations ----

    async def upsert_edges(self, entities: list[Entity]) -> list[str]:
        for e in entities:
            self._edge_path(e.entity_id).write_text(e.model_dump_json(indent=2), encoding="utf-8")
        return [e.entity_id for e in entities]

    async def get_edge(self, entity_id: str) -> Entity | None:
        path = self._edge_path(entity_id)
        if not path.exists():
            return None
        return self._edge_entity_type().model_validate_json(path.read_text(encoding="utf-8"))

    async def list_edges(self, payload_type: type[BaseModel] | None = None) -> list[Entity]:
        etype = self._edge_entity_type()
        out = [etype.model_validate_json(p.read_text(encoding="utf-8")) for p in self._iter_edge_files()]
        if payload_type is None:
            return out
        return [e for e in out if isinstance(e.payload, payload_type)]

    # ---- combined deletes / queries ----

    async def delete(self, entity_ids: list[str]) -> None:
        ids = set(entity_ids)
        for eid in ids:
            self._node_path(eid).unlink(missing_ok=True)
            self._edge_path(eid).unlink(missing_ok=True)

    async def delete_by_source(self, source_id: str) -> None:
        node_etype = self._node_entity_type()
        for path in list(self._iter_node_files()):
            entity = node_etype.model_validate_json(path.read_text(encoding="utf-8"))
            if entity.source_ids == [source_id]:
                path.unlink(missing_ok=True)
        edge_etype = self._edge_entity_type()
        for path in list(self._iter_edge_files()):
            entity = edge_etype.model_validate_json(path.read_text(encoding="utf-8"))
            if entity.source_ids == [source_id]:
                path.unlink(missing_ok=True)

    async def list_source_ids(self) -> set[str]:
        out: set[str] = set()
        for e in await self.list_nodes():
            out.update(e.source_ids)
        for e in await self.list_edges():
            out.update(e.source_ids)
        return out

    async def neighbors(self, entity_id: str, edge_types: list[str] | None = None) -> list[Entity]:
        edges = await self.list_edges()
        edge_type_set = set(edge_types) if edge_types is not None else None

        other_ids: set[str] = set()
        for e in edges:
            if edge_type_set is not None and e.payload.kind not in edge_type_set:
                continue
            src = e.payload.refs.source_mention_id
            tgt = e.payload.refs.target_mention_id
            if src == entity_id:
                other_ids.add(tgt)
            elif tgt == entity_id:
                other_ids.add(src)

        # Fetch the actual node entities — preserves typed payloads
        result: list[Entity] = []
        for other_id in other_ids:
            node = await self.get_node(other_id)
            if node is not None:
                result.append(node)
        return result

    @staticmethod
    def _decode_id(path: Path) -> str:
        """Decode a path back to the entity_id stored in its filename."""
        return unquote(path.name[: -len(_ENTITY_SUFFIX)])
