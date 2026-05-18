"""Store protocols for VectorStorePipeline.

:class:`VectorStore` — async protocol for a vector store sink that supports
upsert, delete, and source-aware queries for incremental sync.

The vector store is the **single source of truth** for provenance.  Each chunk
carries ``source_id`` and ``source_hash`` as first-class fields so that the
sync pipeline can detect unchanged sources, clean up stale chunks, and
discover orphans without any local state file.

Implementations should use ``chunk.source_id`` and ``chunk.source_hash``
(the dedicated fields, not ``chunk.metadata``) for their internal indexing.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragdoc.chunking import Chunk


@runtime_checkable
class VectorStore(Protocol):
    """Async protocol for a vector store sink with source-aware operations.

    Implementations must support:

    * **upsert / delete** — basic CRUD by chunk ID.
    * **get_source_hash** — query the stored hash for a given source_id,
      enabling skip-unchanged detection without local state.
    * **delete_by_source** — remove all chunks for a given source_id,
      enabling cleanup by source identity rather than tracked chunk IDs.
    * **list_source_ids** — enumerate all distinct source identifiers,
      enabling orphan detection when sources are removed.

    Implementations should index on ``chunk.source_id`` and
    ``chunk.source_hash`` (first-class fields), not ``chunk.metadata``.

    All major vector stores (Pinecone, Qdrant, Weaviate, ChromaDB, pgvector)
    support the metadata operations required by this protocol.
    """

    async def upsert(self, chunks: list[Chunk]) -> list[str]:
        """Upload or update *chunks* and return their stored IDs.

        Args:
            chunks: Chunk objects to upsert.

        Returns:
            List of IDs in the same order as *chunks*.
        """
        ...

    async def delete(self, ids: list[str]) -> None:
        """Remove chunks by ID.

        Args:
            ids: IDs to delete.  IDs that do not exist should be silently ignored.
        """
        ...

    async def get_source_hash(self, source_id: str) -> str | None:
        """Return the stored hash for *source_id*, or ``None`` if not present.

        The hash is the SHA-256 hex digest stored in ``chunk.source_hash``
        when chunks were last upserted for this source.

        Args:
            source_id: The source identity key (as set by
                ``VectorStorePipeline.source_id_fn``).

        Returns:
            The stored hash string, or ``None`` if no chunks exist for
            *source_id*.
        """
        ...

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks whose ``source_id`` field equals *source_id*.

        Args:
            source_id: The source identity key identifying chunks to remove.
        """
        ...

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values stored in this vector store.

        Used to discover which sources have chunks present, enabling orphan
        detection when sources are removed from the sync list.

        Returns:
            Set of source identity strings.
        """
        ...
