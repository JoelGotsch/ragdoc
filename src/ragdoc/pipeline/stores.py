"""Store protocols for the sync pipelines.

:class:`VectorStore` — async protocol for a vector store sink (Boundary 2 / direct path).
:class:`DocumentStore` — async protocol for an intermediate Document store (Boundary 1).

The store is the **single source of truth** for provenance. Each chunk carries
``source_id`` and ``source_hash`` as first-class fields so the sync pipeline can detect
unchanged sources, clean up stale chunks, and discover orphans without any local state file.

Implementations should use ``chunk.source_id`` / ``chunk.source_hash`` (the dedicated
fields, not ``chunk.metadata``) for their internal indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ragdoc.chunking import Chunk

if TYPE_CHECKING:
    from ragdoc.document import Document


@dataclass(frozen=True)
class SourceState:
    """The change-detection state stored for a single source.

    Returned in bulk by :meth:`VectorStore.list_source_state` /
    :meth:`DocumentStore.list_source_state` so a sync ``plan()`` can read every
    source's hashes in one round-trip and load full payloads only for sources
    that actually changed.

    Attributes:
        source_hash: SHA-256 of the original source file bytes (Boundary-1 /
            direct-path change detection).
        content_hash: Renderer-stable Document content hash
            (``Document.content_hash()``) used for Boundary-2
            (DocumentStore -> VectorStore) change detection.  ``None`` is treated
            as "always changed".
    """

    source_hash: str
    content_hash: str | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Async protocol for a vector store sink with source-aware operations.

    Implementations must support:

    * **upsert / delete** — basic CRUD by chunk ID.
    * **delete_by_source** — remove all chunks for a given source_id,
      enabling cleanup by source identity rather than tracked chunk IDs.
    * **list_source_ids** — enumerate all distinct source identifiers
      (consumer-facing; the sync engine itself uses ``list_source_state``).
    * **list_source_state** — bulk-read every source's change-detection hashes
      (``source_hash`` + ``content_hash``) in one round-trip.

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

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks whose ``source_id`` field equals *source_id*.

        Args:
            source_id: The source identity key identifying chunks to remove.
        """
        ...

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values stored in this vector store.

        Not used by the sync engine (which reads :meth:`list_source_state`);
        provided for consumers doing their own orphan/reporting logic.

        Returns:
            Set of source identity strings.
        """
        ...

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return the change-detection state for every known source, in one call.

        Lets a sync ``plan()`` compare all sources without a per-source query.
        A source absent from the result is treated as new; a present source
        whose :attr:`SourceState.content_hash` is ``None`` is treated as
        "always changed" by Boundary-2 detection.

        Returns:
            Mapping of ``source_id`` to :class:`SourceState`.
        """
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Async protocol for an intermediate Document store (Boundary 1 sink).

    Sits between parsing and chunking: a :class:`~ragdoc.pipeline.DocumentStorePipeline`
    writes parsed/processed :class:`~ragdoc.document.Document` objects here, and a
    :class:`~ragdoc.pipeline.vectorstore.VectorStorePipeline` reads them back to chunk
    and embed.  Stores one Document per ``source_id`` (no versioning).

    Implementations index on ``document.source_id`` / ``document.source_hash`` and
    persist ``document.content_hash()`` so :meth:`list_source_state` is cheap
    (no need to re-render every document).
    """

    async def upsert(self, documents: list[Document]) -> list[str]:
        """Store or replace *documents* (keyed by ``source_id``); return their source_ids."""
        ...

    async def delete_by_source(self, source_id: str) -> None:
        """Delete the document stored for *source_id* (no-op if absent)."""
        ...

    async def get_document(self, source_id: str) -> Document | None:
        """Return the single Document stored for *source_id*, or ``None`` if absent."""
        ...

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values stored here.

        Not used by the sync engine (which reads :meth:`list_source_state`);
        provided for consumers doing their own orphan/reporting logic.
        """
        ...

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` for every stored document."""
        ...
