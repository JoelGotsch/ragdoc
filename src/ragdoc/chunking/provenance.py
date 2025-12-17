"""Chunk provenance resolution and deterministic chunk-id minting.

One fallback chain (:func:`resolve_chunk_provenance`) is shared by every chunker and by
:meth:`~ragdoc.pipeline.ChunkPipeline.run`, so all chunks of one source carry
identical provenance.  Chunk ids are minted **pipeline-side** by ``ChunkPipeline.run`` over
``(source_id, split_sequence, chunk_ordinal, content_hash)`` — a chunker cannot know its
split's position, so id minting does not live in chunkers.

The default id scheme (:func:`mint_chunk_id`) is deterministic: an unchanged source yields
the same source_id, content_hash, split count/order, and chunk counts (for chunkers with
deterministic chunk counts, e.g. ``SimpleChunker``), hence identical ids across runs —
idempotent vector-store upserts without collisions between identical-content splits.
``LLMChunker`` ids are stable only while the model returns the same number of summaries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ragdoc.document import Document

ChunkIdFn = Callable[[str, int, int, str], str]
"""``(source_id, split_sequence, chunk_ordinal, content_hash) -> chunk id``.

``split_sequence`` is the 1-based position of the split within its source document (from
positional enumeration in ``ChunkPipeline.run``); ``chunk_ordinal`` is the 0-based index of the
chunk within its split (``LLMChunker`` emits several chunks per split).
"""


class ChunkProvenance(BaseModel):
    """Resolved per-document provenance stamped uniformly onto every chunk of one source."""

    source_id: str = Field(description="Sync identity key: document.source_id or document.source_path or document.id.")
    source_hash: str | None = Field(
        default=None,
        description="File-byte SHA-256 from the sync pipeline's hash_fn; None when unknown. Never faked.",
    )
    content_hash: str = Field(description="Canonical document content hash (Document.content_hash()).")


def resolve_chunk_provenance(document: Document) -> ChunkProvenance:
    """Resolve the provenance triple for chunks derived from *document*.

    The single fallback chain used everywhere:

    * ``source_id`` — ``document.source_id or document.source_path or document.id``
      (never None: every document has an id).
    * ``source_hash`` — ``document.source_hash`` verbatim. ``None`` when the document was not
      produced through a sync pipeline; there is **no** content-hash fallback — a fabricated
      file hash would masquerade as file provenance.
    * ``content_hash`` — ``document.content_hash()`` (computed once here).

    Args:
        document: A parsed (and typically processed) document.

    Returns:
        The resolved :class:`ChunkProvenance`.
    """
    return ChunkProvenance(
        source_id=document.source_id or document.source_path or document.id,
        source_hash=document.source_hash,
        content_hash=document.content_hash(),
    )


def mint_chunk_id(source_id: str, split_sequence: int, chunk_ordinal: int, content_hash: str) -> str:
    """Deterministic chunk id over the 4-tuple identity of a chunk.

    SHA-256 hex digest of the four parts joined with ``"\\x1f"`` (the unit separator —
    unambiguous versus any raw string concatenation, so ``("a1", 1, ...)`` and
    ``("a", 11, ...)`` never collide).

    Args:
        source_id: Sync identity key of the source document.
        split_sequence: 1-based split position within the source (positional enumeration).
        chunk_ordinal: 0-based chunk index within the split.
        content_hash: Canonical content hash of the source document.

    Returns:
        64-character lowercase hex string.
    """
    joined = "\x1f".join((source_id, str(split_sequence), str(chunk_ordinal), content_hash))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
