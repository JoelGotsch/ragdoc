"""Mention store adapters for ragdoc's MentionStore protocol.

* ``ragdoc[qdrant]`` — :class:`~ragdoc.integrations.mention_stores.qdrant.QdrantMentionStore`

For local development, see :class:`~ragdoc.extraction.stores.LocalMentionStore` (no extra).
"""

from __future__ import annotations

try:
    from ragdoc.integrations.mention_stores.qdrant import QdrantMentionStore

    __all__ = ["QdrantMentionStore"]
except ImportError:
    __all__ = []
