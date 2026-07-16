"""Document store adapters for ragdoc's DocumentStore protocol.

Each adapter is an optional integration — install the corresponding extra to use it:

* ``ragdoc[qdrant]`` — :class:`~ragdoc.integrations.document_stores.qdrant.QdrantDocumentStore`

For local development / single-machine use, see
:class:`~ragdoc.pipeline.local_document_store.LocalDocumentStore` (no extra required).
"""

from __future__ import annotations

try:
    from ragdoc.integrations.document_stores.qdrant import (
        DocumentTooLargeError,
        QdrantDocumentStore,
    )

    __all__ = ["DocumentTooLargeError", "QdrantDocumentStore"]
except ImportError:
    __all__ = []
