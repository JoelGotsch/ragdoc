"""LocalDocumentStore: a filesystem-backed :class:`DocumentStore`.

Stores one JSON file per source under a directory (``<quoted source_id>.doc.json``).  Intended
for local development and **human editing** of parsed documents between the parse/process stage
and the chunk/embed stage::

    store = LocalDocumentStore("data/docs/")
    await doc_store_pipeline.run(source_files)   # writes JSON files here
    # ... a human edits data/docs/*.doc.json ...
    await vec_pipeline.run()                      # reads them back, re-chunks edited ones

The store is **index-free**: :meth:`list_source_state` computes each document's
``content_hash`` — cheap, pure Python — from the live file on disk, so manual edits are
detected at Boundary 2 (no stale cached index).

Notes:
    * One Document per ``source_id`` (no versioning).
    * Document JSON uses Pydantic ``model_dump_json``; images are inlined (binary sidecars are
      a future extension).
    * File I/O is synchronous inside ``async`` methods — fine for local dev, not for
      high-concurrency production access.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote, unquote

from ragdoc.document import Document
from ragdoc.pipeline.stores import SourceState

_DOC_SUFFIX = ".doc.json"


class LocalDocumentStore:
    """Filesystem-backed :class:`~ragdoc.pipeline.stores.DocumentStore`.

    Args:
        directory: Root directory for the store.  Created if it does not exist.
    """

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _doc_path(self, source_id: str) -> Path:
        """Filesystem-safe, reversible file path for *source_id*."""
        return self._dir / (quote(source_id, safe="") + _DOC_SUFFIX)

    def _iter_doc_files(self) -> Iterator[Path]:
        return self._dir.glob("*" + _DOC_SUFFIX)

    async def upsert(self, documents: list[Document]) -> list[str]:
        """Write each document to its own JSON file (replacing any existing one).

        Raises:
            ValueError: if any document has no ``source_id`` (the store key).
        """
        written: list[str] = []
        for doc in documents:
            if not doc.source_id:
                raise ValueError(
                    "LocalDocumentStore.upsert requires document.source_id to be set "
                    "(normally done by DocumentStorePipeline before upsert)."
                )
            self._doc_path(doc.source_id).write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            written.append(doc.source_id)
        return written

    async def delete_by_source(self, source_id: str) -> None:
        """Delete the document file for *source_id* (no-op if absent)."""
        self._doc_path(source_id).unlink(missing_ok=True)

    async def get_document(self, source_id: str) -> Document | None:
        """Read and return the document for *source_id*, or ``None`` if absent."""
        path = self._doc_path(source_id)
        if not path.exists():
            return None
        return Document.model_validate_json(path.read_text(encoding="utf-8"))

    async def list_source_ids(self) -> set[str]:
        """Return all stored source_ids (decoded from filenames, no document loads)."""
        return {unquote(p.name[: -len(_DOC_SUFFIX)]) for p in self._iter_doc_files()}

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState`, computing ``content_hash`` from the
        **live** files so manual edits are detected."""
        state: dict[str, SourceState] = {}
        for path in self._iter_doc_files():
            doc: Document = Document.model_validate_json(path.read_text(encoding="utf-8"))
            if doc.source_id:
                state[doc.source_id] = SourceState(
                    source_hash=doc.source_hash or "",
                    content_hash=doc.content_hash(),
                )
        return state
