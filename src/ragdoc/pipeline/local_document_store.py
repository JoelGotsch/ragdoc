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

**The filename is the authoritative source_id.** Files are named ``<quoted source_id>.doc.json``
and every id-keyed read (:meth:`list_source_ids`, :meth:`list_source_state`,
:meth:`get_document`) derives the id by unquoting the filename — never from the JSON-internal
``source_id`` field. A human editing that field therefore cannot desynchronise the store: the
document stays syncable under its filename-derived id, and :meth:`get_document` re-stamps a
disagreeing internal ``source_id`` (with a warning). Rename the *file* to change a source's id.

Notes:
    * One Document per ``source_id`` (no versioning).
    * Document JSON uses Pydantic ``model_dump_json``; images are inlined (binary sidecars are
      a future extension).
    * File reads/writes go through :mod:`aiofiles`, so the store does not block the event loop.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote, unquote

import aiofiles

from ragdoc.document import Document
from ragdoc.pipeline.stores import SourceState

logger = logging.getLogger(__name__)

_DOC_SUFFIX = ".doc.json"


def _decode_source_id(path: Path) -> str:
    """Recover the authoritative ``source_id`` from a store filename (inverse of quoting)."""
    return unquote(path.name[: -len(_DOC_SUFFIX)])


class LocalDocumentStore:
    """Filesystem-backed :class:`~ragdoc.pipeline.stores.DocumentStore`.

    The filename is the authoritative ``source_id`` key (see the module docstring): id
    listings derive ids from filenames, and a JSON-internal ``source_id`` edited to disagree
    is re-stamped on read.

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
            async with aiofiles.open(self._doc_path(doc.source_id), "w", encoding="utf-8") as f:
                await f.write(doc.model_dump_json(indent=2))
            written.append(doc.source_id)
        return written

    async def delete_by_source(self, source_id: str) -> None:
        """Delete the document file for *source_id* (no-op if absent)."""
        self._doc_path(source_id).unlink(missing_ok=True)

    async def get_document(self, source_id: str) -> Document | None:
        """Read and return the document for *source_id*, or ``None`` if absent.

        The filename is authoritative: if a human edited the JSON-internal ``source_id``
        to disagree with the filename-derived id, the document is re-stamped with the
        filename-derived id (and a warning is logged) so it stays syncable.
        """
        path = self._doc_path(source_id)
        if not path.exists():
            return None
        async with aiofiles.open(path, encoding="utf-8") as f:
            content = await f.read()
        doc = Document.model_validate_json(content)
        if doc.source_id != source_id:
            logger.warning(
                f"stored document {path.name} carries source_id {doc.source_id!r}, but the filename "
                f"is authoritative — re-stamping to {source_id!r} (rename the file to change the id)"
            )
            doc.source_id = source_id
        return doc

    async def list_source_ids(self) -> set[str]:
        """Return all stored source_ids (decoded from filenames — the authoritative key —
        with no document loads)."""
        return {_decode_source_id(p) for p in self._iter_doc_files()}

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState`, computing ``content_hash`` from the
        **live** files so manual edits are detected.

        Keys are filename-derived (the authoritative id), so an edited JSON-internal
        ``source_id`` never changes the sync identity (and never orphans live chunks).
        """
        state: dict[str, SourceState] = {}
        for path in self._iter_doc_files():
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()
            doc: Document = Document.model_validate_json(content)
            state[_decode_source_id(path)] = SourceState(
                source_hash=doc.source_hash or "",
                content_hash=doc.content_hash(),
            )
        return state
