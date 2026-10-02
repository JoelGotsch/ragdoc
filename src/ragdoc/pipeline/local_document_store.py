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

**Writes are atomic.** A document is written to a temporary file in the same directory,
flushed to disk, and moved into place with ``os.replace`` — a crash mid-write leaves either
the previous file or the new one, never a truncated JSON.

**An unreadable file never poisons the rest.** A file that cannot be read or validated (a
bad hand edit, a file truncated by an older non-atomic write) is logged and reported by
:meth:`list_source_state` with blank hashes — "always changed" — instead of raising:

* it stays listed, so it is **not** an orphan (its chunks are not deleted downstream);
* a Boundary-1 re-ingest of its source sees "changed" and overwrites it;
* a Boundary-2 run tries to load it, :meth:`get_document` raises, and the sync engine records
  that one source in ``UpdateResult.errors`` while every other document syncs normally.

Notes:
    * One Document per ``source_id`` (no versioning).
    * Document JSON uses Pydantic ``model_dump_json``; images are inlined (binary sidecars are
      a future extension).
    * File reads/writes go through :mod:`aiofiles`, so the store does not block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote, unquote

import aiofiles

from ragdoc.document import Document
from ragdoc.pipeline.stores import SourceState

logger = logging.getLogger(__name__)

_DOC_SUFFIX = ".doc.json"


_TMP_SUFFIX = ".tmp"
# Read/parse failures that mark ONE stored file as unreadable. pydantic's ValidationError and
# UnicodeDecodeError are both ValueError subclasses; OSError covers permission / I/O errors.
_UNREADABLE = (ValueError, OSError)
_UNREADABLE_STATE = SourceState(source_hash="", content_hash=None)


async def write_text_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file in the same directory + ``os.replace``).

    The temp file is fsynced before the rename, so after a crash *path* holds either its
    previous content or the complete new content. The temp file is removed if the write fails.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}{_TMP_SUFFIX}")
    try:
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(text)
            await f.flush()
            await asyncio.to_thread(os.fsync, f.fileno())
        await asyncio.to_thread(os.replace, tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
        """Write each document to its own JSON file, atomically replacing any existing one.

        Each file is replaced in one step (see :func:`write_text_atomic`), so a crash never
        leaves a truncated document; an unreadable file already at that path is overwritten.

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
            await write_text_atomic(self._doc_path(doc.source_id), doc.model_dump_json(indent=2))
            written.append(doc.source_id)
        return written

    async def replace_source(self, source_id: str, documents: list[Document]) -> list[str]:
        """Replace the document stored for *source_id* in place (no delete-first window).

        Implements :class:`~ragdoc.pipeline.sync.SourceReplaceStore`: one file per source,
        replaced atomically, so a failed write leaves the previous document intact.

        Raises:
            ValueError: if *documents* is not exactly one document carrying *source_id*.
        """
        if len(documents) != 1 or documents[0].source_id != source_id:
            raise ValueError(f"replace_source({source_id!r}) requires exactly one document with that source_id.")
        return await self.upsert(documents)

    async def delete_by_source(self, source_id: str) -> None:
        """Delete the document file for *source_id* (no-op if absent)."""
        self._doc_path(source_id).unlink(missing_ok=True)

    async def get_document(self, source_id: str) -> Document | None:
        """Read and return the document for *source_id*, or ``None`` if absent.

        The filename is authoritative: if a human edited the JSON-internal ``source_id``
        to disagree with the filename-derived id, the document is re-stamped with the
        filename-derived id (and a warning is logged) so it stays syncable.

        Raises:
            pydantic.ValidationError: if the stored file is not a valid Document (bad manual
                edit, truncated file). ``UnicodeDecodeError`` / ``OSError`` if it cannot be
                read. The sync engine records these per source in ``UpdateResult.errors``.
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

        Per-file isolation: a file that cannot be read or validated is logged and reported
        with blank hashes (``source_hash=""``, ``content_hash=None`` — "always changed")
        instead of aborting the listing. See the module docstring for what that means at
        each boundary.
        """
        state: dict[str, SourceState] = {}
        for path in self._iter_doc_files():
            source_id = _decode_source_id(path)
            try:
                async with aiofiles.open(path, encoding="utf-8") as f:
                    content = await f.read()
                doc: Document = Document.model_validate_json(content)
                state[source_id] = SourceState(source_hash=doc.source_hash or "", content_hash=doc.content_hash())
            except FileNotFoundError:
                continue  # deleted between listing and reading — it is simply gone
            except _UNREADABLE as exc:
                logger.error(
                    f"stored document {path.name} (source_id={source_id!r}) is unreadable: {exc!r}. "
                    "Reporting it as 'always changed'; fix the file or re-ingest its source to overwrite it."
                )
                state[source_id] = _UNREADABLE_STATE
        return state
