"""Secondary-file resolvers for :class:`~ragdoc.parsing.multi_source.MultiSourceParser`.

A resolver maps a primary file path to a secondary file path (or ``None`` if
no secondary file is available).  The most common resolver is
:func:`sibling_resolver`, which swaps the file suffix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol


class SecondaryResolver(Protocol):
    """Maps a primary file path to a secondary file path, or ``None``."""

    def __call__(self, path: Path) -> Path | None: ...


def sibling_resolver(suffix: str) -> Callable[[Path], Path | None]:
    """Return a resolver that swaps the file suffix.

    Returns ``None`` if the sibling file does not exist on disk.

    Args:
        suffix: The target suffix (e.g. ``".pdf"``).

    Example::

        resolve = sibling_resolver(".pdf")
        resolve(Path("report.docx"))  # -> Path("report.pdf") if it exists
    """

    def resolve(path: Path) -> Path | None:
        candidate = path.with_suffix(suffix)
        return candidate if candidate.exists() else None

    return resolve
