"""MultiSourceParser — run two parsers and merge the results.

Satisfies the :class:`~ragdoc.parsing.parser.Parser` base class so it can be
registered in the parser registry like any other parser.

Two use cases:

- **Two parsing strategies on the same file** (``secondary_resolver=None``):
  both parsers receive the same path.
- **Sibling files** (``secondary_resolver`` provided): the primary parser gets
  the input path; the secondary parser gets the resolved path.  If the resolver
  returns ``None``, the primary document is returned unchanged.

Customize merge behavior via ``functools.partial`` on
:func:`~ragdoc.merging.merge.merge_documents`::

    from functools import partial
    from ragdoc.document import ElementTypeEnum
    from ragdoc.merging import merge_documents

    my_merge = partial(merge_documents, prefer_source={ElementTypeEnum.TABLE: "b"})
    MultiSourceParser(primary=..., secondary=..., merge=my_merge)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from pydantic import ConfigDict, Field, model_validator

from ragdoc.document import Document
from ragdoc.parsing.parser import Parser

logger = logging.getLogger(__name__)

MergeFn = Callable[[Document, Document], Document]
SecondaryResolverFn = Callable[[Path], "Path | None"]


class MultiSourceParser(Parser):
    """Parser that runs two parsers and merges the results.

    ``name`` and ``patterns`` default to ``"multi_source"`` and ``[]``
    respectively, which is appropriate when using the instance purely as a
    callable combinator without registering it.  Supply explicit values when
    registering::

        merger = MultiSourceParser(
            primary=get_parser("mineru"),
            secondary=get_parser("html"),
            secondary_resolver=sibling_resolver(".html"),
            name="mineru_html",
            patterns=["_middle.json"],
            priority=60,
        )
        register_parser(merger)

    Args:
        primary: The primary parser.
        secondary: The secondary parser.
        secondary_resolver: Optional resolver that maps the primary path to the
            secondary file path.  When ``None``, both parsers receive the same
            path.
        merge: Merge function ``(doc_a, doc_b) -> Document``.  Defaults to
            :func:`~ragdoc.merging.merge.merge_documents`.
        name: Registry name.  Defaults to ``"multi_source"``.
        patterns: Suffix patterns to register under.  Defaults to ``[]``.
        priority: Registry priority.  Defaults to ``0``.
        description: Human-readable description.  Auto-derived from primary and
            secondary descriptions if not supplied.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Override base-class required fields with defaults suitable for combinator use
    name: str = Field(default="multi_source", description="Unique registry key")
    patterns: list[str] = Field(default_factory=list, description="Suffix patterns to register under")

    primary: Parser = Field(description="Primary parser")
    secondary: Parser = Field(description="Secondary parser")
    secondary_resolver: SecondaryResolverFn | None = Field(
        default=None,
        description=(
            "Maps primary path to secondary file path. "
            "None = both parsers receive the same path."
        ),
    )
    merge: MergeFn | None = Field(
        default=None,
        description="Merge function (doc_a, doc_b) -> Document. Defaults to merge_documents.",
    )

    @model_validator(mode="after")
    def _derive_description(self) -> "MultiSourceParser":
        if not self.description:
            self.description = (
                f"Merge of {self.primary.description} + {self.secondary.description}"
            )
        return self

    async def __call__(self, path: Path) -> Document:
        logger.debug(f"MultiSourceParser: parsing primary {path.name}")
        primary_doc = await self.primary(path)

        secondary_path = self.secondary_resolver(path) if self.secondary_resolver else path
        if secondary_path is None:
            logger.debug("MultiSourceParser: secondary path resolved to None, returning primary only")
            return primary_doc

        logger.debug(f"MultiSourceParser: parsing secondary {secondary_path.name}")
        secondary_doc = await self.secondary(secondary_path)

        merge_fn = self.merge
        if merge_fn is None:
            from ragdoc.merging import merge_documents

            merge_fn = merge_documents

        logger.debug("MultiSourceParser: merging primary + secondary")
        return merge_fn(primary_doc, secondary_doc)
