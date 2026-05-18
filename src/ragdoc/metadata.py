"""
Canonical metadata types and serialization for ragdoc models.

All three metadata-bearing models (Document, BaseElement, Chunk) use
MetadataDict as their field type, and serialize_metadata_value as the
canonical serializer.

Supported metadata value types
--------------------------------
  Scalars  : str, int, float, bool, None
  Collections: list[MetadataValue], dict[str, MetadataValue]
  Pydantic : BaseModel subclasses (serialized via model_dump(mode='json'))

NOT supported
-------------
  datetime, date, time  – convert to ISO 8601 str: dt.isoformat()
  UUID                  – convert to str: str(uid)
  Path                  – convert to str: str(path)
  Enum                  – use .value or wrap in a BaseModel
  bytes                 – not JSON-serializable
  BaseModel nested inside a plain dict/list value – wrap in a parent BaseModel

Consumer notes
--------------
  Renderer (metadata_keys):
      All types rendered as their JSON representation.
      BaseModel → JSON object string. list → JSON array string. Scalar → value.

  Visualizer exporter (export_chunks_to_db):
      str, int, float, bool → native SQLite column types.
      dict, list, BaseModel → stored as JSON TEXT; not available as filter facets.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar, Union

from pydantic import BaseModel
from typing_extensions import Required, TypedDict

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

# Recursive type alias using Any for nested collections to avoid Pydantic
# forward-reference resolution errors. The full recursive constraint
# (list[MetadataValue] / dict[str, MetadataValue]) is enforced at runtime
# by validate_metadata_dict via json.dumps().
MetadataValue = Union[str, int, float, bool, None, list[Any], dict[str, Any], BaseModel]
"""
Canonical type for a single metadata value.

Covers all JSON-serializable scalars, recursively nested lists and dicts,
and Pydantic BaseModel instances (serialized via model_dump(mode='json')).

list and dict contents may themselves be MetadataValue (recursively), but this
is enforced at runtime by validate_metadata_dict rather than statically, to
avoid Pydantic forward-reference resolution issues.

Note: BaseModel instances nested *inside* a plain dict or list are not
auto-serialized — use a wrapper BaseModel or convert to a plain dict first.
"""

MetadataDict = dict[str, MetadataValue]
"""
Canonical type for a metadata mapping. Used as the field type on Document,
BaseElement, and Chunk.

Keys must be strings. Values must be MetadataValue (JSON-serializable).
"""


class BaseMetadata(TypedDict, total=False):
    """Metadata keys that every ragdoc parser sets by convention.

    ``filename`` is required (set by all parsers via ``path.name``) while all
    other keys declared in subclasses default to optional (``total=False``).

    Projects extend this TypedDict to declare their own keys::

        class MyMetadata(ragdoc.BaseMetadata, total=False):
            document_name: str | None
            document_date: str | None

    Use ``Required[T]`` (from ``typing_extensions``) to mark a key required in
    your subclass — the runtime check in ``DocumentPipeline`` will enforce it::

        from typing import Annotated
        from typing_extensions import Required

        class MyMetadata(ragdoc.BaseMetadata, total=False):
            document_name: Required[str]   # processor must set this
            document_date: str | None      # optional

    Attributes:
        filename: Set by all parsers: ``path.name``.
        split_sequence: 1-based reading-order position of this document in
            the split list returned by ``split_document()``.  ``1`` for a document
            that was not split (split_total is also ``1``).
        split_total: Total number of splits produced from the same source
            document by ``split_document()``.  Together with ``split_sequence``,
            this tells a retrieval client whether the chunk has predecessors
            (``split_sequence > 1``) or successors
            (``split_sequence < split_total``).
    """

    filename: Required[str]  # set by all parsers: path.name
    split_sequence: int | None  # set by split_document: 1-based position in the split list
    split_total: int | None  # set by split_document: total number of splits from the same source


TMetadata = TypeVar("TMetadata", bound=BaseMetadata)
"""TypeVar for user-defined metadata TypedDicts that extend BaseMetadata.

Used to parametrise ``Chunk[TMetadata]``, ``Document[TMetadata]``,
``DocumentPipeline[TMetadata]``, and ``VectorStorePipeline[TMetadata]``
so that metadata keys are typed end-to-end.
"""


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------


def metadata_json_schema(metadata_type: type) -> dict:  # type: ignore[type-arg]
    """Return a JSON Schema dict for *metadata_type*, suitable for LLM self-query prompts.

    Uses ``pydantic.TypeAdapter`` to generate the schema, so all standard
    Pydantic schema customisation applies.  ``Annotated`` extras that Pydantic
    does not recognise (e.g. ``QdrantIndex``) are silently ignored.

    The result is a plain ``dict`` ready to be ``json.dumps()``-ed into a
    system prompt.

    Args:
        metadata_type: A TypedDict class (typically a subclass of
            :class:`BaseMetadata`) or any type accepted by
            ``pydantic.TypeAdapter``.

    Returns:
        JSON Schema dict.

    Example:
        ```python
        import ragdoc

        class MyMetadata(ragdoc.BaseMetadata, total=False):
            document_name: str | None
            document_date: str | None

        schema = ragdoc.metadata_json_schema(MyMetadata)
        # → {"properties": {"filename": {...}, "document_name": {...}, ...}, ...}
        system_prompt = f"Filter by these fields: {json.dumps(schema)}"
        ```
    """
    from pydantic import TypeAdapter

    return TypeAdapter(metadata_type).json_schema()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_metadata_value(val: MetadataValue) -> object:
    """Convert a MetadataValue to a JSON-serializable primitive.

    Conversion rules:
      str / int / float / bool / None → returned as-is
      BaseModel  → model_dump(mode='json')   (full recursive Pydantic serialization)
      dict       → {k: serialize_metadata_value(v) for ...}
      list       → [serialize_metadata_value(v) for ...]

    Returns a plain Python object (str, int, float, bool, None, dict, list)
    that can be passed directly to json.dumps().
    """
    if isinstance(val, BaseModel):
        return val.model_dump(mode="json")
    if isinstance(val, dict):
        return {k: serialize_metadata_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [serialize_metadata_value(v) for v in val]
    return val  # str, int, float, bool, None — already JSON-serializable


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_metadata_dict(metadata: dict) -> None:  # type: ignore[type-arg]
    """Raise ValueError if any metadata value is not a supported MetadataValue.

    Validates by attempting a full json.dumps() of the serialized metadata,
    catching non-serializable values early (at model construction time)
    rather than silently at render or export time.

    Raises:
        ValueError: if any value cannot be JSON-serialized, with a message
            describing the problem and listing supported types.
    """
    try:
        serialized = {k: serialize_metadata_value(v) for k, v in metadata.items()}
        json.dumps(serialized)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata contains a value that cannot be JSON-serialized: {exc}. "
            "Supported types: str, int, float, bool, None, list, dict, pydantic.BaseModel. "
            "For datetime/UUID/Path, convert to str first (e.g. dt.isoformat())."
        ) from exc
