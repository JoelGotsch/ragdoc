"""GraphSchema, EdgeRef, and discriminated-union helpers for the typed knowledge graph.

A :class:`GraphSchema` is a **closed declaration** of the node types, edge types, and the legal
``(SourceNode, EdgeType, TargetNode)`` patterns between them. This is the only mechanism that lets
the LLM enforce ontology compliance via structured outputs without every (subject, predicate,
object) combination exploding the schema — the same approach LlamaIndex's
``SchemaLLMPathExtractor.kg_validation_schema`` and neo4j-graphrag's ``SimpleKGPipeline.patterns``
converged on.

Each node and edge type is a Pydantic ``BaseModel``. The first field must be a
``kind: Literal["..."]`` discriminator (OpenAI structured-output strict mode requirement). Edge
types must additionally carry ``refs: EdgeRef`` — the chunk-local endpoint references the LLM
emits, rewritten to real ``mention_id``\\ s at envelope-wrap time and then to ``entity_id``\\ s
after node resolution.

Union sizes are bounded by the ``KG_MAX_UNION_SIZE`` environment variable (default ``10``).
Above this, OpenAI strict-mode reliability dips and schema size pushes prompt tokens; the
processor advises splitting into multiple processors with smaller schemas.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Default per the design — 10 variants per discriminated union for OpenAI strict-mode reliability.
_DEFAULT_MAX_UNION_SIZE = 10


class EdgeRef(BaseModel):
    """Endpoint references for an edge mention.

    The two ``str`` fields are **chunk-local ids** at extraction time (e.g. ``"n0"``, ``"n1"`` —
    the LLM connects nodes and edges via these ids inside one ``.parse()`` call). At
    envelope-wrap time the processor rewrites them to real ``mention_id``\\ s. After node
    resolution, a parallel pass rewrites them again to ``entity_id``\\ s. All three lifecycle
    stages share the same field shape on purpose — no separate types — so the migration is a
    pure relabel, not a type change.
    """

    source_mention_id: str = Field(
        description="The source endpoint id (chunk-local at extraction; entity id after resolution)."
    )
    target_mention_id: str = Field(
        description="The target endpoint id (chunk-local at extraction; entity id after resolution)."
    )


class GraphSchema(BaseModel):
    """A closed schema declaring node types, edge types, and legal patterns between them.

    Configure via the fields below (see each ``Field`` description).

    Raises:
        ValueError: on any registration mismatch — unknown pattern types, missing/wrongly-typed
            ``kind`` discriminator, edge type without ``refs: EdgeRef``, union size above
            ``KG_MAX_UNION_SIZE``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_types: tuple[type[BaseModel], ...] = Field(
        description="Registered node-entity models; each must declare a `kind: Literal[...]` discriminator first."
    )
    edge_types: tuple[type[BaseModel], ...] = Field(
        description=(
            "Registered relationship models; each must declare a `kind: Literal[...]` discriminator first "
            "and carry a `refs: EdgeRef` field for its endpoints."
        )
    )
    patterns: tuple[tuple[type[BaseModel], type[BaseModel], type[BaseModel]], ...] = Field(
        description="Legal (source_node, edge, target_node) triples; every type must be registered above."
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_to_tuples(cls, data: Any) -> Any:
        """Accept lists for user convenience; store as tuples (hashable / immutable)."""
        if isinstance(data, dict):
            for key in ("node_types", "edge_types"):
                if key in data and isinstance(data[key], list):
                    data[key] = tuple(data[key])
            if "patterns" in data and isinstance(data["patterns"], list):
                data["patterns"] = tuple(tuple(p) for p in data["patterns"])
        return data

    @model_validator(mode="after")
    def _validate_schema(self) -> GraphSchema:
        max_size = _max_union_size()
        if len(self.node_types) > max_size:
            raise ValueError(
                f"node_types has {len(self.node_types)} entries, above KG_MAX_UNION_SIZE={max_size}. "
                f"OpenAI structured-output reliability drops with large discriminated unions; split "
                f"this schema across multiple KnowledgeGraphProcessors, or raise the limit."
            )
        if len(self.edge_types) > max_size:
            raise ValueError(
                f"edge_types has {len(self.edge_types)} entries, above KG_MAX_UNION_SIZE={max_size}. "
                f"OpenAI structured-output reliability drops with large discriminated unions; split "
                f"this schema across multiple KnowledgeGraphProcessors, or raise the limit."
            )

        for t in self.node_types:
            _assert_kind_discriminator_first(t)
        for t in self.edge_types:
            _assert_kind_discriminator_first(t)
            _assert_edge_carries_edgeref(t)

        for src, edge, tgt in self.patterns:
            if src not in self.node_types:
                raise ValueError(
                    f"Pattern references unknown node type {src.__name__!r} (not in node_types). "
                    f"Registered node_types: {[t.__name__ for t in self.node_types]}."
                )
            if tgt not in self.node_types:
                raise ValueError(
                    f"Pattern references unknown node type {tgt.__name__!r} (not in node_types). "
                    f"Registered node_types: {[t.__name__ for t in self.node_types]}."
                )
            if edge not in self.edge_types:
                raise ValueError(
                    f"Pattern references unknown edge type {edge.__name__!r} (not in edge_types). "
                    f"Registered edge_types: {[t.__name__ for t in self.edge_types]}."
                )
        return self


# ---------------------------------------------------------------------------
# Discriminated-union builders
# ---------------------------------------------------------------------------


def build_node_union(types: tuple[type[BaseModel], ...] | list[type[BaseModel]]) -> Any:
    """Build a discriminated union over *types*, keyed on the ``kind`` field.

    Single-type input returns the type unchanged (avoids a degenerate union). Multi-type input
    returns ``Annotated[Union[...], Field(discriminator="kind")]`` — Pydantic's idiomatic
    discriminated-union shape, which OpenAI strict mode handles correctly when ``kind`` is the
    first field on every variant.
    """
    types = tuple(types)
    if not types:
        raise ValueError("build_node_union requires at least one type.")
    if len(types) == 1:
        return types[0]
    return Annotated[Union[types], Field(discriminator="kind")]  # noqa: UP007


def build_edge_union(types: tuple[type[BaseModel], ...] | list[type[BaseModel]]) -> Any:
    """Build a discriminated union over edge *types* (same shape as :func:`build_node_union`).

    Two named functions one shape — call sites read more clearly when intent (nodes vs edges)
    is in the name.
    """
    return build_node_union(types)


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _max_union_size() -> int:
    """Read ``KG_MAX_UNION_SIZE`` env var (default ``10``).

    Raises ``ValueError`` if the env var is not a positive integer.
    """
    raw = os.environ.get("KG_MAX_UNION_SIZE")
    if raw is None:
        return _DEFAULT_MAX_UNION_SIZE
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(f"KG_MAX_UNION_SIZE must be a positive int, got {raw!r}.") from exc
    if v < 1:
        raise ValueError(f"KG_MAX_UNION_SIZE must be a positive int, got {raw!r}.")
    return v


def _assert_kind_discriminator_first(t: type[BaseModel]) -> None:
    """Enforce that ``kind`` is the first declared field on *t*.

    OpenAI structured-output strict mode requires the discriminator to be the first key in every
    union variant; we check at schema-registration time so the error is readable, not a cryptic
    Pydantic discriminated-union build failure later.
    """
    fields = list(t.model_fields.keys())
    if not fields:
        raise ValueError(f"{t.__name__} has no fields; a `kind: Literal[...]` discriminator is required.")
    if fields[0] != "kind":
        raise ValueError(
            f"{t.__name__}: a `kind: Literal[...]` discriminator field must be declared FIRST "
            f"(currently first fields are: {fields[: min(3, len(fields))]}). This is required by "
            f"OpenAI structured-output strict mode for reliable union-member dispatch."
        )


def _assert_edge_carries_edgeref(t: type[BaseModel]) -> None:
    """Enforce that *t* has a ``refs: EdgeRef`` field (edge-endpoint references)."""
    fields = t.model_fields
    if "refs" not in fields:
        raise ValueError(
            f"{t.__name__} is registered as an edge type but has no `refs: EdgeRef` field. "
            f"Every edge must declare `refs: EdgeRef` for endpoint references."
        )
    ann = fields["refs"].annotation
    # Accept EdgeRef directly, or Annotated[EdgeRef, ...] (unwrap once).
    if get_origin(ann) is Annotated:
        ann = get_args(ann)[0]
    if ann is not EdgeRef:
        raise ValueError(
            f"{t.__name__}.refs must be annotated as EdgeRef (got {ann!r}). "
            f"Edges use the shared EdgeRef type so endpoint rewriting at resolution is uniform."
        )
