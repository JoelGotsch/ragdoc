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

Patterns are **load-bearing**: :class:`~ragdoc.extraction.kg.KnowledgeGraphExtractor` steers the
LLM with :func:`render_patterns_prompt` and validates every extracted edge's
``(source_kind, edge_kind, target_kind)`` triple against :func:`allowed_pattern_kinds`. An edge
type that appears in no pattern is dead configuration and is rejected at declaration time.

Union sizes are bounded by ``ExtractionSettings.max_union_size`` (env
``EXTRACTION_MAX_UNION_SIZE``, default ``10``), checked at extractor construction — schema
validation itself is environment-independent. Above the limit, OpenAI strict-mode reliability dips
and schema size pushes prompt tokens; split into multiple extractors with smaller schemas.
"""

from __future__ import annotations

from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
            ``kind`` discriminator, edge type without ``refs: EdgeRef``, or an edge type that
            appears in no pattern (it could never survive pattern enforcement).
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

        # Patterns are enforced (prompt + post-parse validation): an edge type with no legal
        # (src, edge, tgt) triple could never be emitted — reject that dead configuration here.
        edges_in_patterns = {edge for _, edge, _ in self.patterns}
        for t in self.edge_types:
            if t not in edges_in_patterns:
                raise ValueError(
                    f"Edge type {t.__name__!r} appears in edge_types but in no pattern; with "
                    f"pattern enforcement it can never be emitted. Declare at least one "
                    f"(source, {t.__name__}, target) pattern."
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
# Pattern enforcement helpers (consumed by KnowledgeGraphExtractor)
# ---------------------------------------------------------------------------


def kind_of(model: type[BaseModel]) -> str:
    """Return the ``kind`` Literal value of a registered node/edge model.

    Raises:
        ValueError: if *model* has no ``kind`` field or its annotation is not a single-value
            ``Literal`` (registered schema types always are — see ``GraphSchema`` validation).
    """
    field = model.model_fields.get("kind")
    if field is None:
        raise ValueError(f"{model.__name__} has no `kind` field; cannot derive its kind literal.")
    args = get_args(field.annotation)
    if not args:
        raise ValueError(
            f"{model.__name__}.kind is not a Literal annotation ({field.annotation!r}); cannot derive its kind."
        )
    return str(args[0])


def allowed_pattern_kinds(schema: GraphSchema) -> frozenset[tuple[str, str, str]]:
    """Lower ``schema.patterns`` to the set of legal ``(source_kind, edge_kind, target_kind)`` strings.

    Pure function of the schema; :class:`~ragdoc.extraction.kg.KnowledgeGraphExtractor` caches it
    per schema for the post-parse edge validation.
    """
    return frozenset((kind_of(src), kind_of(edge), kind_of(tgt)) for src, edge, tgt in schema.patterns)


def render_patterns_prompt(schema: GraphSchema) -> str:
    """Render the schema's legal triples as a system-prompt section, keyed by each type's ``kind``.

    Example output::

        Legal relationship patterns (source)-[edge]->(target); emit ONLY these combinations:
          (Person)-[Employment]->(Company)

    Returns an empty string when the schema declares no patterns (i.e. no edge types).
    """
    if not schema.patterns:
        return ""
    lines = "\n".join(f"  ({kind_of(src)})-[{kind_of(edge)}]->({kind_of(tgt)})" for src, edge, tgt in schema.patterns)
    return f"Legal relationship patterns (source)-[edge]->(target); emit ONLY these combinations:\n{lines}"


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


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
