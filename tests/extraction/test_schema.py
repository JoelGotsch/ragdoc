"""Tests for the typed-KG schema container (GraphSchema, EdgeRef, union helpers)."""

from __future__ import annotations

import os
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, TypeAdapter

from ragdoc.extraction.schema import (
    EdgeRef,
    GraphSchema,
    build_edge_union,
    build_node_union,
)

# ---------------------------------------------------------------------------
# Sample types used across tests
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """A real person mentioned in the document."""

    kind: Literal["Person"] = "Person"
    full_name: str


class Company(BaseModel):
    """A legal entity (LLC, GmbH, Corp., ...)."""

    kind: Literal["Company"] = "Company"
    name: str


class Employment(BaseModel):
    """The relationship: a Person employed at a Company."""

    kind: Literal["Employment"] = "Employment"
    refs: EdgeRef
    role: str | None = None


# ---------------------------------------------------------------------------
# 1. Registration round-trip
# ---------------------------------------------------------------------------


def test_registration_round_trip():
    schema = GraphSchema(
        node_types=(Person, Company),
        edge_types=(Employment,),
        patterns=((Person, Employment, Company),),
    )
    assert schema.node_types == (Person, Company)
    assert schema.edge_types == (Employment,)
    assert schema.patterns == ((Person, Employment, Company),)


def test_registration_accepts_lists():
    """User convenience: lists work as well as tuples."""
    schema = GraphSchema(
        node_types=[Person, Company],
        edge_types=[Employment],
        patterns=[(Person, Employment, Company)],
    )
    assert Person in schema.node_types
    assert Employment in schema.edge_types


# ---------------------------------------------------------------------------
# 2-3. Unknown types in patterns
# ---------------------------------------------------------------------------


def test_pattern_unknown_node_type_raises():
    class NotRegistered(BaseModel):
        kind: Literal["NotRegistered"] = "NotRegistered"
        x: int = 0

    with pytest.raises(ValueError, match="unknown node type"):
        GraphSchema(
            node_types=(Person,),  # Company not registered
            edge_types=(Employment,),
            patterns=((Person, Employment, NotRegistered),),
        )


def test_pattern_unknown_edge_type_raises():
    class NotEdge(BaseModel):
        kind: Literal["NotEdge"] = "NotEdge"
        refs: EdgeRef

    with pytest.raises(ValueError, match="unknown edge type"):
        GraphSchema(
            node_types=(Person, Company),
            edge_types=(Employment,),  # NotEdge not registered
            patterns=((Person, NotEdge, Company),),
        )


# ---------------------------------------------------------------------------
# 4. Discriminator-first enforcement
# ---------------------------------------------------------------------------


def test_discriminator_first_enforced_on_node_type():
    class BadOrder(BaseModel):
        full_name: str  # comes BEFORE kind
        kind: Literal["BadOrder"] = "BadOrder"

    with pytest.raises(ValueError, match=r"kind.*FIRST"):
        GraphSchema(node_types=(BadOrder,), edge_types=(), patterns=())


def test_discriminator_first_enforced_on_edge_type():
    class BadEdge(BaseModel):
        refs: EdgeRef
        kind: Literal["BadEdge"] = "BadEdge"

    with pytest.raises(ValueError, match=r"kind.*FIRST"):
        GraphSchema(
            node_types=(Person, Company),
            edge_types=(BadEdge,),
            patterns=((Person, BadEdge, Company),),
        )


def test_missing_kind_field_raises():
    class NoKind(BaseModel):
        full_name: str

    with pytest.raises(ValueError, match=r"kind"):
        GraphSchema(node_types=(NoKind,), edge_types=(), patterns=())


# ---------------------------------------------------------------------------
# 5. EdgeRef contract
# ---------------------------------------------------------------------------


def test_edge_model_missing_edgeref_raises():
    class NoRefs(BaseModel):
        kind: Literal["NoRefs"] = "NoRefs"
        role: str | None = None

    with pytest.raises(ValueError, match=r"refs.*EdgeRef"):
        GraphSchema(
            node_types=(Person, Company),
            edge_types=(NoRefs,),
            patterns=((Person, NoRefs, Company),),
        )


def test_edge_refs_field_wrong_type_raises():
    class WrongRefs(BaseModel):
        kind: Literal["WrongRefs"] = "WrongRefs"
        refs: str  # not EdgeRef

    with pytest.raises(ValueError, match="EdgeRef"):
        GraphSchema(
            node_types=(Person, Company),
            edge_types=(WrongRefs,),
            patterns=((Person, WrongRefs, Company),),
        )


def test_edgeref_round_trip():
    """EdgeRef itself is a tiny BaseModel — sanity round-trip."""
    er = EdgeRef(source_mention_id="m1", target_mention_id="m2")
    assert er.source_mention_id == "m1"
    assert er.target_mention_id == "m2"


# ---------------------------------------------------------------------------
# 6-8. build_node_union / build_edge_union
# ---------------------------------------------------------------------------


def test_build_node_union_single_type_returns_type_unchanged():
    """Single-type input is a degenerate union — return the type directly."""
    assert build_node_union((Person,)) is Person


def test_build_node_union_multi_returns_discriminated_union():
    union = build_node_union((Person, Company))
    adapter = TypeAdapter(union)
    p = adapter.validate_python({"kind": "Person", "full_name": "Alice"})
    c = adapter.validate_python({"kind": "Company", "name": "Acme"})
    assert isinstance(p, Person)
    assert isinstance(c, Company)


def test_build_node_union_rejects_bad_kind():
    union = build_node_union((Person, Company))
    adapter = TypeAdapter(union)
    with pytest.raises(Exception):  # Pydantic ValidationError
        adapter.validate_python({"kind": "UnknownKind", "full_name": "Alice"})


def test_build_edge_union_same_shape():
    """build_edge_union is the same discriminated-union builder; one name per intent."""
    union = build_edge_union((Employment,))
    assert union is Employment  # single-type degenerate case


# ---------------------------------------------------------------------------
# 9-10. Union size limit
# ---------------------------------------------------------------------------


def _make_node_type(name: str) -> type[BaseModel]:
    """Synthesize a Pydantic node BaseModel with a unique discriminator."""
    from pydantic import create_model

    return create_model(
        name,
        kind=(Literal[name], name),  # type: ignore[valid-type]
        x=(int, 0),
    )


def test_union_size_limit_default_10_rejects_eleven():
    types = tuple(_make_node_type(f"N{i}") for i in range(11))
    # Ensure env var is unset for the default
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KG_MAX_UNION_SIZE", None)
        with pytest.raises(ValueError, match=r"KG_MAX_UNION_SIZE"):
            GraphSchema(node_types=types, edge_types=(), patterns=())


def test_union_size_limit_default_10_accepts_ten():
    types = tuple(_make_node_type(f"N{i}") for i in range(10))
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KG_MAX_UNION_SIZE", None)
        schema = GraphSchema(node_types=types, edge_types=(), patterns=())
        assert len(schema.node_types) == 10


def test_union_size_limit_env_var_override():
    types = tuple(_make_node_type(f"N{i}") for i in range(15))
    with patch.dict(os.environ, {"KG_MAX_UNION_SIZE": "20"}):
        schema = GraphSchema(node_types=types, edge_types=(), patterns=())
        assert len(schema.node_types) == 15


def test_union_size_limit_invalid_env_raises():
    with patch.dict(os.environ, {"KG_MAX_UNION_SIZE": "not-a-number"}):
        with pytest.raises(ValueError, match="positive int"):
            GraphSchema(node_types=(Person,), edge_types=(), patterns=())


def _make_edge_type(name: str) -> type[BaseModel]:
    from pydantic import create_model

    return create_model(
        name,
        kind=(Literal[name], name),  # type: ignore[valid-type]
        refs=(EdgeRef, ...),
    )


def test_union_size_limit_applies_to_edges_too():
    edge_types = tuple(_make_edge_type(f"E{i}") for i in range(11))
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KG_MAX_UNION_SIZE", None)
        with pytest.raises(ValueError, match=r"KG_MAX_UNION_SIZE"):
            GraphSchema(
                node_types=(Person, Company),
                edge_types=edge_types,
                patterns=(),
            )
