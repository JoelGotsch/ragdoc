"""Unit tests for format_metadata_value (pure function)."""

import json

import pytest
from pydantic import BaseModel

from ragdoc.rendering.elements import format_metadata_value

# --- Helper models ---


class _SampleModel(BaseModel):
    name: str
    count: int


class _InnerModel(BaseModel):
    """Deeply nested: own dict field with plain scalar values."""

    additional_data: dict[str, str]


class _OuterModel(BaseModel):
    """Top-level model: a dict *field* whose values are another BaseModel."""

    name: str
    metadata: dict[str, _InnerModel]


# --- TestSerializeMetadataValue ---


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param("hello", "hello", id="str"),
        pytest.param(42, "42", id="int"),
        pytest.param(3.14, "3.14", id="float"),
        pytest.param(True, "True", id="bool_true"),
        pytest.param(False, "False", id="bool_false"),
        pytest.param(None, "", id="none"),
    ],
)
def test_serialize_metadata_scalar(value, expected):
    assert format_metadata_value(value) == expected


def test_serialize_metadata_dict():
    result = format_metadata_value({"a": 1, "b": "x"})
    assert json.loads(result) == {"a": 1, "b": "x"}


def test_serialize_metadata_nested_dict():
    """Recursively nested dicts are JSON-serialized."""
    value = {"outer": {"inner": [1, 2, 3]}}
    result = format_metadata_value(value)
    assert json.loads(result) == {"outer": {"inner": [1, 2, 3]}}


def test_serialize_metadata_list():
    result = format_metadata_value([1, "two", 3.0])
    assert json.loads(result) == [1, "two", 3.0]


def test_serialize_metadata_nested_list():
    """Recursively nested lists are JSON-serialized."""
    value = [[1, 2], ["a", "b"]]
    result = format_metadata_value(value)
    assert json.loads(result) == [[1, 2], ["a", "b"]]


def test_serialize_metadata_pydantic_model():
    model = _SampleModel(name="doc", count=5)
    result = format_metadata_value(model)
    assert json.loads(result) == {"name": "doc", "count": 5}


def test_serialize_metadata_pydantic_nested_in_dict():
    """Pydantic models inside a plain dict are serialized via MetadataEncoder."""
    value = {"nested": _SampleModel(name="x", count=1)}
    result = format_metadata_value(value)
    parsed = json.loads(result)
    assert parsed == {"nested": {"name": "x", "count": 1}}


def test_serialize_metadata_deeply_nested_pydantic_fields():
    """model_dump_json() recurses into Pydantic-typed dict *fields* automatically.

    _OuterModel.metadata is typed dict[str, _InnerModel] -- a Pydantic field,
    not a plain Python dict.  _InnerModel.additional_data is a further dict.
    model_dump_json() handles the full nesting without help from our code.
    """
    inner = _InnerModel(additional_data={"region": "EMEA", "priority": "high"})
    outer = _OuterModel(name="quarterly-report", metadata={"section-a": inner})
    result = format_metadata_value(outer)
    parsed = json.loads(result)
    assert parsed["name"] == "quarterly-report"
    assert parsed["metadata"]["section-a"]["additional_data"]["region"] == "EMEA"


def test_serialize_metadata_pydantic_deeply_nested_in_plain_dict():
    """BaseModel inside a plain dict with its own BaseModel-typed fields.

    _OuterModel.metadata is a Pydantic field dict[str, _InnerModel].
    Wrapping _OuterModel in a *plain* Python dict exercises MetadataEncoder
    recursively: json.dumps encounters _OuterModel -> model_dump() -> the
    resulting dict is plain JSON-safe because Pydantic already resolved
    _InnerModel within its own field.
    """
    inner = _InnerModel(additional_data={"region": "EMEA"})
    outer = _OuterModel(name="quarterly-report", metadata={"section-a": inner})
    plain_dict_wrapper = {"report": outer}
    result = format_metadata_value(plain_dict_wrapper)
    parsed = json.loads(result)
    assert parsed["report"]["name"] == "quarterly-report"
    assert parsed["report"]["metadata"]["section-a"]["additional_data"]["region"] == "EMEA"
