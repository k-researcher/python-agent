"""Tests for deterministic tool-source and schema hashing."""

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any

import pytest

from agent.source_revision import schema_hash, source_revision
from agent.tool_sources import ToolSource
from agent.tools.base import RiskLevel

_VALID = {
    "source_id": "server-1",
    "workspace_id": "Workspace-UUID",
    "command": ["server", "run", "--port", "8080"],
}


def _source(**overrides: object) -> ToolSource:
    return ToolSource.model_validate({**_VALID, **overrides})


@pytest.mark.parametrize("schema", [None, [], "object", 1, (), True])
def test_schema_hash_rejects_non_object_root(schema: Any) -> None:
    with pytest.raises(ValueError, match="must be an object"):
        schema_hash(schema)


def test_revision_is_deterministic() -> None:
    source = _source()
    assert source_revision(source) == source_revision(source)


def test_set_order_invariance() -> None:
    base = _source(project_ids=["a", "b", "c", "d"], allowed_modes=["dev", "ask"])
    shuffled = _source(
        project_ids=["d", "a", "c", "b"],
        allowed_modes=["ask", "dev"],
    )
    assert source_revision(base) == source_revision(shuffled)


def test_dict_key_order_invariance() -> None:
    first = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
        },
    }
    second = {
        "properties": {
            "b": {"type": "integer"},
            "a": {"type": "string"},
        },
        "required": ["a", "b"],
        "type": "object",
    }
    assert schema_hash(first) == schema_hash(second)


def test_schema_list_order_is_preserved() -> None:
    first = {"enum": [1, 2, 3]}
    second = {"enum": [3, 1, 2]}
    assert schema_hash(first) != schema_hash(second)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "server-2"),
        ("workspace_id", "other-workspace"),
        ("project_ids", frozenset({"other-project"})),
        ("command", ("server", "build")),
        ("env_keys", ("OTHER_TOKEN",)),
        ("enabled", True),
        ("risk_level", RiskLevel.destructive),
        ("allowed_modes", frozenset({"ask"})),
        ("max_result_chars", 100),
    ],
)
def test_every_field_change_invalidates_revision(field: str, value: object) -> None:
    base = _source()
    other = _source(**{field: value})
    assert source_revision(base) != source_revision(other)


def test_command_order_matters() -> None:
    ordered = _source(command=["server", "run", "--flag"])
    reordered = _source(command=["run", "--flag", "server"])
    assert source_revision(ordered) != source_revision(reordered)


def test_env_keys_order_matters() -> None:
    ordered = _source(env_keys=["A", "B"])
    reordered = _source(env_keys=["B", "A"])
    assert source_revision(ordered) != source_revision(reordered)


def test_schema_hash_does_not_mutate_input() -> None:
    schema = {
        "type": "object",
        "properties": {
            "b": {"type": "integer"},
            "a": {"enum": [3, 1, 2]},
        },
        "required": ["a"],
    }
    snapshot = deepcopy(schema)
    schema_hash(schema)
    assert schema == snapshot


def test_source_revision_does_not_mutate_source() -> None:
    source = _source(
        project_ids=["b", "a"],
        command=["server", "run"],
        allowed_modes=["dev", "ask"],
    )
    source_revision(source)
    assert set(source.project_ids) == {"b", "a"}
    assert source.command == ("server", "run")
    assert source.allowed_modes == frozenset({"dev", "ask"})


@pytest.mark.parametrize(
    "bad",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
        {"nested": {"inner": float("nan")}},
        {"array": [1.0, float("inf")]},
    ],
)
def test_schema_hash_rejects_nonfinite(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schema_hash(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"value": object()},
        {"value": b"bytes"},
        {"value": {"nested": object()}},
        {"value": [object()]},
    ],
)
def test_schema_hash_rejects_non_json_types(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schema_hash(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"tags": {1, 2, 3}},
        {"tags": frozenset({1, 2, 3})},
        {"value": (1, 2)},
        {"nested": {"inner": {1}}},
        {"array": [(1, 2)]},
    ],
)
def test_schema_hash_rejects_sets_and_tuples(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schema_hash(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {1: "x"},
        {1: "x", "1": "y"},
        {"a": {1: "x"}},
        {"a": {"b": {2.5: "c"}}},
        {"nested": {"list": [{7: "deep"}]}},
    ],
)
def test_schema_hash_rejects_non_string_keys(bad: dict[object, object]) -> None:
    with pytest.raises(ValueError):
        schema_hash(bad)


def test_schema_hash_rejects_cycle() -> None:
    schema: dict[str, object] = {}
    schema["self"] = schema
    with pytest.raises(ValueError):
        schema_hash(schema)

    shared_list: list[object] = []
    shared_list.append(shared_list)
    with pytest.raises(ValueError):
        schema_hash({"array": shared_list})


def test_schema_hash_shared_reference_is_allowed() -> None:
    shared = {"type": "string"}
    schema = {"a": shared, "b": shared}
    assert schema_hash(schema) == schema_hash(deepcopy(schema))


def test_schema_hash_deterministic() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert schema_hash(schema) == schema_hash(deepcopy(schema))


def test_schema_hash_differs_on_semantic_change() -> None:
    assert schema_hash({"type": "string"}) != schema_hash({"type": "integer"})


def test_revision_digest_format() -> None:
    digest = source_revision(_source())
    assert len(digest) == 64
    int(digest, 16)


def test_output_is_stable_sha256() -> None:
    payload = {"a": 1, "b": "x"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert schema_hash(payload) == sha256(raw).hexdigest()


def test_schema_numeric_types_and_nesting() -> None:
    first = {"minimum": 1, "maximum": 10, "type": "integer"}
    second = {"maximum": 10, "type": "integer", "minimum": 1}
    assert schema_hash(first) == schema_hash(second)
