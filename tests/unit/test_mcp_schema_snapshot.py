"""Tests for schema snapshotting/aliasing protection in McpBackedTool.

Covers deep snapshot on construction, mutation isolation between the original
wire schema, the returned parameters(), and the internal schema, plus isolation
between multiple tools built from the same schema.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent.tools.mcp_tools import McpBackedTool, McpToolPolicy, build_mcp_tools


class FakeMcpClient:
    """A minimal fake MCP client returning canned tools."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools = tools

    async def list_tools(self) -> list[Any]:
        return [type("RemoteTool", (), dict(t))() for t in self.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"content": [], "isError": False}


def _policy(**overrides: Any) -> McpToolPolicy:
    values: dict[str, Any] = {"server_id": "svc"}
    values.update(overrides)
    return McpToolPolicy(**values)


def _nested_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
            "nested": {
                "type": "object",
                "properties": {"flag": {"type": "boolean"}},
            },
        },
        "required": ["items", "nested"],
    }


def _build_tools(schema: dict[str, Any], count: int = 1) -> list[McpBackedTool]:
    tools = [{"name": f"t{i}", "description": "D.", "input_schema": schema} for i in range(count)]
    client = FakeMcpClient(tools)
    return asyncio.run(build_mcp_tools(client, _policy()))


def test_parameters_returns_deep_copy_that_cannot_mutate_internal() -> None:
    tool = _build_tools(_nested_schema())[0]

    params = tool.parameters()
    # Mutate nested lists and dicts in the returned snapshot.
    params["properties"]["items"]["items"]["properties"]["value"]["type"] = "string"
    params["properties"]["nested"]["properties"]["flag"]["type"] = "number"
    params["required"].append("extra")

    # The internal schema must be unaffected.
    internal = tool.input_schema
    assert internal == _nested_schema()
    assert internal["properties"]["items"]["items"]["properties"]["value"]["type"] == "integer"
    assert internal["properties"]["nested"]["properties"]["flag"]["type"] == "boolean"
    assert "extra" not in internal["required"]


def test_mutating_parameters_does_not_affect_later_calls() -> None:
    tool = _build_tools(_nested_schema())[0]

    first = tool.parameters()
    first["properties"]["items"]["items"]["properties"]["value"]["type"] = "string"
    first["required"].clear()

    second = tool.parameters()
    assert second == _nested_schema()
    assert second["properties"]["items"]["items"]["properties"]["value"]["type"] == "integer"
    assert second["required"] == ["items", "nested"]


def test_mutating_original_wire_schema_does_not_affect_tool() -> None:
    schema = _nested_schema()
    tool = _build_tools(schema)[0]

    # Mutate the original schema passed into the constructor.
    schema["properties"]["items"]["items"]["properties"]["value"]["type"] = "string"
    schema["properties"]["nested"]["properties"]["flag"]["type"] = "number"
    schema["required"].clear()

    assert tool.input_schema == _nested_schema()
    assert tool.parameters() == _nested_schema()
    assert tool.input_schema["properties"]["items"]["items"]["properties"]["value"]["type"] == (
        "integer"
    )
    assert tool.input_schema["properties"]["nested"]["properties"]["flag"]["type"] == "boolean"
    assert tool.input_schema["required"] == ["items", "nested"]


def test_registry_validation_uses_original_schema_after_mutation() -> None:
    """Registry validation inspects the stored input_schema and stays pristine."""
    tool = _build_tools(_nested_schema())[0]

    # A caller requests parameters and mutates the returned value.
    params = tool.parameters()
    params["properties"]["items"]["items"] = {"type": "string"}

    # The registry still validates against the original nested schema.
    internal_items = tool.input_schema["properties"]["items"]
    assert internal_items["type"] == "array"
    assert internal_items["items"]["properties"]["value"]["type"] == "integer"
    # And a fresh parameters() call reflects the unchanged schema.
    fresh_items = tool.parameters()["properties"]["items"]
    assert fresh_items == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    }


def test_multiple_tools_from_same_schema_are_isolated() -> None:
    tools = _build_tools(_nested_schema(), count=2)
    ta, tb = tools[0], tools[1]
    assert ta is not tb
    assert ta.input_schema is not tb.input_schema

    # Mutate tool a's parameters; tool b is untouched.
    pa = ta.parameters()
    pa["properties"]["items"]["items"]["properties"]["value"]["type"] = "string"
    pa["required"].append("extra")

    assert tb.parameters()["properties"]["items"]["items"]["properties"]["value"]["type"] == (
        "integer"
    )
    assert "extra" not in tb.parameters()["required"]

    # Stored schemas are distinct objects but equal.
    assert ta.input_schema is not tb.input_schema
    assert ta.input_schema == tb.input_schema == _nested_schema()


def test_parameters_returns_fresh_object_each_call() -> None:
    tool = _build_tools(_nested_schema())[0]

    p1 = tool.parameters()
    p2 = tool.parameters()
    assert p1 is not p2
    assert p1 == p2 == _nested_schema()
    assert p1["properties"]["items"] is not p2["properties"]["items"]


def test_deep_nested_aliasing_is_broken() -> None:
    tool = _build_tools(_nested_schema())[0]

    # Mutate a deep leaf through the returned parameters() snapshot.
    params = tool.parameters()
    params["properties"]["items"]["items"]["properties"]["value"]["type"] = "string"

    # The internal schema is not aliased: parameters() served a copy.
    internal_items = tool.input_schema["properties"]["items"]["items"]
    assert internal_items["properties"]["value"]["type"] == "integer"


def test_fallback_for_malformed_top_level_schema_is_snapshot() -> None:
    tool = _build_tools("not a schema")[0]
    assert tool.input_schema == {"type": "object"}
    params = tool.parameters()
    assert params == {"type": "object"}
    params["type"] = "array"
    assert tool.parameters() == {"type": "object"}
