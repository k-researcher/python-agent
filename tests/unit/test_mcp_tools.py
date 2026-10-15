"""Unit tests for the MCP tool wrapper module.

Use a fake client object.
The fake client does not spawn a process.
It returns canned tool lists and call results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.mcp_client import McpError
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.mcp_tools import McpBackedTool, McpToolPolicy, build_mcp_tools


class FakeMcpClient:
    """A fake MCP client for tests."""

    def __init__(
        self,
        tools: list[dict[str, Any]] | None = None,
        *,
        results: dict[str, dict[str, Any]] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.tools = tools or []
        self.results = results or {}
        self.errors = errors or {}
        self.called: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[Any]:
        return [type("RemoteTool", (), dict(t))() for t in self.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.called.append((name, arguments))
        if name in self.errors:
            raise self.errors[name]
        return self.results.get(name, {"content": [], "isError": False})


def _policy(**overrides: Any) -> McpToolPolicy:
    values: dict[str, Any] = {"server_id": "svc"}
    values.update(overrides)
    return McpToolPolicy(**values)


def _context() -> ToolContext:
    return ToolContext(session_id="s1", project_root=Path("/tmp"))


async def test_build_creates_tools_with_correct_names() -> None:
    client = FakeMcpClient(
        tools=[
            {"name": "echo", "description": "Echo text.", "input_schema": {}},
            {"name": "add", "description": "Add numbers.", "input_schema": {}},
        ]
    )
    tools = await build_mcp_tools(client, _policy())

    assert [tool.name for tool in tools] == ["mcp__svc__echo", "mcp__svc__add"]


async def test_normalized_name_replaces_unsafe_characters() -> None:
    client = FakeMcpClient(
        tools=[{"name": "a.b:c", "description": "Odd name.", "input_schema": {}}]
    )
    tools = await build_mcp_tools(client, _policy())

    assert tools[0].name == "mcp__svc__a_b_c"


async def test_long_name_is_trimmed_with_hash() -> None:
    long_name = "x" * 100
    client = FakeMcpClient(tools=[{"name": long_name, "description": "Long.", "input_schema": {}}])
    tools = await build_mcp_tools(client, _policy())

    full = tools[0].name
    # The whole name, prefix included, fits the 64-character limit of providers.
    assert full.startswith("mcp__svc__")
    assert len(full) == 64
    assert full[: 64 - 9] == ("mcp__svc__" + "x" * 100)[: 64 - 9]
    assert len(full.split("_")[-1]) == 8


async def test_long_names_that_differ_at_the_end_stay_apart() -> None:
    names = ["x" * 100 + "a", "x" * 100 + "b"]
    client = FakeMcpClient(
        tools=[{"name": name, "description": "D.", "input_schema": {}} for name in names]
    )
    tools = await build_mcp_tools(client, _policy())
    assert len({tool.name for tool in tools}) == 2
    assert all(len(tool.name) <= 64 for tool in tools)


async def test_long_description_is_cut() -> None:
    client = FakeMcpClient(tools=[{"name": "t", "description": "d" * 5000, "input_schema": {}}])
    tools = await build_mcp_tools(client, _policy())
    assert len(tools[0].description) <= len("[MCP svc] ") + 1024


async def test_schema_not_object_becomes_default() -> None:
    client = FakeMcpClient(
        tools=[{"name": "t", "description": "D.", "input_schema": "not a schema"}]
    )
    tools = await build_mcp_tools(client, _policy())

    assert tools[0].input_schema == {"type": "object"}


async def test_schema_object_is_kept() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    client = FakeMcpClient(tools=[{"name": "t", "description": "D.", "input_schema": schema}])
    tools = await build_mcp_tools(client, _policy())

    assert tools[0].input_schema == schema


async def test_allowed_tools_filters_remote_tools() -> None:
    client = FakeMcpClient(
        tools=[
            {"name": "keep", "description": "K.", "input_schema": {}},
            {"name": "drop", "description": "D.", "input_schema": {}},
        ]
    )
    tools = await build_mcp_tools(client, _policy(allowed_tools=frozenset({"keep"})))

    assert [tool.name for tool in tools] == ["mcp__svc__keep"]


async def test_allowed_tools_none_keeps_all() -> None:
    client = FakeMcpClient(
        tools=[
            {"name": "a", "description": "A.", "input_schema": {}},
            {"name": "b", "description": "B.", "input_schema": {}},
        ]
    )
    tools = await build_mcp_tools(client, _policy())

    assert len(tools) == 2


async def test_duplicate_after_normalization_raises() -> None:
    client = FakeMcpClient(
        tools=[
            {"name": "a.b", "description": "X.", "input_schema": {}},
            {"name": "a_b", "description": "Y.", "input_schema": {}},
        ]
    )
    with pytest.raises(ToolError, match="Duplicate"):
        await build_mcp_tools(client, _policy())


async def test_execute_returns_text_content() -> None:
    client = FakeMcpClient(
        tools=[{"name": "echo", "description": "E.", "input_schema": {}}],
        results={"echo": {"content": [{"type": "text", "text": "hello"}], "isError": False}},
    )
    tool = (await build_mcp_tools(client, _policy()))[0]
    result = await tool.execute(_context(), {"text": "hi"})

    assert result == {"content": [{"type": "text", "text": "hello"}], "is_error": False}
    assert client.called == [("echo", {"text": "hi"})]


async def test_execute_replaces_non_text_parts() -> None:
    client = FakeMcpClient(
        tools=[{"name": "t", "description": "D.", "input_schema": {}}],
        results={
            "t": {
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "data": "x"},
                    {"type": "audio", "data": "y"},
                ],
                "isError": False,
            }
        },
    )
    tool = (await build_mcp_tools(client, _policy()))[0]
    result = await tool.execute(_context(), {})

    assert result["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "[image content omitted]"},
        {"type": "text", "text": "[audio content omitted]"},
    ]


async def test_execute_truncates_long_result() -> None:
    long_text = "x" * 100
    client = FakeMcpClient(
        tools=[{"name": "t", "description": "D.", "input_schema": {}}],
        results={"t": {"content": [{"type": "text", "text": long_text}], "isError": False}},
    )
    tool = (await build_mcp_tools(client, _policy(max_result_chars=10)))[0]
    result = await tool.execute(_context(), {})

    assert result["content"][0]["text"] == "x" * 10
    assert result["content"][1]["text"] == "[truncated]"


async def test_execute_returns_is_error_flag() -> None:
    client = FakeMcpClient(
        tools=[{"name": "t", "description": "D.", "input_schema": {}}],
        results={"t": {"content": [{"type": "text", "text": "boom"}], "isError": True}},
    )
    tool = (await build_mcp_tools(client, _policy()))[0]
    result = await tool.execute(_context(), {})

    assert result["is_error"] is True
    assert result["content"] == [{"type": "text", "text": "boom"}]


async def test_mcp_error_becomes_tool_error() -> None:
    client = FakeMcpClient(
        tools=[{"name": "t", "description": "D.", "input_schema": {}}],
        errors={"t": McpError(-32000, "boom")},
    )
    tool = (await build_mcp_tools(client, _policy()))[0]
    with pytest.raises(ToolError) as exc_info:
        await tool.execute(_context(), {})

    assert str(exc_info.value) == "boom"


async def test_policy_risk_and_modes_are_applied() -> None:
    policy = _policy(
        risk_level=RiskLevel.destructive,
        allowed_modes=frozenset({"ask", "dev"}),
    )
    client = FakeMcpClient(tools=[{"name": "t", "description": "D.", "input_schema": {}}])
    tool = (await build_mcp_tools(client, policy))[0]

    assert tool.risk_level is RiskLevel.destructive
    assert tool.allowed_modes == frozenset({"ask", "dev"})


async def test_parallel_safe_is_false() -> None:
    client = FakeMcpClient(tools=[{"name": "t", "description": "D.", "input_schema": {}}])
    tool = (await build_mcp_tools(client, _policy()))[0]

    assert tool.parallel_safe is False


async def test_description_has_server_prefix() -> None:
    client = FakeMcpClient(tools=[{"name": "t", "description": "My tool.", "input_schema": {}}])
    tool = (await build_mcp_tools(client, _policy()))[0]

    assert tool.description == "[MCP svc] My tool."


async def test_invalid_server_id_raises_value_error() -> None:
    client = FakeMcpClient()
    with pytest.raises(ValueError, match="server id"):
        await build_mcp_tools(client, _policy(server_id="Bad ID!"))


async def test_mcp_backed_tool_is_tool_instance() -> None:
    client = FakeMcpClient(tools=[{"name": "t", "description": "D.", "input_schema": {}}])
    tool = (await build_mcp_tools(client, _policy()))[0]

    assert isinstance(tool, Tool)
    assert isinstance(tool, McpBackedTool)
