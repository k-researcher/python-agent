"""Adversarial tests for the MCP approval-revision source binding.

Covers the optional keyword ``source`` on McpBackedTool, the ``source``
property, approval_revision stable snapshots and rejection of unbound/disabled/
mismatched sources, alias protection for the remote schema and returned
parameters, whether genuine schema/source changes move the revision, whether
binding-critical policy mismatches fail, that hashing never touches the MCP
client, and that build_mcp_tools forwards the trusted source. Invalid nested
JSON and cyclic schemas must surface as first generic ToolError.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.tool_sources import ToolSource
from agent.tools.base import ToolError
from agent.tools.mcp_tools import McpBackedTool, McpToolPolicy, build_mcp_tools


class SpyingClient:
    """Fake MCP client that records every interaction with the remote server."""

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self.tools = tools or [{"name": "t", "description": "D.", "input_schema": {}}]
        self.list_calls = 0
        self.call_calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[Any]:
        self.list_calls += 1
        return [type("RemoteTool", (), dict(entry))() for entry in self.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_calls.append((name, arguments))
        return {"content": [], "isError": False}


def _source(**kw: Any) -> ToolSource:
    base: dict[str, Any] = {
        "source_id": "svc",
        "workspace_id": "ws",
        "project_ids": frozenset(),
        "command": ("server",),
        "enabled": True,
    }
    base.update(kw)
    return ToolSource(**base)


def _policy(**kw: Any) -> McpToolPolicy:
    values: dict[str, Any] = {"server_id": "svc"}
    values.update(kw)
    return McpToolPolicy(**values)


def _build(schema: dict[str, Any] | None = None, source: ToolSource | None = None) -> McpBackedTool:
    client = SpyingClient()
    return McpBackedTool(
        client,
        _policy(),
        "t",
        "D.",
        schema if schema is not None else {},
        source=source,
    )


def _build_via_factory(
    source: ToolSource | None = None, policy: McpToolPolicy | None = None
) -> tuple[list[McpBackedTool], SpyingClient]:
    client = SpyingClient()
    tools = asyncio.run(build_mcp_tools(client, policy or _policy(), source=source))
    return tools, client


def test_source_is_optional_keyword_and_property() -> None:
    unbound = _build()
    assert unbound.source is None

    src = _source()
    bound = _build(source=src)
    assert bound.source is src


def test_unbound_tool_rejects_approval_revision() -> None:
    tool = _build()
    with pytest.raises(ToolError, match="no trusted source"):
        tool.approval_revision()


def test_legitimate_snapshot_is_stable() -> None:
    tool = _build(source=_source())
    rev1 = tool.approval_revision()
    rev2 = tool.approval_revision()
    assert rev1 == rev2
    assert isinstance(rev1, str)
    assert len(rev1) == 64


def test_approval_revision_hashes_all_bound_inputs() -> None:
    base_src = _source()
    base_revision = _build(source=base_src).approval_revision()

    diff_name = McpBackedTool(
        SpyingClient(), _policy(), "other", "D.", {}, source=base_src
    ).approval_revision()
    assert diff_name != base_revision

    diff_desc = _build(source=base_src)
    diff_desc.description = "[MCP svc] Changed."
    assert diff_desc.approval_revision() != base_revision

    diff_allow = _build(source=base_src)
    diff_allow._policy = _policy(allowed_tools=frozenset({"t"}))
    assert diff_allow.approval_revision() != base_revision


def test_remote_schema_input_alias_cannot_change_revision() -> None:
    remote_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    tool = McpBackedTool(SpyingClient(), _policy(), "t", "D.", remote_schema, source=_source())
    rev = tool.approval_revision()
    remote_schema["properties"]["x"]["type"] = "integer"
    assert tool.approval_revision() == rev


def test_parameters_return_alias_does_not_change_revision() -> None:
    tool = _build(source=_source())
    rev = tool.approval_revision()
    params = tool.parameters()
    assert isinstance(params, dict)
    if isinstance(params, dict):
        params["type"] = "array"
    assert tool.parameters() == {"type": "object"}
    assert tool.approval_revision() == rev


def test_real_schema_mutation_changes_revision() -> None:
    tool = McpBackedTool(
        SpyingClient(),
        _policy(),
        "t",
        "D.",
        {"type": "object", "properties": {"x": {"type": "string"}}},
        source=_source(),
    )
    rev = tool.approval_revision()
    tool.input_schema["properties"]["x"]["type"] = "integer"
    assert tool.approval_revision() != rev


@pytest.mark.parametrize(
    "label,field_value",
    [
        ("command", {"command": ("server", "--extra")}),
        ("env_keys", {"env_keys": ("TOKEN",)}),
        ("workspace", {"workspace_id": "workspace-two"}),
        ("project_ids", {"project_ids": frozenset({"p-1"})}),
    ],
)
def test_source_config_change_moves_revision(label: str, field_value: dict[str, Any]) -> None:
    rev = _build(source=_source()).approval_revision()
    moved = _build(source=_source(**field_value)).approval_revision()
    assert moved != rev, f"source change via {label} did not move the revision"


def test_source_config_change_order_stays_stable() -> None:
    first = _build(source=_source(project_ids=frozenset({"p1", "p2", "p3"})))
    second = _build(source=_source(project_ids=frozenset({"p3", "p1", "p2"})))
    assert first.approval_revision() == second.approval_revision()


@pytest.mark.parametrize(
    "label,source_kwargs,policy_kwargs",
    [
        ("enabled", {"enabled": False}, {}),
        ("risk_level", {"risk_level": "read_only"}, {}),
        (
            "allowed_modes",
            {"allowed_modes": frozenset({"ask"})},
            {"allowed_modes": frozenset({"dev"})},
        ),
        ("max_result_chars", {"max_result_chars": 50}, {"max_result_chars": 100}),
        ("source_id", {"source_id": "other"}, {}),
    ],
)
def test_policy_mismatch_cannot_bind(
    label: str, source_kwargs: dict[str, Any], policy_kwargs: dict[str, Any]
) -> None:
    src = _source(**source_kwargs)
    with pytest.raises(ToolError, match="does not match its trusted source"):
        McpBackedTool(SpyingClient(), _policy(**policy_kwargs), "t", "D.", {}, source=src)


def test_allowed_tools_excluding_remote_name_cannot_bind() -> None:
    src = _source()
    policy = _policy(allowed_tools=frozenset({"different-tool"}))
    with pytest.raises(ToolError, match="does not match its trusted source"):
        McpBackedTool(SpyingClient(), policy, "t", "D.", {}, source=src)


def test_disabled_source_fails_closed() -> None:
    src = _source(enabled=False)
    with pytest.raises(ToolError, match="does not match its trusted source"):
        _build(source=src)


def test_builtin_client_never_called_by_hashing() -> None:
    tool = McpBackedTool(SpyingClient(), _policy(), "t", "D.", {}, source=_source())
    tool.approval_revision()
    tool.approval_revision()
    assert tool._client.list_calls == 0
    assert tool._client.call_calls == []


def test_build_mcp_tools_forwards_source() -> None:
    src = _source()
    tools, client = _build_via_factory(source=src)
    assert len(tools) == 1
    assert tools[0].source is src
    assert tools[0].approval_revision() == tools[0].approval_revision()


def test_build_mcp_tools_without_source_leaves_unbound() -> None:
    tools, _ = _build_via_factory(source=None)
    assert tools[0].source is None
    with pytest.raises(ToolError, match="no trusted source"):
        tools[0].approval_revision()


def test_cyclic_schema_raises_generic_tool_error() -> None:
    cyclic: dict[str, Any] = {"type": "object"}
    cyclic["self"] = cyclic
    with pytest.raises(ToolError, match="Invalid MCP approval snapshot"):
        _build(schema=cyclic, source=_source())


def test_invalid_nested_json_raises_generic_tool_error() -> None:
    for bad_schema in (
        {"type": "object", "properties": {"x": {"default": float("nan")}}},
        {"type": "object", "properties": {"x": {"default": object()}}},
        {"type": "object", "properties": {"x": {"default": float("inf")}}},
    ):
        with pytest.raises(ToolError, match="Invalid MCP approval snapshot"):
            _build(schema=bad_schema, source=_source())
