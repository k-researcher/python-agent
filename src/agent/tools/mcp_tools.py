"""Wrap MCP server tools into agent tools.

This module is separate from the tool registry.
It maps each remote tool to an agent tool.
The mapping uses a policy for safety and limits.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agent.mcp_client import McpError
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError

_SERVER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
# Providers accept function names of up to 64 characters; the limit covers the whole name.
_MAX_TOOL_NAME_LENGTH = 64
# The description comes from the server and goes to the model: keep it short.
_MAX_DESCRIPTION_CHARS = 1024
_DEFAULT_MAX_RESULT_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class McpToolPolicy:
    """Define how MCP tools are exposed to the agent."""

    server_id: str
    risk_level: RiskLevel = RiskLevel.network_access
    allowed_modes: frozenset[str] = frozenset({"dev"})
    allowed_tools: frozenset[str] | None = None
    max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS


class McpToolClient(Protocol):
    """Minimal client surface used by the MCP tool wrapper."""

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _build_full_name(server_id: str, tool_name: str) -> str:
    """Build the agent tool name "mcp__<server>__<tool>" of at most 64 characters.

    Unsafe characters become an underscore. A longer name is cut, and a short hash of
    the full name keeps different long names apart.
    """
    full = f"mcp__{server_id}__{_SAFE_NAME_PATTERN.sub('_', tool_name)}"
    if len(full) <= _MAX_TOOL_NAME_LENGTH:
        return full
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:8]
    return f"{full[: _MAX_TOOL_NAME_LENGTH - len(digest) - 1]}_{digest}"


class McpBackedTool(Tool):
    """An agent tool that calls a remote MCP tool."""

    def __init__(
        self,
        client: McpToolClient,
        policy: McpToolPolicy,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.name = _build_full_name(policy.server_id, name)
        self.description = f"[MCP {policy.server_id}] {str(description)[:_MAX_DESCRIPTION_CHARS]}"
        self.risk_level = policy.risk_level
        self.allowed_modes = policy.allowed_modes
        self.parallel_safe = False
        self._client = client
        self._original_name = name
        self._max_result_chars = policy.max_result_chars
        if isinstance(input_schema, dict) and input_schema.get("type") == "object":
            self.input_schema = input_schema
        else:
            self.input_schema = {"type": "object"}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._client.call_tool(self._original_name, arguments)
        except (McpError, OSError) as exc:
            raise ToolError(str(exc)) from exc
        content = result.get("content", [])
        if not isinstance(content, list):
            content = []
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": str(item.get("text", ""))})
                else:
                    parts.append(
                        {"type": "text", "text": f"[{item.get('type', 'unknown')} content omitted]"}
                    )
            else:
                parts.append({"type": "text", "text": "[unknown content omitted]"})
        total_chars = sum(len(part["text"]) for part in parts)
        if total_chars > self._max_result_chars:
            remaining = self._max_result_chars
            truncated_parts: list[dict[str, Any]] = []
            for part in parts:
                text = part["text"]
                if len(text) <= remaining:
                    truncated_parts.append(part)
                    remaining -= len(text)
                else:
                    truncated_parts.append({"type": "text", "text": text[:remaining]})
                    remaining = 0
                    break
            truncated_parts.append({"type": "text", "text": "[truncated]"})
            parts = truncated_parts
        return {
            "content": parts,
            "is_error": bool(result.get("isError", False)),
        }


async def build_mcp_tools(client: McpToolClient, policy: McpToolPolicy) -> list[McpBackedTool]:
    """Build agent tools from the client's remote tools."""
    if not _SERVER_ID_PATTERN.match(policy.server_id):
        raise ValueError(f"Invalid server id: {policy.server_id}")
    remote_tools = await client.list_tools()
    built: list[McpBackedTool] = []
    seen: set[str] = set()
    for remote in remote_tools:
        name = remote.name
        if policy.allowed_tools is not None and name not in policy.allowed_tools:
            continue
        if not isinstance(name, str):
            continue
        full_name = _build_full_name(policy.server_id, name)
        if full_name in seen:
            raise ToolError(f"Duplicate tool name after normalization: {full_name}")
        seen.add(full_name)
        built.append(
            McpBackedTool(
                client=client,
                policy=policy,
                name=name,
                description=remote.description,
                input_schema=remote.input_schema,
            )
        )
    return built
