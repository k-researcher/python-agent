"""Wrap MCP server tools into agent tools.

This module is separate from the tool registry.
It maps each remote tool to an agent tool.
The mapping uses a policy for safety and limits.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol

from agent.mcp_client import McpError
from agent.source_revision import schema_hash, source_revision
from agent.tool_sources import ToolSource
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
        *,
        source: ToolSource | None = None,
    ) -> None:
        self.name = _build_full_name(policy.server_id, name)
        self.description = f"[MCP {policy.server_id}] {str(description)[:_MAX_DESCRIPTION_CHARS]}"
        self.risk_level = policy.risk_level
        self.allowed_modes = policy.allowed_modes
        self.parallel_safe = False
        self._client = client
        self._original_name = name
        self._source = source
        self._policy = policy
        self._max_result_chars = policy.max_result_chars
        if isinstance(input_schema, dict) and input_schema.get("type") == "object":
            self.input_schema = copy.deepcopy(input_schema)
        else:
            self.input_schema = {"type": "object"}

        if source is not None:
            self.approval_revision()

    @property
    def source(self) -> ToolSource | None:
        return self._source

    def approval_revision(self) -> str:
        """Bind approval to the trusted source and exact exposed tool snapshot."""
        source = self._source
        if source is None:
            raise ToolError("MCP tool has no trusted source")
        if (
            not source.enabled
            or self._policy.server_id != source.source_id
            or self.risk_level != source.risk_level
            or self.allowed_modes != source.allowed_modes
            or self._max_result_chars != source.max_result_chars
            or self.name != _build_full_name(source.source_id, self._original_name)
            or (
                self._policy.allowed_tools is not None
                and self._original_name not in self._policy.allowed_tools
            )
        ):
            raise ToolError("MCP policy does not match its trusted source")
        try:
            return schema_hash(
                {
                    "source": source_revision(source),
                    "name": self.name,
                    "remote_name": self._original_name,
                    "description": self.description,
                    "parameters": self.parameters(),
                    "allowed_tools": (
                        sorted(self._policy.allowed_tools)
                        if self._policy.allowed_tools is not None
                        else None
                    ),
                }
            )
        except (ValueError, RecursionError) as exc:
            raise ToolError("Invalid MCP approval snapshot") from exc

    def parameters(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """Return a deep snapshot so callers cannot mutate the internal schema."""
        return copy.deepcopy(self.input_schema)

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


async def build_mcp_tools(
    client: McpToolClient, policy: McpToolPolicy, *, source: ToolSource | None = None
) -> list[McpBackedTool]:
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
                source=source,
            )
        )
    return built
