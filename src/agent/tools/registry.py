from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from agent.config import Settings
from agent.modes import tool_profile
from agent.tools.advanced import advanced_tools
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.builtin import BUILTIN_TOOLS
from agent.tools.check_tools import RunChecksTool
from agent.tools.git_tools import GitDiffTool, GitLogTool, GitShowTool, GitStatusTool
from agent.tools.knowledge import knowledge_tools
from agent.tools.patch_tools import ApplyPatchTool, EditByLinesTool
from agent.tools.session_tools import GetProjectsTool, ReadSessionTool
from agent.tools.todo_tools import TodoReadTool, TodoWriteTool
from agent.tools.web_fetch import WebFetchTool


def _wire_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return the schema sent to the model; an object schema always has "properties".

    An empty properties map makes argument-free object schemas explicit.
    The tool's own schema is not changed: argument validation uses it as it is.
    """
    if parameters.get("type") == "object" and "properties" not in parameters:
        return {**parameters, "properties": {}}
    return parameters


class ToolRegistry:
    def __init__(
        self,
        settings: Settings | None = None,
        tools: tuple[Tool, ...] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if tools is None:
            tools = (
                *BUILTIN_TOOLS,
                ApplyPatchTool(),
                EditByLinesTool(),
                GitStatusTool(),
                GitDiffTool(),
                GitLogTool(),
                GitShowTool(),
                RunChecksTool(),
                TodoWriteTool(),
                TodoReadTool(),
                ReadSessionTool(),
                GetProjectsTool(),
                *advanced_tools(self.settings),
                WebFetchTool(self.settings),
                *knowledge_tools(self.settings),
            )
        self._tools = {tool.name: tool for tool in tools}

    def with_tools(self, tools: tuple[Tool, ...]) -> ToolRegistry:
        """Create an isolated registry without replacing existing tool names."""
        combined = dict(self._tools)
        for tool in tools:
            if tool.name in combined:
                raise ToolError(f"Tool name is already registered: {tool.name}")
            combined[tool.name] = tool
        return ToolRegistry(settings=self.settings, tools=tuple(combined.values()))

    def _enabled(self, tool: Tool) -> bool:
        if tool.network_capability and not self.settings.allow_network_tools:
            return False
        return not tool.knowledge_capability or self.settings.knowledge_base_enabled

    def definitions(self, mode: str, *, model_ids: list[str] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": _wire_parameters(tool.parameters(model_ids)),
                },
            }
            for tool in self._tools.values()
            if tool_profile(mode) in tool.allowed_modes and self._enabled(tool)
        ]

    def get(self, name: str, mode: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None or tool_profile(mode) not in tool.allowed_modes or not self._enabled(tool):
            raise ToolError(f"Tool is unavailable in mode {mode}: {name}")
        return tool

    def validate(
        self,
        name: str,
        mode: str,
        arguments: dict[str, Any],
        *,
        model_ids: list[str] | None = None,
    ) -> None:
        """Reject arguments that do not match the tool's JSON Schema, before any approval."""
        tool = self.get(name, mode)
        validator = Draft202012Validator(tool.parameters(model_ids))
        problems = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
        if problems:
            details = "; ".join(
                f"{'.'.join(map(str, error.path)) or '<arguments>'}: {error.message}"
                for error in problems[:5]
            )
            raise ToolError(f"Invalid arguments for {name}: {details}")
        tool.check_arguments(arguments)

    def risk(
        self,
        name: str,
        mode: str,
        arguments: dict[str, Any] | None = None,
        project_root: Path | None = None,
    ) -> RiskLevel:
        """Return the risk of a tool, or of one call when its arguments are given."""
        tool = self.get(name, mode)
        if arguments is None or not self.settings.trusted_read_commands:
            return tool.risk_level
        try:
            return tool.call_risk(arguments, project_root)
        except Exception:
            # Any doubt keeps the tool's own risk and so the approval.
            return tool.risk_level

    def requires_confirmation(
        self,
        name: str,
        mode: str,
        arguments: dict[str, Any] | None = None,
        project_root: Path | None = None,
    ) -> bool:
        return self.risk(name, mode, arguments, project_root) is not RiskLevel.read_only

    async def execute(
        self, name: str, mode: str, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.get(name, mode).execute(context, arguments)
