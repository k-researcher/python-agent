from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from agent.config import Settings
from agent.tools.advanced import advanced_tools
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.builtin import BUILTIN_TOOLS
from agent.tools.knowledge import knowledge_tools


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
                *advanced_tools(self.settings),
                *knowledge_tools(self.settings),
            )
        self._tools = {tool.name: tool for tool in tools}

    def _enabled(self, tool: Tool) -> bool:
        if tool.network_capability and not self.settings.allow_network_tools:
            return False
        return not tool.knowledge_capability or self.settings.knowledge_base_enabled

    def definitions(
        self, mode: str, *, model_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters(model_ids),
                },
            }
            for tool in self._tools.values()
            if mode in tool.allowed_modes and self._enabled(tool)
        ]

    def get(self, name: str, mode: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None or mode not in tool.allowed_modes or not self._enabled(tool):
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
        validator = Draft202012Validator(self.get(name, mode).parameters(model_ids))
        problems = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
        if problems:
            details = "; ".join(
                f"{'.'.join(map(str, error.path)) or '<arguments>'}: {error.message}"
                for error in problems[:5]
            )
            raise ToolError(f"Invalid arguments for {name}: {details}")

    def risk(self, name: str, mode: str) -> RiskLevel:
        return self.get(name, mode).risk_level

    def requires_confirmation(self, name: str, mode: str) -> bool:
        return self.risk(name, mode) is not RiskLevel.read_only

    async def execute(
        self, name: str, mode: str, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.get(name, mode).execute(context, arguments)
