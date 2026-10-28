"""Read-only access to project conversations and workspace projects."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from agent.database import session_factory
from agent.models import Message, Project, Session
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError

MAX_SESSION_MESSAGES = 200
DEFAULT_SESSION_MESSAGES = 30
MAX_MESSAGE_CHARS = 4000


def _message_tools(message: Message) -> list[str]:
    """Return the names of tools invoked by an assistant message."""
    names: list[str] = []
    calls = message.tool_calls or []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


class ReadSessionTool(Tool):
    """Read the recent conversation of a session scoped to the current project."""

    name = "read_session"
    description = "Read the recent conversation of another session in the same project."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev"})
    parallel_safe = False
    input_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "message_id": {"type": "integer", "minimum": 1},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SESSION_MESSAGES,
                "default": DEFAULT_SESSION_MESSAGES,
            },
        },
        "required": ["session_id"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        target_id = str(arguments["session_id"])
        limit = max(1, min(MAX_SESSION_MESSAGES, int(arguments.get("limit", 30))))

        async with session_factory() as db:
            current = await db.get(Session, context.session_id)
            target = await db.get(Session, target_id)
            # A foreign or missing session must be indistinguishable: same error for both.
            if current is None or target is None or current.project_id != target.project_id:
                raise ToolError("Session not found")

            message_id = arguments.get("message_id")
            rows = list(
                (
                    await db.execute(
                        select(Message)
                        .where(
                            Message.session_id == target.id,
                            Message.kind == "normal",
                            Message.skipped.is_(False)
                            if message_id is None
                            else Message.id == int(message_id),
                        )
                        .order_by(Message.id.desc())
                        .limit(limit)
                    )
                ).scalars()
            )

        messages: list[dict[str, Any]] = []
        for row in reversed(rows):
            content = row.content or ""
            truncated = len(content) > MAX_MESSAGE_CHARS
            messages.append(
                {
                    "id": row.id,
                    "role": row.role,
                    "content": content[:MAX_MESSAGE_CHARS],
                    "truncated": truncated,
                    "tools": _message_tools(row),
                }
            )
        return {
            "session_id": target.id,
            "status": target.status,
            "title": target.title,
            "messages": messages,
        }


class GetProjectsTool(Tool):
    """List projects in the workspace of the current session's project."""

    name = "get_projects"
    description = (
        "List the projects of the current workspace: id, name, root path and which one is current."
    )
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev"})
    parallel_safe = False
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        async with session_factory() as db:
            current = await db.get(Session, context.session_id)
            if current is None:
                raise ToolError("Session not found")
            project = await db.get(Project, current.project_id)
            if project is None:
                raise ToolError("Session not found")
            scope = (
                Project.workspace_id == project.workspace_id
                if project.workspace_id is not None
                else Project.workspace_id.is_(None)
            )
            projects = list(
                (
                    await db.execute(
                        select(Project).where(scope).order_by(Project.name, Project.id)
                    )
                ).scalars()
            )
            return {
                "projects": [
                    {
                        "id": row.id,
                        "name": row.name,
                        "root_path": row.root_path,
                        "current": row.id == current.project_id,
                    }
                    for row in projects
                ]
            }


class MessageToSessionTool(Tool):
    """Queue untrusted data for another session in the current project."""

    name = "message_to_session"
    description = (
        "Queue a message to another session in the same project; delivery is asynchronous."
    )
    risk_level = RiskLevel.network_access
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "message": {"type": "string", "minLength": 1, "maxLength": 8000},
        },
        "required": ["session_id", "message"],
        "additionalProperties": False,
    }

    def check_arguments(self, arguments: dict[str, Any]) -> None:
        from agent.session_inbox import validate_text

        validate_text(str(arguments["message"]))

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        self.check_arguments(arguments)
        if context.post_message is None:
            raise ToolError("Message posting is unavailable outside a tool call")
        return await context.post_message(str(arguments["session_id"]), str(arguments["message"]))
