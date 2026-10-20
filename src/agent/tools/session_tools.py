"""Read-only access to the conversation of another session in the same project."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from agent.database import session_factory
from agent.models import Message, Session
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

            rows = list(
                (
                    await db.execute(
                        select(Message)
                        .where(
                            Message.session_id == target.id,
                            Message.kind == "normal",
                            Message.skipped.is_(False),
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
