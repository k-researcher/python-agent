"""Session task plan tools for the agent (phase 4)."""

from __future__ import annotations

from typing import Any

from agent.database import session_factory
from agent.models import Session
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError

MAX_PLAN_ITEMS = 50
MAX_PLAN_TEXT_LENGTH = 500
MAX_PLAN_ID_LENGTH = 100
VALID_STATUSES = ("pending", "in_progress", "done")

_PLAN_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": MAX_PLAN_ID_LENGTH},
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_PLAN_TEXT_LENGTH,
        },
        "status": {"type": "string", "enum": list(VALID_STATUSES)},
    },
    "required": ["id", "text", "status"],
    "additionalProperties": False,
}


class TodoWriteTool(Tool):
    """Replace the session task plan stored in the session configuration."""

    name = "todo_write"
    description = "Replace the session task plan with up to 50 items."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev", "ask"})
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": _PLAN_ITEM_SCHEMA,
                "maxItems": MAX_PLAN_ITEMS,
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_items = arguments["items"]
        if not isinstance(raw_items, list):
            raise ToolError("Plan items must be a list")
        if len(raw_items) > MAX_PLAN_ITEMS:
            raise ToolError(f"Plan cannot exceed {MAX_PLAN_ITEMS} items")

        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        in_progress_count = 0
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ToolError("Each plan item must be an object")
            item_id = str(raw.get("id") or "")
            if not item_id:
                raise ToolError("Plan item id cannot be empty")
            if len(item_id) > MAX_PLAN_ID_LENGTH:
                raise ToolError(f"Plan item id exceeds {MAX_PLAN_ID_LENGTH} characters")
            if item_id in seen_ids:
                raise ToolError(f"Duplicate plan item id: {item_id}")
            seen_ids.add(item_id)
            text = str(raw.get("text") or "")
            if not text.strip():
                raise ToolError("Plan item text cannot be empty")
            if len(text) > MAX_PLAN_TEXT_LENGTH:
                raise ToolError(f"Plan item text exceeds {MAX_PLAN_TEXT_LENGTH} characters")
            status = str(raw.get("status") or "")
            if status not in VALID_STATUSES:
                raise ToolError(f"Unknown plan item status: {status}")
            if status == "in_progress":
                in_progress_count += 1
            items.append({"id": item_id, "text": text, "status": status})

        if in_progress_count > 1:
            raise ToolError("Only one plan item can be in_progress")

        from agent.message_tree import TreeError, lock_session

        async with session_factory() as db:
            # Lock the session row first (in a run: check the run fence), then read the current
            # configuration, so a concurrent change of another key is not lost.
            try:
                await lock_session(db, context.session_id)
            except TreeError as exc:
                raise ToolError("Session not found") from exc
            session = await db.get(Session, context.session_id, populate_existing=True)
            if session is None:
                raise ToolError("Session not found")
            configuration = dict(session.configuration or {})
            configuration["plan"] = items
            session.configuration = configuration
            await db.commit()

        counters: dict[str, int] = {
            "pending": sum(1 for item in items if item["status"] == "pending"),
            "in_progress": in_progress_count,
            "done": sum(1 for item in items if item["status"] == "done"),
        }
        return {"plan": items, "counters": counters}


class TodoReadTool(Tool):
    """Read the current session task plan."""

    name = "todo_read"
    description = "Read the current session task plan."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev", "ask"})
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        async with session_factory() as db:
            session = await db.get(Session, context.session_id)
            if session is None:
                raise ToolError("Session not found")
            plan = session.configuration.get("plan", []) if session.configuration else []
        return {"plan": plan}
