"""Tests for the session task plan tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.database import init_database, session_factory
from agent.models import Project, Session
from agent.tools.base import ToolContext, ToolError
from agent.tools.todo_tools import MAX_PLAN_ITEMS, TodoReadTool, TodoWriteTool


async def make_session(tmp_path: Path, name: str = "todo") -> tuple[str, ToolContext]:
    """Create an isolated project and session with a tool context."""
    await init_database()
    async with session_factory() as db:
        project = Project(name=name, root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title=name, configuration={})
        db.add(session)
        await db.commit()
        return session.id, ToolContext(session_id=session.id, project_root=tmp_path)


def items_for(*statuses: str) -> list[dict[str, str]]:
    """Build a plan item list with ids and texts derived from statuses."""
    return [
        {"id": str(index), "text": f"Task {index}", "status": status}
        for index, status in enumerate(statuses)
    ]


async def test_write_and_read(tmp_path: Path) -> None:
    """todo_write stores a plan and todo_read returns it unchanged."""
    _sid, context = await make_session(tmp_path / "test-write-read")
    write_tool = TodoWriteTool()
    read_tool = TodoReadTool()

    items = items_for("pending", "in_progress", "done")
    result = await write_tool.execute(context, {"items": items})

    assert result["plan"] == items
    assert result["counters"] == {"pending": 1, "in_progress": 1, "done": 1}

    read_result = await read_tool.execute(context, {})
    assert read_result["plan"] == items


async def test_replace_plan(tmp_path: Path) -> None:
    """A second write replaces the previous plan completely."""
    _sid, context = await make_session(tmp_path / "test-replace")
    write_tool = TodoWriteTool()
    read_tool = TodoReadTool()

    first = items_for("pending", "done")
    await write_tool.execute(context, {"items": first})

    second = items_for("in_progress")
    result = await write_tool.execute(context, {"items": second})
    assert result["plan"] == second
    assert result["counters"] == {"pending": 0, "in_progress": 1, "done": 0}

    read_result = await read_tool.execute(context, {})
    assert read_result["plan"] == second


async def test_preserves_other_configuration_keys(tmp_path: Path) -> None:
    """Writing the plan does not overwrite other session configuration keys."""
    _sid, context = await make_session(tmp_path / "test-other-keys")
    async with session_factory() as db:
        session = await db.get(Session, context.session_id)
        assert session is not None
        session.configuration = {"theme": "dark", "mode": "focused"}
        await db.commit()

    items = items_for("pending")
    result = await TodoWriteTool().execute(context, {"items": items})
    assert result["plan"] == items

    async with session_factory() as db:
        session = await db.get(Session, context.session_id)
        assert session is not None
        assert session.configuration["theme"] == "dark"
        assert session.configuration["mode"] == "focused"
        assert session.configuration["plan"] == items


async def test_read_returns_empty_list_when_no_plan(tmp_path: Path) -> None:
    """todo_read returns an empty list when no plan is stored."""
    _sid, context = await make_session(tmp_path / "test-empty-read")
    result = await TodoReadTool().execute(context, {})
    assert result["plan"] == []


async def test_duplicate_id_raises_error(tmp_path: Path) -> None:
    """Writing duplicate item ids raises ToolError."""
    _sid, context = await make_session(tmp_path / "test-duplicate")
    items = [
        {"id": "same", "text": "First", "status": "pending"},
        {"id": "same", "text": "Second", "status": "pending"},
    ]
    with pytest.raises(ToolError, match="Duplicate plan item id"):
        await TodoWriteTool().execute(context, {"items": items})


async def test_two_in_progress_raises_error(tmp_path: Path) -> None:
    """Writing two in_progress items raises ToolError."""
    _sid, context = await make_session(tmp_path / "test-two-progress")
    items = items_for("in_progress", "in_progress")
    with pytest.raises(ToolError, match="Only one plan item can be in_progress"):
        await TodoWriteTool().execute(context, {"items": items})


async def test_empty_text_raises_error(tmp_path: Path) -> None:
    """Writing an item with empty or whitespace text raises ToolError."""
    _sid, context = await make_session(tmp_path / "test-empty-text")
    cases = [
        {"id": "1", "text": "", "status": "pending"},
        {"id": "1", "text": "   ", "status": "pending"},
        {"id": "1", "text": "\n\t", "status": "pending"},
    ]
    for case in cases:
        with pytest.raises(ToolError, match="text cannot be empty"):
            await TodoWriteTool().execute(context, {"items": [case]})


async def test_too_many_items_raises_error(tmp_path: Path) -> None:
    """Writing more than MAX_PLAN_ITEMS items raises ToolError."""
    _sid, context = await make_session(tmp_path / "test-too-many")
    items = [
        {"id": str(index), "text": f"Task {index}", "status": "pending"}
        for index in range(MAX_PLAN_ITEMS + 1)
    ]
    with pytest.raises(ToolError, match="cannot exceed"):
        await TodoWriteTool().execute(context, {"items": items})


async def test_unknown_status_raises_error(tmp_path: Path) -> None:
    """Writing an item with an unknown status raises ToolError."""
    _sid, context = await make_session(tmp_path / "test-unknown-status")
    items = [
        {"id": "1", "text": "Task", "status": "blocked"},
    ]
    with pytest.raises(ToolError, match="Unknown plan item status"):
        await TodoWriteTool().execute(context, {"items": items})


async def test_todo_write_rejects_a_too_long_id(tmp_path: Path) -> None:
    _session_id, context = await make_session(tmp_path, "long-id")
    with pytest.raises(ToolError, match="id exceeds"):
        await TodoWriteTool().execute(
            context, {"items": [{"id": "x" * 101, "text": "step", "status": "pending"}]}
        )
