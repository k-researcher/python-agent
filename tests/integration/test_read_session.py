from __future__ import annotations

from pathlib import Path

import pytest

from agent.database import init_database, session_factory
from agent.models import Message, Project, Session
from agent.tools.base import ToolContext, ToolError
from agent.tools.session_tools import MAX_MESSAGE_CHARS, ReadSessionTool


async def make_sessions(tmp_path: Path) -> tuple[str, str, str]:
    """Create project A with a parent/child pair and project B with one foreign session."""
    await init_database(recover_running=False)
    async with session_factory() as db:
        project_a = Project(name="a", root_path=str(tmp_path / "a"))
        project_b = Project(name="b", root_path=str(tmp_path / "b"))
        db.add_all([project_a, project_b])
        await db.flush()

        parent = Session(project_id=project_a.id, title="parent", status="running")
        db.add(parent)
        await db.flush()
        child = Session(
            project_id=project_a.id,
            title="child",
            status="completed",
            parent_id=parent.id,
        )
        foreign = Session(project_id=project_b.id, title="foreign", status="completed")
        db.add_all([child, foreign])
        await db.flush()

        db.add(Message(session_id=parent.id, role="user", content="A question"))
        db.add(
            Message(
                session_id=parent.id,
                role="assistant",
                content="An answer",
                tool_calls=[
                    {"type": "function", "function": {"name": "read_file"}},
                    {"type": "function", "function": {"name": "search_files"}},
                ],
            )
        )
        db.add(Message(session_id=child.id, role="user", content="Child task"))
        db.add(Message(session_id=foreign.id, role="user", content="Foreign task"))
        await db.commit()
        return parent.id, child.id, foreign.id


def context(session_id: str, tmp_path: Path) -> ToolContext:
    return ToolContext(session_id=session_id, project_root=tmp_path)


async def test_read_own_and_child_session(tmp_path: Path) -> None:
    parent_id, child_id, _ = await make_sessions(tmp_path)
    tool = ReadSessionTool()
    ctx = context(parent_id, tmp_path)

    own = await tool.execute(ctx, {"session_id": parent_id})
    assert own["session_id"] == parent_id
    assert own["status"] == "running"
    assert own["title"] == "parent"
    assert [m["role"] for m in own["messages"]] == ["user", "assistant"]
    assert own["messages"][1]["tools"] == ["read_file", "search_files"]
    assert own["messages"][0]["content"] == "A question"

    child = await tool.execute(ctx, {"session_id": child_id})
    assert child["session_id"] == child_id
    assert child["status"] == "completed"
    assert child["title"] == "child"
    assert child["messages"][0]["content"] == "Child task"


async def test_foreign_project_raises_not_found(tmp_path: Path) -> None:
    parent_id, _, foreign_id = await make_sessions(tmp_path)
    tool = ReadSessionTool()
    ctx = context(parent_id, tmp_path)

    with pytest.raises(ToolError, match="Session not found"):
        await tool.execute(ctx, {"session_id": foreign_id})


async def test_nonexistent_session_raises_same_error(tmp_path: Path) -> None:
    parent_id, _, _ = await make_sessions(tmp_path)
    tool = ReadSessionTool()
    ctx = context(parent_id, tmp_path)

    with pytest.raises(ToolError, match="Session not found"):
        await tool.execute(ctx, {"session_id": "does-not-exist"})


async def test_limit_caps_number_of_messages(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name="limit", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="limit", status="completed")
        db.add(session)
        await db.flush()
        for index in range(5):
            db.add(Message(session_id=session.id, role="user", content=f"msg-{index}"))
        await db.commit()
        session_id = session.id

    tool = ReadSessionTool()
    result = await tool.execute(
        context(session_id, tmp_path), {"session_id": session_id, "limit": 3}
    )
    assert [m["content"] for m in result["messages"]] == ["msg-2", "msg-3", "msg-4"]


async def test_long_message_is_truncated_with_flag(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    long_text = "x" * (MAX_MESSAGE_CHARS + 100)
    async with session_factory() as db:
        project = Project(name="long", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="long", status="completed")
        db.add(session)
        await db.flush()
        db.add(Message(session_id=session.id, role="user", content=long_text))
        await db.commit()
        session_id = session.id

    tool = ReadSessionTool()
    result = await tool.execute(context(session_id, tmp_path), {"session_id": session_id})
    message = result["messages"][0]
    assert message["truncated"] is True
    assert len(message["content"]) == MAX_MESSAGE_CHARS
