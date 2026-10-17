from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy import select

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import Message, Project
from agent.runtime import AgentSupervisor
from agent.session_service import create_session_record
from agent.tools.base import ToolContext, ToolError


def write_skill(root: Path, name: str, description: str, body: str) -> None:
    """Helper to write a valid skill to the project directory."""
    skill_dir = root / ".agent" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


async def test_session_creation_loads_skills(tmp_path: Path) -> None:
    """Verify that skills are loaded into the system prompt during session creation."""
    project_root = tmp_path / "proj_skills"
    project_root.mkdir()
    write_skill(project_root, "math-helper", "Does math", "Use formulas.")

    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("System prompt base", encoding="utf-8")

    await init_database()
    async with session_factory() as db:
        project = Project(name="skills-proj", root_path=str(project_root))
        db.add(project)
        await db.flush()

        session = await create_session_record(
            db=db,
            settings=settings,
            project=project,
            prompt="Hello",
            mode="dev",
        )

        system_msg = await db.scalar(
            select(Message).where(Message.session_id == session.id, Message.role == "system")
        )
        assert system_msg is not None

        assert "Available skills:" in system_msg.content
        assert "- math-helper: Does math" in system_msg.content
        assert "read_skill" in system_msg.content


async def test_session_creation_no_skills_preserves_prompt(tmp_path: Path) -> None:
    """Verify that empty skill set doesn't add empty sections to prompt."""
    project_root = tmp_path / "proj_empty"
    project_root.mkdir()

    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("System prompt base", encoding="utf-8")

    await init_database()
    async with session_factory() as db:
        project = Project(name="empty-proj", root_path=str(project_root))
        db.add(project)
        await db.flush()

        session = await create_session_record(
            db=db,
            settings=settings,
            project=project,
            prompt="Hello",
            mode="dev",
        )

        system_msg = await db.scalar(
            select(Message).where(Message.session_id == session.id, Message.role == "system")
        )
        assert system_msg is not None

        assert "Available skills:" not in system_msg.content
        assert "read_skill" not in system_msg.content
        assert system_msg.content.startswith("System prompt base")


async def test_session_creation_rejects_symlink_skills(tmp_path: Path) -> None:
    """Verify that symlinked skill directories are ignored for security."""
    project_root = tmp_path / "proj_sym"
    project_root.mkdir()

    outside = tmp_path / "outside_skills"
    outside.mkdir()
    write_skill(outside, "evil-skill", "Evil", "Body")

    (project_root / ".agent").mkdir()
    os.symlink(outside, project_root / ".agent" / "skills")

    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("Base", encoding="utf-8")

    await init_database()
    async with session_factory() as db:
        project = Project(name="sym-proj", root_path=str(project_root))
        db.add(project)
        await db.flush()

        session = await create_session_record(db, settings, project, "Hello", "dev")

        system_msg = await db.scalar(
            select(Message).where(Message.session_id == session.id, Message.role == "system")
        )
        assert system_msg is not None

        assert "evil-skill" not in system_msg.content


async def test_runtime_scoped_registry_isolation(tmp_path: Path) -> None:
    """Verify that scoped registries provide isolation between concurrent runs."""
    root_a = tmp_path / "proj_a"
    root_a.mkdir()
    write_skill(root_a, "skill-a", "A desc", "A body")

    root_b = tmp_path / "proj_b"
    root_b.mkdir()
    write_skill(root_b, "skill-b", "B desc", "B body")

    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("Base", encoding="utf-8")
    await init_database()

    async with session_factory() as db:
        proj_a = Project(name="A", root_path=str(root_a))
        proj_b = Project(name="B", root_path=str(root_b))
        db.add_all([proj_a, proj_b])
        await db.commit()

        sess_a = await create_session_record(db, settings, proj_a, "Hello A", "dev")
        sess_b = await create_session_record(db, settings, proj_b, "Hello B", "dev")
        await db.commit()

    supervisor = AgentSupervisor(settings)
    entered = 0
    start_barrier = asyncio.Event()
    finished = 0
    finish_barrier = asyncio.Event()
    results: dict[str, dict[str, str]] = {}

    async def mock_run_session(session_id: str) -> None:
        nonlocal entered, finished
        entered += 1
        if entered == 2:
            start_barrier.set()

        await asyncio.wait_for(start_barrier.wait(), 2)

        read_tool = supervisor._tools.get("read_skill", "dev")

        actual_root = root_a if session_id == sess_a.id else root_b
        context = ToolContext(session_id=session_id, project_root=actual_root)

        session_results = {}
        for skill_name in ["skill-a", "skill-b"]:
            try:
                result = await read_tool.execute(context, {"name": skill_name})
                session_results[skill_name] = result["instructions"]
            except ToolError:
                session_results[skill_name] = "ToolError"

        results[session_id] = session_results

        finished += 1
        if finished == 2:
            finish_barrier.set()
        await asyncio.wait_for(finish_barrier.wait(), 2)

    supervisor._run_session = mock_run_session

    await asyncio.wait_for(
        asyncio.gather(supervisor._run(sess_a.id), supervisor._run(sess_b.id)), 5
    )

    res_a = results[sess_a.id]
    assert res_a["skill-a"] == "A body"
    assert res_a["skill-b"] == "ToolError"

    res_b = results[sess_b.id]
    assert res_b["skill-b"] == "B body"
    assert res_b["skill-a"] == "ToolError"
    assert supervisor._tools is supervisor.registry


async def test_runtime_cleanup_on_failure(tmp_path: Path) -> None:
    """Verify that the scoped registry is cleaned up even on exception."""
    root = tmp_path / "proj_fail"
    root.mkdir()
    write_skill(root, "fail-skill", "desc", "body")

    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("Base", encoding="utf-8")
    await init_database()

    async with session_factory() as db:
        proj = Project(name="Fail", root_path=str(root))
        db.add(proj)
        await db.commit()
        sess = await create_session_record(db, settings, proj, "Fail", "dev")
        await db.commit()

    supervisor = AgentSupervisor(settings)
    base_registry = supervisor.registry

    async def breaking_run(session_id: str) -> None:
        raise RuntimeError("Synthetic failure")

    supervisor._run_session = breaking_run

    with pytest.raises(RuntimeError, match="Synthetic failure"):
        await supervisor._run(sess.id)

    assert supervisor._tools is base_registry


async def test_runtime_cleanup_on_cancel(tmp_path: Path) -> None:
    """Verify that the scoped registry is cleaned up on cancellation."""
    root = tmp_path / "proj_cancel"
    root.mkdir()
    write_skill(root, "cancel-skill", "desc", "body")

    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("Base", encoding="utf-8")
    await init_database()

    async with session_factory() as db:
        proj = Project(name="Cancel", root_path=str(root))
        db.add(proj)
        await db.commit()
        sess = await create_session_record(db, settings, proj, "Cancel", "dev")
        await db.commit()

    supervisor = AgentSupervisor(settings)
    base_registry = supervisor.registry
    started = asyncio.Event()

    async def hanging_run(session_id: str) -> None:
        started.set()
        await asyncio.Event().wait()

    supervisor._run_session = hanging_run

    async def run_and_check() -> None:
        try:
            await supervisor._run(sess.id)
        except asyncio.CancelledError:
            assert supervisor._tools is supervisor.registry
            raise

    task = asyncio.create_task(run_and_check())

    await asyncio.wait_for(started.wait(), 2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)

    assert supervisor._tools is base_registry
