from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from agent.database import init_database, session_factory
from agent.models import Project, Session, Workspace
from agent.tools.base import ToolContext, ToolError
from agent.tools.session_tools import GetProjectsTool


async def test_workspace_projects_are_sorted_and_mark_the_current_project(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="team")
        db.add(workspace)
        await db.flush()
        prefix = uuid4().hex[:30]
        current = Project(
            name="zeta", root_path=str(tmp_path / "current"), workspace_id=workspace.id
        )
        first = Project(
            id=f"{prefix}-a",
            name="alpha",
            root_path=str(tmp_path / "first"),
            workspace_id=workspace.id,
        )
        second = Project(
            id=f"{prefix}-z",
            name="alpha",
            root_path=str(tmp_path / "second"),
            workspace_id=workspace.id,
        )
        db.add_all([current, second, first])
        await db.flush()
        session = Session(project_id=current.id, title="current", status="running")
        db.add(session)
        await db.commit()
        session_id = session.id
        expected = [
            {"id": row.id, "name": row.name, "root_path": row.root_path, "current": row is current}
            for row in (first, second, current)
        ]

    result = await GetProjectsTool().execute(
        ToolContext(session_id=session_id, project_root=tmp_path), {}
    )
    assert result == {"projects": expected}


async def test_other_workspaces_and_unscoped_projects_are_not_visible(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="team")
        foreign_workspace = Workspace(name="foreign")
        db.add_all([workspace, foreign_workspace])
        await db.flush()
        current = Project(
            name="current", root_path=str(tmp_path / "current"), workspace_id=workspace.id
        )
        foreign = Project(
            name="foreign", root_path=str(tmp_path / "foreign"), workspace_id=foreign_workspace.id
        )
        unscoped = Project(name="unscoped", root_path=str(tmp_path / "unscoped"))
        db.add_all([current, foreign, unscoped])
        await db.flush()
        session = Session(project_id=current.id, title="current", status="completed")
        db.add(session)
        await db.commit()
        session_id, current_id = session.id, current.id

    result = await GetProjectsTool().execute(
        ToolContext(session_id=session_id, project_root=tmp_path), {}
    )
    assert [row["id"] for row in result["projects"]] == [current_id]


async def test_unscoped_session_sees_only_unscoped_projects(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="team")
        db.add(workspace)
        await db.flush()
        current = Project(name="unscoped", root_path=str(tmp_path / "current"))
        other = Project(name="other", root_path=str(tmp_path / "other"))
        foreign = Project(
            name="team", root_path=str(tmp_path / "foreign"), workspace_id=workspace.id
        )
        db.add_all([current, other, foreign])
        await db.flush()
        session = Session(project_id=current.id, title="current", status="completed")
        db.add(session)
        await db.commit()
        session_id, current_id, other_id, foreign_id = session.id, current.id, other.id, foreign.id
        unscoped_ids = set(
            (await db.scalars(select(Project.id).where(Project.workspace_id.is_(None)))).all()
        )

    result = await GetProjectsTool().execute(
        ToolContext(session_id=session_id, project_root=tmp_path), {}
    )
    projects = result["projects"]
    assert {row["id"] for row in projects} == unscoped_ids
    assert current_id in unscoped_ids and other_id in unscoped_ids
    assert foreign_id not in unscoped_ids
    assert [row["id"] for row in projects if row["current"]] == [current_id]
    assert projects == sorted(projects, key=lambda row: (row["name"], row["id"]))


async def test_missing_session_is_not_found(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    with pytest.raises(ToolError, match="^Session not found$"):
        await GetProjectsTool().execute(
            ToolContext(session_id="does-not-exist", project_root=tmp_path), {}
        )
