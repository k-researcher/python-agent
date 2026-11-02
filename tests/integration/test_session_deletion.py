"""Sessions and projects can be archived, restored and deleted without touching files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from agent import api
from agent.database import session_factory
from agent.models import Message, OutboundAudit, Project, Session, Workspace
from tests.integration.test_settings_api import client

__all__ = ["client"]


def run(client: TestClient, work: Any) -> Any:
    assert client.portal is not None
    return client.portal.call(work)


def seed_project(
    client: TestClient, root: Path, name: str = "Demo", workspace: bool = False
) -> str:
    root.mkdir(parents=True, exist_ok=True)

    async def seed() -> str:
        async with session_factory() as db:
            workspace_id = None
            if workspace:
                team = Workspace(name="team")
                db.add(team)
                await db.flush()
                workspace_id = team.id
            project = Project(name=name, root_path=str(root), workspace_id=workspace_id)
            db.add(project)
            await db.commit()
            return project.id

    return run(client, seed)


def seed_session(
    client: TestClient, project_id: str, status: str = "completed", parent_id: str | None = None
) -> str:
    async def seed() -> str:
        async with session_factory() as db:
            session = Session(
                project_id=project_id,
                parent_id=parent_id,
                title=f"{status} session",
                status=status,
                llm_profile="fast",
                run_generation=0,
            )
            db.add(session)
            await db.flush()
            db.add(Message(session_id=session.id, role="user", content="history"))
            db.add(
                OutboundAudit(
                    session_id=session.id,
                    category="llm",
                    destination="example.com",
                    operation="chat",
                )
            )
            await db.commit()
            return session.id

    return run(client, seed)


def counts(client: TestClient, ids: list[str]) -> tuple[int, int, int]:
    async def read() -> tuple[int, int, int]:
        async with session_factory() as db:
            sessions = await db.scalar(select(func.count()).where(Session.id.in_(ids)))
            messages = await db.scalar(select(func.count()).where(Message.session_id.in_(ids)))
            audit = await db.scalar(select(func.count()).select_from(OutboundAudit))
            return sessions or 0, messages or 0, audit or 0

    return run(client, read)


def test_a_session_with_children_is_deleted_only_on_explicit_request(
    client: TestClient, tmp_path: Path
) -> None:
    project = seed_project(client, tmp_path / "demo")
    root = seed_session(client, project)
    child = seed_session(client, project, "stopped", root)
    grandchild = seed_session(client, project, "error", child)
    other = seed_session(client, project)
    tree = [root, child, grandchild]
    _, _, audit_before = counts(client, tree)

    refused = client.delete(f"/api/sessions/{root}")
    assert refused.status_code == 409
    assert counts(client, tree)[:2] == (3, 3)

    assert client.delete(f"/api/sessions/{root}?with_children=true").status_code == 204
    assert counts(client, tree) == (0, 0, audit_before)
    assert client.get(f"/api/sessions/{child}").status_code == 404
    assert client.get(f"/api/sessions/{other}").status_code == 200


def test_a_session_without_children_is_deleted_as_before(
    client: TestClient, tmp_path: Path
) -> None:
    project = seed_project(client, tmp_path / "demo")
    alone = seed_session(client, project, "interrupted")
    assert client.delete(f"/api/sessions/{alone}").status_code == 204
    assert counts(client, [alone])[:2] == (0, 0)
    assert client.delete(f"/api/sessions/{alone}").status_code == 404


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
def test_an_active_session_or_child_blocks_the_whole_deletion(
    client: TestClient, tmp_path: Path, status: str
) -> None:
    project = seed_project(client, tmp_path / "demo")
    active_root = seed_session(client, project, status)
    assert client.delete(f"/api/sessions/{active_root}?with_children=true").status_code == 409

    root = seed_session(client, project)
    child = seed_session(client, project, status, root)
    response = client.delete(f"/api/sessions/{root}?with_children=true")
    assert response.status_code == 409
    assert "Остановите дочерние сессии" in response.json()["detail"]
    assert counts(client, [active_root, root, child])[:2] == (3, 3)


def test_a_run_that_starts_after_the_check_keeps_every_session(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = seed_project(client, tmp_path / "demo")
    root = seed_session(client, project)
    child = seed_session(client, project, "running", root)

    async def stale_check(*_args: Any) -> None:
        return None

    monkeypatch.setattr(api, "_check_deletable", stale_check)
    response = client.delete(f"/api/sessions/{root}?with_children=true")
    assert response.status_code == 409
    assert counts(client, [root, child])[:2] == (2, 2)


def test_an_archived_session_can_be_restored(client: TestClient, tmp_path: Path) -> None:
    project = seed_project(client, tmp_path / "demo")
    session = seed_session(client, project)

    def listed(archived: bool) -> list[str]:
        response = client.get(f"/api/sessions?archived={str(archived).lower()}")
        return [row["id"] for row in response.json()]

    assert client.post(f"/api/sessions/{session}/archive").json() == {"archived": True}
    assert session in listed(True) and session not in listed(False)
    assert client.post(f"/api/sessions/{session}/unarchive").json() == {"archived": False}
    assert session in listed(False) and session not in listed(True)
    assert client.post("/api/sessions/absent/unarchive").status_code == 404


def test_a_project_is_deleted_with_its_history_but_its_files_stay(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "demo"
    project = seed_project(client, folder)
    (folder / "notes.txt").write_text("keep me")
    root = seed_session(client, project)
    child = seed_session(client, project, "stopped", root)
    archived = seed_session(client, project)
    assert client.post(f"/api/sessions/{archived}/archive").status_code == 200
    kept_project = seed_project(client, tmp_path / "other", "Other")
    kept = seed_session(client, kept_project)

    wrong = client.delete(f"/api/projects/{project}", params={"name": "demo"})
    assert wrong.status_code == 409
    assert counts(client, [root, child, archived])[:2] == (3, 3)

    response = client.delete(f"/api/projects/{project}", params={"name": "Demo"})
    assert response.status_code == 204
    assert counts(client, [root, child, archived])[:2] == (0, 0)
    assert [row["id"] for row in client.get("/api/projects").json()] == [kept_project]
    assert client.get(f"/api/sessions/{kept}").status_code == 200
    assert (folder / "notes.txt").read_text() == "keep me"

    again = client.post("/api/projects", json={"name": "Demo", "root_path": str(folder)})
    assert again.status_code == 201


def test_a_project_with_an_active_session_is_kept(client: TestClient, tmp_path: Path) -> None:
    project = seed_project(client, tmp_path / "demo")
    idle = seed_session(client, project)
    busy = seed_session(client, project, "running")
    response = client.delete(f"/api/projects/{project}", params={"name": "Demo"})
    assert response.status_code == 409
    assert "Остановите все сессии проекта" in response.json()["detail"]
    assert counts(client, [idle, busy])[:2] == (2, 2)
    assert any(row["id"] == project for row in client.get("/api/projects").json())


def test_project_deletion_refuses_unknown_and_workspace_projects(
    client: TestClient, tmp_path: Path
) -> None:
    assert client.delete("/api/projects/absent", params={"name": "x"}).status_code == 404
    team = seed_project(client, tmp_path / "team", "Team", workspace=True)
    response = client.delete(f"/api/projects/{team}", params={"name": "Team"})
    assert response.status_code == 409
    assert client.delete(f"/api/projects/{team}").status_code == 422
