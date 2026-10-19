from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from agent.api import settings
from agent.auth import issue_login_code
from agent.config import Settings
from agent.database import init_database, session_factory
from agent.main import app
from agent.models import AuthBootstrapCode, Project, Session, User
from agent.runtime import AgentSupervisor
from agent.session_service import create_session_record
from tests.integration.helpers import ORIGIN


def bound_login(client: TestClient, *, bound: bool) -> str:
    async def issue() -> tuple[str, str]:
        async with session_factory() as db:
            user = User(display_name="Session identity")
            db.add(user)
            await db.commit()
            code = await issue_login_code(db, settings)
            if bound:
                await db.execute(
                    update(AuthBootstrapCode)
                    .where(AuthBootstrapCode.code_hash == hashlib.sha256(code.encode()).hexdigest())
                    .values(user_id=user.id)
                )
                await db.commit()
            return code, user.id

    code, user_id = client.portal.call(issue)
    response = client.post("/auth/login", json={"code": code}, headers={"Origin": ORIGIN})
    assert response.status_code == 200
    client.headers.update({"Origin": ORIGIN, "X-Agent-CSRF": response.json()["csrf_token"]})
    return user_id


def creator(client: TestClient, session_id: str) -> str | None:
    async def read() -> str | None:
        async with session_factory() as db:
            record = await db.get(Session, session_id)
            assert record is not None
            return record.created_by_user_id

    return client.portal.call(read)


@pytest.mark.parametrize("bound", [True, False])
def test_session_creator_comes_only_from_authenticated_cookie(tmp_path: Path, bound: bool) -> None:
    with TestClient(app) as client:
        user_id = bound_login(client, bound=bound)
        project = client.post(
            "/api/projects", json={"name": "identity", "root_path": str(tmp_path)}
        )
        assert project.status_code == 201
        created = client.post(
            "/api/sessions",
            json={
                "project_id": project.json()["id"],
                "prompt": "test",
                "mode": "dev",
                "auto_start": False,
                "created_by_user_id": "forged",
                "user_id": "forged",
            },
            headers={"X-User-ID": "forged", "X-Actor-User-ID": "forged"},
        )
        assert created.status_code == 201, created.text
        assert creator(client, created.json()["id"]) == (user_id if bound else None)


def test_unbound_api_token_does_not_borrow_cookie_identity(tmp_path: Path) -> None:
    previous = settings.api_token
    settings.api_token = "identity-test-token"
    try:
        with TestClient(app) as client:
            bound_login(client, bound=True)
            project = client.post(
                "/api/projects", json={"name": "identity-token", "root_path": str(tmp_path)}
            )
            assert project.status_code == 201
            created = client.post(
                "/api/sessions",
                json={
                    "project_id": project.json()["id"],
                    "prompt": "test",
                    "mode": "dev",
                    "auto_start": False,
                },
                headers={"X-Agent-Token": settings.api_token, "X-User-ID": "forged"},
            )
            assert created.status_code == 201, created.text
            assert creator(client, created.json()["id"]) is None
    finally:
        settings.api_token = previous


@pytest.mark.parametrize("bound", [True, False])
def test_fork_belongs_to_requesting_actor_not_source_creator(tmp_path: Path, bound: bool) -> None:
    with TestClient(app) as client:
        user_id = bound_login(client, bound=bound)
        project = client.post(
            "/api/projects", json={"name": "identity-fork", "root_path": str(tmp_path)}
        )
        assert project.status_code == 201
        response = client.post(
            "/api/sessions",
            json={
                "project_id": project.json()["id"],
                "prompt": "test",
                "mode": "dev",
                "auto_start": False,
            },
        )
        assert response.status_code == 201
        session_id = response.json()["id"]

        async def prepare() -> None:
            async with session_factory() as db:
                other = User(display_name="Source creator")
                db.add(other)
                await db.flush()
                await db.execute(
                    update(Session)
                    .where(Session.id == session_id)
                    .values(status="completed", created_by_user_id=other.id)
                )
                await db.commit()

        client.portal.call(prepare)
        messages = client.get(f"/api/sessions/{session_id}/messages").json()
        leaf = next(message["id"] for message in messages if message["role"] == "user")
        forked = client.post(f"/api/sessions/{session_id}/messages/{leaf}/fork", json={})
        assert forked.status_code == 201, forked.text
        assert creator(client, forked.json()["id"]) == (user_id if bound else None)


@pytest.mark.parametrize("bound", [True, False])
async def test_child_inherits_only_explicit_parent_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: bool
) -> None:
    await init_database(recover_running=False)
    (tmp_path / "dev.md").write_text("Test system prompt")
    config = Settings(prompts_dir=tmp_path)
    async with session_factory() as db:
        user = User(display_name="Parent creator")
        project = Project(name="child-identity", root_path=str(tmp_path))
        db.add_all([user, project])
        await db.flush()
        expected = user.id if bound else None
        parent = await create_session_record(
            db, config, project, "Parent", "dev", actor_user_id=expected
        )
    supervisor = AgentSupervisor(config)
    monkeypatch.setattr(supervisor, "start", AsyncMock(return_value=True))
    result = await supervisor._spawn_child(parent, "dev", "Child", False, None)
    async with session_factory() as db:
        child = await db.get(Session, result["session_id"])
        assert child is not None
        assert child.created_by_user_id == expected
