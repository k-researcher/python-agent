from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import update

from agent.api import settings
from agent.auth import issue_login_code
from agent.database import session_factory
from agent.main import app
from agent.models import AuthBootstrapCode, Session, User
from tests.integration.helpers import ORIGIN


def bound_login(client: TestClient, name: str) -> str:
    """Create a fresh bound user, issue a one-time code for it and log the client in."""

    async def issue() -> tuple[str, str]:
        async with session_factory() as db:
            user = User(display_name=name)
            db.add(user)
            await db.commit()
            code = await issue_login_code(db, settings)
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


def create_session(client: TestClient, project_id: str, prompt: str) -> str:
    """POST /api/sessions with forged authority in both payload and headers."""
    response = client.post(
        "/api/sessions",
        json={
            "project_id": project_id,
            "prompt": prompt,
            "mode": "dev",
            "auto_start": False,
            "created_by_user_id": "forged",
            "user_id": "forged",
        },
        headers={"X-User-ID": "forged", "X-Actor-User-ID": "forged"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_alternating_cookie_identities_do_not_leak_across_requests(tmp_path: Path) -> None:
    with TestClient(app) as client_a, TestClient(app) as client_b:
        user_a = bound_login(client_a, "Actor A")
        user_b = bound_login(client_b, "Actor B")
        assert user_a != user_b

        project = client_a.post(
            "/api/projects", json={"name": "alternating", "root_path": str(tmp_path)}
        )
        assert project.status_code == 201, project.text
        project_id = project.json()["id"]

        s1 = create_session(client_a, project_id, "a-1")
        s2 = create_session(client_b, project_id, "b-1")
        s3 = create_session(client_a, project_id, "a-2")
        s4 = create_session(client_b, project_id, "b-2")

        assert creator(client_a, s1) == user_a
        assert creator(client_b, s2) == user_b
        assert creator(client_a, s3) == user_a
        assert creator(client_b, s4) == user_b


def test_same_cookie_then_token_then_cookie_records_X_none_X(tmp_path: Path) -> None:
    previous = settings.api_token
    settings.api_token = "actor-isolation-token"
    try:
        with TestClient(app) as client:
            user_id = bound_login(client, "Cookie actor")
            project = client.post(
                "/api/projects", json={"name": "sequence", "root_path": str(tmp_path)}
            )
            assert project.status_code == 201, project.text
            project_id = project.json()["id"]

            cookie_first = create_session(client, project_id, "cookie-first")
            assert creator(client, cookie_first) == user_id

            token_response = client.post(
                "/api/sessions",
                json={
                    "project_id": project_id,
                    "prompt": "token-mid",
                    "mode": "dev",
                    "auto_start": False,
                    "created_by_user_id": "forged",
                    "user_id": "forged",
                },
                headers={
                    "X-Agent-Token": settings.api_token,
                    "X-User-ID": "forged",
                    "X-Actor-User-ID": "forged",
                },
            )
            assert token_response.status_code == 201, token_response.text
            token_id = token_response.json()["id"]
            assert creator(client, token_id) is None

            cookie_last = create_session(client, project_id, "cookie-last")
            assert creator(client, cookie_last) == user_id
    finally:
        settings.api_token = previous
