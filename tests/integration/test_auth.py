from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.api import settings
from agent.auth import issue_login_code
from agent.database import session_factory
from agent.main import app
from tests.integration.helpers import ORIGIN, authed_client, login


def issue_code(client: TestClient) -> str:
    async def issue() -> str:
        async with session_factory() as db:
            return await issue_login_code(db, settings)

    return str(client.portal.call(issue))  # type: ignore[union-attr]


def test_api_requires_login_and_health_stays_public() -> None:
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/projects").status_code == 401
        assert client.get("/auth/session").json() == {"authenticated": False}


def test_rebinding_host_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health", headers={"Host": "evil.example"})
    assert response.status_code == 400


def test_login_code_is_single_use_and_sets_hardened_cookie() -> None:
    with TestClient(app) as client:
        code = issue_code(client)
        first = client.post("/auth/login", json={"code": code}, headers={"Origin": ORIGIN})
        assert first.status_code == 200
        cookie = first.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
        assert first.headers["cache-control"] == "no-store"

        second = client.post("/auth/login", json={"code": code}, headers={"Origin": ORIGIN})
        assert second.status_code == 401


def test_login_requires_allowed_origin() -> None:
    with TestClient(app) as client:
        code = issue_code(client)
        missing = client.post("/auth/login", json={"code": code})
        foreign = client.post(
            "/auth/login", json={"code": code}, headers={"Origin": "https://evil.example"}
        )
    assert missing.status_code == 403
    assert foreign.status_code == 403


def test_mutations_require_origin_and_csrf(tmp_path: object) -> None:
    with authed_client() as client:
        assert client.get("/api/projects").status_code == 200
        body = {"name": "csrf", "root_path": "/"}

        no_csrf = client.post(
            "/api/projects", json=body, headers={"X-Agent-CSRF": "wrong-token-value"}
        )
        assert no_csrf.status_code == 403

        foreign = client.post("/api/projects", json=body, headers={"Origin": "https://evil.example"})
        assert foreign.status_code == 403


def test_logout_revokes_session() -> None:
    with authed_client() as client:
        assert client.post("/auth/logout").status_code == 204
        assert client.get("/api/projects").status_code == 401


def test_api_token_still_works_for_cli_but_not_with_foreign_origin() -> None:
    original = settings.api_token
    settings.api_token = "integration-secret"
    try:
        with TestClient(app) as client:
            ok = client.get("/api/ready", headers={"X-Agent-Token": "integration-secret"})
            wrong = client.get("/api/ready", headers={"X-Agent-Token": "nope"})
            foreign = client.get(
                "/api/ready",
                headers={"X-Agent-Token": "integration-secret", "Origin": "https://evil.example"},
            )
        assert ok.status_code == 200
        assert wrong.status_code == 401
        assert foreign.status_code == 403
    finally:
        settings.api_token = original


def test_websocket_requires_session_cookie_and_origin() -> None:
    with TestClient(app) as client:
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws", headers={"Origin": ORIGIN}),
        ):
            pass

        login(client)
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}),
        ):
            pass
        with client.websocket_connect("/ws", headers={"Origin": ORIGIN}) as websocket:
            assert websocket.receive_json()["type"] == "connected"
