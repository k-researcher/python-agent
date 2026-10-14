from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent import main
from agent.config import get_settings
from agent.database import session_factory
from agent.message_tree import append_message
from agent.models import Project, Session, SessionStatus
from tests.integration.helpers import login

TREE_PATHS = (
    ("POST", "/sessions/missing/messages/1/edit", {"content": "edited"}),
    ("POST", "/sessions/missing/messages/1/regenerate", {}),
    ("POST", "/sessions/missing/branch", {"leaf_id": 1}),
    ("POST", "/sessions/missing/messages/1/fork", {}),
    ("GET", "/sessions/missing/messages/1/siblings", None),
)


@pytest.fixture(params=[False, True], ids=["without-spa", "with-spa"])
def fresh_app(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FastAPI:
    directory = tmp_path / "dist"
    if request.param:
        directory.mkdir()
        (directory / "index.html").write_text("<html>tree mount test</html>", encoding="utf-8")
    monkeypatch.setattr(get_settings(), "frontend_dist", directory)
    specification = importlib.util.spec_from_file_location("agent._tree_mount_test", main.__file__)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert isinstance(module.app, FastAPI)
    return module.app


def test_tree_routes_are_in_fresh_application_schema(fresh_app: FastAPI) -> None:
    paths = fresh_app.openapi()["paths"]
    expected = (
        "/api/sessions/{session_id}/messages/{message_id}/edit",
        "/api/sessions/{session_id}/messages/{message_id}/regenerate",
        "/api/sessions/{session_id}/branch",
        "/api/sessions/{session_id}/messages/{message_id}/fork",
        "/api/sessions/{session_id}/messages/{message_id}/siblings",
    )
    for path in expected:
        assert path in paths
        method = "get" if path.endswith("/siblings") else "post"
        assert method in paths[path]


def test_siblings_uses_the_real_tree_handler(fresh_app: FastAPI, tmp_path: Path) -> None:
    async def seed() -> tuple[str, int]:
        async with session_factory() as database:
            project = Project(name="tree-mount", root_path=str(tmp_path))
            database.add(project)
            await database.flush()
            session = Session(
                project_id=project.id, title="tree mount", status=SessionStatus.completed.value
            )
            database.add(session)
            await database.flush()
            message = await append_message(database, session.id, role="user", content="question")
            await database.commit()
            return session.id, message.id

    with TestClient(fresh_app) as client:
        login(client)
        assert client.portal is not None
        session_id, message_id = client.portal.call(seed)
        response = client.get(f"/api/sessions/{session_id}/messages/{message_id}/siblings")
        assert response.status_code == 200
        assert response.json() == {"ids": [message_id], "active_index": 0}


@pytest.mark.parametrize(("method", "path", "payload"), TREE_PATHS)
def test_tree_routes_require_authentication(
    fresh_app: FastAPI, method: str, path: str, payload: dict[str, Any] | None
) -> None:
    with TestClient(fresh_app) as client:
        response = client.request(method, "/api" + path, json=payload)
        assert response.status_code in {401, 403}
        assert "detail" in response.json()


@pytest.mark.parametrize(("method", "path", "payload"), TREE_PATHS)
def test_missing_tree_rows_return_api_errors(
    fresh_app: FastAPI, method: str, path: str, payload: dict[str, Any] | None
) -> None:
    with TestClient(fresh_app) as client:
        login(client)
        response = client.request(method, "/api" + path, json=payload)
        assert response.status_code == 404
        assert "detail" in response.json()


@pytest.mark.parametrize(("method", "path", "payload"), TREE_PATHS[:4])
def test_tree_mutations_require_csrf(
    fresh_app: FastAPI, method: str, path: str, payload: dict[str, Any] | None
) -> None:
    with TestClient(fresh_app) as client:
        login(client)
        del client.headers["X-Agent-CSRF"]
        response = client.request(method, "/api" + path, json=payload)
        assert response.status_code == 403
        assert "detail" in response.json()
