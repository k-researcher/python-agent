"""MCP servers from the UI: the API, the stored secrets and the sessions they serve."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent import api, mcp_api
from agent.config import Settings, get_settings
from agent.database import session_factory
from agent.mcp_servers import to_source, ui_sources
from agent.model_overrides import decrypt_secret
from agent.models import McpServer, Project, Session, Workspace
from agent.source_access import can_use_session_source
from agent.source_binding import source_scope
from tests.integration.helpers import authed_client

SECRET = "value-that-must-stay-hidden"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An app with one file source named "files" and a program in a narrow directory."""
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {"sources": [{"source_id": "files", "workspace_id": "w", "command": ["server"]}]}
        )
    )
    monkeypatch.setattr(api.settings, "tool_sources_file", sources)
    app_settings = api.settings
    from agent.main import app

    app.dependency_overrides[get_settings] = lambda: app_settings
    try:
        with authed_client() as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_settings, None)


def run(client: TestClient, call: Any) -> Any:
    return client.portal.call(call)  # type: ignore[union-attr]


async def stored(server_id: str) -> McpServer | None:
    async with session_factory() as db:
        return await db.get(McpServer, server_id)


def create(client: TestClient, **values: Any) -> Any:
    body = {"id": "docs", "command": ["/opt/tools/bin/docs-server", "--stdio"], **values}
    return client.post("/api/mcp/servers", json=body)


def by_id(body: dict[str, Any]) -> dict[str, Any]:
    return {server["id"]: server for server in body["servers"]}


def test_secret_values_are_encrypted_and_never_returned(client: TestClient) -> None:
    response = create(client, env={"DOCS_TOKEN": SECRET}, read_paths=["/opt/tools"])
    assert response.status_code == 201, response.text
    assert SECRET not in response.text
    listed = client.get("/api/mcp/servers")
    assert SECRET not in listed.text
    view = by_id(listed.json())["docs"]
    assert view["env_keys"] == ["DOCS_TOKEN"]
    assert view["read_paths"] == ["/opt/tools"]
    assert view["allowed_modes"] == ["dev"]
    assert listed.json()["file_sources"] == ["files"]
    row = run(client, lambda: stored("docs"))
    assert row is not None
    ciphertext = row.env_secrets["DOCS_TOKEN"]
    assert ciphertext != SECRET and SECRET not in ciphertext
    assert decrypt_secret(ciphertext, api.settings) == SECRET


@pytest.mark.parametrize(
    "values",
    [
        {"id": "Docs"},
        {"id": "files"},
        {"command": []},
        {"command": [""]},
        {"env": {"PATH": "/tmp"}},
        {"env": {"DYLD_INSERT_LIBRARIES": "/tmp/x.dylib"}},
        {"read_paths": ["/"]},
        {"read_paths": ["relative"]},
        {"allowed_modes": []},
        {"allowed_modes": ["root"]},
        {"risk_level": "read_only"},
    ],
)
def test_unsafe_or_invalid_definitions_are_refused(
    client: TestClient, values: dict[str, Any]
) -> None:
    response = create(client, **values)
    assert response.status_code == 422, response.text
    assert run(client, lambda: stored(values.get("id", "docs"))) is None


def test_read_paths_cannot_open_the_agent_data(client: TestClient) -> None:
    for path in (api.settings.data_dir, api.settings.data_dir.parent):
        response = create(client, read_paths=[str(path)])
        assert response.status_code == 422, response.text


def test_duplicate_id_is_a_conflict(client: TestClient) -> None:
    assert create(client).status_code == 201
    assert create(client).status_code == 409


def test_projects_must_exist_and_have_no_workspace(client: TestClient, tmp_path: Path) -> None:
    async def team_project() -> str:
        async with session_factory() as db:
            workspace = Workspace(name="Team")
            db.add(workspace)
            await db.flush()
            project = Project(
                name="Team", root_path=str(tmp_path / "team"), workspace_id=workspace.id
            )
            db.add(project)
            await db.commit()
            return project.id

    team = run(client, team_project)
    assert create(client, project_ids=[team]).status_code == 422
    assert create(client, project_ids=["missing"]).status_code == 422
    (tmp_path / "own").mkdir()
    own = client.post("/api/projects", json={"name": "own", "root_path": str(tmp_path / "own")})
    assert own.status_code == 201, own.text
    created = create(client, project_ids=[own.json()["id"]])
    assert created.status_code == 201, created.text
    assert by_id(created.json())["docs"]["project_ids"] == [own.json()["id"]]


def test_update_keeps_replaces_and_removes_secrets(client: TestClient) -> None:
    create(client, env={"KEEP": "one", "DROP": "two", "SWAP": "three"})
    before = run(client, lambda: stored("docs"))
    assert before is not None
    response = client.put(
        "/api/mcp/servers/docs", json={"env": {"DROP": None, "SWAP": "four", "NEW": "five"}}
    )
    assert response.status_code == 200, response.text
    assert by_id(response.json())["docs"]["env_keys"] == ["KEEP", "NEW", "SWAP"]
    after = run(client, lambda: stored("docs"))
    assert after is not None
    assert after.env_secrets["KEEP"] == before.env_secrets["KEEP"]
    assert decrypt_secret(after.env_secrets["SWAP"], api.settings) == "four"
    assert "four" not in response.text and "five" not in response.text


def test_a_new_launch_drops_the_old_check_but_a_switch_keeps_it(client: TestClient) -> None:
    create(client)

    async def mark() -> None:
        async with session_factory() as db:
            row = await db.get(McpServer, "docs")
            assert row is not None
            row.last_check = {"ok": True, "tools": []}
            await db.commit()

    run(client, mark)
    off = client.put("/api/mcp/servers/docs", json={"enabled": False})
    assert by_id(off.json())["docs"]["last_check"] == {"ok": True, "tools": []}
    assert by_id(off.json())["docs"]["enabled"] is False
    moved = client.put("/api/mcp/servers/docs", json={"command": ["other-server"]})
    assert by_id(moved.json())["docs"]["last_check"] is None
    run(client, mark)
    network = client.put("/api/mcp/servers/docs", json={"allow_network": True})
    assert by_id(network.json())["docs"]["last_check"] is None


def test_delete_removes_the_server(client: TestClient) -> None:
    create(client)
    response = client.delete("/api/mcp/servers/docs")
    assert response.status_code == 200
    assert "docs" not in by_id(response.json())
    assert client.delete("/api/mcp/servers/docs").status_code == 404
    assert client.post("/api/mcp/servers/docs/check").status_code == 404


def test_check_keeps_its_result(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    create(client)
    seen: list[list[str]] = []

    async def fake_check(settings: Settings, server: McpServer) -> dict[str, Any]:
        seen.append(list(server.command))
        return {"ok": True, "checked_at": "now", "tools": [{"name": "search"}], "tool_count": 1}

    monkeypatch.setattr(mcp_api, "check_server", fake_check)
    response = client.post("/api/mcp/servers/docs/check")
    assert response.status_code == 200, response.text
    assert by_id(response.json())["docs"]["last_check"]["tools"] == [{"name": "search"}]
    assert seen == [["/opt/tools/bin/docs-server", "--stdio"]]


def test_check_result_of_an_old_launch_is_not_kept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    create(client)

    async def change_during_check(settings: Settings, server: McpServer) -> dict[str, Any]:
        async with session_factory() as db:
            row = await db.get(McpServer, "docs")
            assert row is not None
            row.command = ["changed"]
            await db.commit()
        return {"ok": True, "tools": []}

    monkeypatch.setattr(mcp_api, "check_server", change_during_check)
    response = client.post("/api/mcp/servers/docs/check")
    assert response.status_code == 409
    row = run(client, lambda: stored("docs"))
    assert row is not None and row.last_check is None


def test_one_check_at_a_time(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    create(client)
    calls: list[str] = []

    async def fake_check(settings: Settings, server: McpServer) -> dict[str, Any]:
        calls.append(server.id)
        return {"ok": True, "tools": []}

    monkeypatch.setattr(mcp_api, "check_server", fake_check)
    # A check of this server is running: a second one must wait for it.
    monkeypatch.setattr(mcp_api, "_checking", {"docs"})
    assert client.post("/api/mcp/servers/docs/check").status_code == 409
    assert calls == []
    # When that check ends, the next one runs.
    monkeypatch.setattr(mcp_api, "_checking", set())
    assert client.post("/api/mcp/servers/docs/check").status_code == 200
    assert calls == ["docs"]
    assert mcp_api._checking == set()


def test_failed_check_releases_the_server(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    create(client)

    async def broken_check(settings: Settings, server: McpServer) -> dict[str, Any]:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(mcp_api, "check_server", broken_check)
    with pytest.raises(RuntimeError):
        client.post("/api/mcp/servers/docs/check")
    assert mcp_api._checking == set()


async def _session(project_id: str, mode: str = "dev") -> str:
    async with session_factory() as db:
        session = Session(project_id=project_id, mode=mode, title="t", configuration={})
        db.add(session)
        await db.commit()
        return session.id


def test_enabled_ui_servers_serve_sessions_of_their_projects(
    client: TestClient, tmp_path: Path
) -> None:
    roots = [tmp_path / "a", tmp_path / "b"]
    projects = []
    for root in roots:
        root.mkdir()
        response = client.post("/api/projects", json={"name": root.name, "root_path": str(root)})
        projects.append(response.json()["id"])
    create(client, project_ids=[projects[0]], env={"DOCS_TOKEN": SECRET})
    create(client, id="off", enabled=False)

    async def check() -> tuple[list[str], dict[str, str], bool, bool, bool]:
        async with session_factory() as db:
            found = await ui_sources(db, api.settings)
            source, environment = next(item for item in found if item[0].source_id == "docs")
            scope = source_scope(source)
            first = await _session(projects[0])
            second = await _session(projects[1])
            ask = await _session(projects[0], mode="ask")
            return (
                [item[0].source_id for item in found],
                environment,
                await can_use_session_source(db, first, scope),
                await can_use_session_source(db, second, scope),
                await can_use_session_source(db, ask, scope),
            )

    ids, environment, first, second, ask = run(client, check)
    assert ids == ["docs"]
    assert environment["DOCS_TOKEN"] == SECRET
    assert (first, second, ask) == (True, False, False)


def test_ui_server_never_serves_a_workspace_project(client: TestClient, tmp_path: Path) -> None:
    create(client)

    async def check() -> bool:
        async with session_factory() as db:
            workspace = Workspace(name="Team")
            db.add(workspace)
            await db.flush()
            project = Project(name="T", root_path=str(tmp_path / "t"), workspace_id=workspace.id)
            db.add(project)
            await db.commit()
            session_id = await _session(project.id)
            row = (await db.scalars(select(McpServer).where(McpServer.id == "docs"))).one()
            return await can_use_session_source(db, session_id, source_scope(to_source(row)))

    assert run(client, check) is False
