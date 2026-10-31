"""The "now working" list: every agent across projects with its step, time and cost."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from agent.database import session_factory
from agent.models import OutboundAudit, Session, SessionStatus, ToolCall, ToolCallStatus
from tests.integration.helpers import authed_client


def _session(client: TestClient, project_id: str, title: str) -> str:
    response = client.post(
        "/api/sessions",
        json={"project_id": project_id, "prompt": title, "mode": "dev", "auto_start": False},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _prepare(working: str, child: str, parent: str) -> None:
    async def apply() -> None:
        async with session_factory() as db:
            row = await db.get(Session, working)
            assert row is not None
            row.status = SessionStatus.running.value
            db.add(
                ToolCall(
                    id="call-monitor",
                    session_id=working,
                    name="write_file",
                    arguments={},
                    risk_level="write",
                    status=ToolCallStatus.running.value,
                )
            )
            db.add(
                OutboundAudit(
                    session_id=working,
                    category="llm",
                    destination="https://llm.example",
                    operation="chat:test",
                    status="completed",
                    prompt_tokens=10,
                    completion_tokens=20,
                    cost_amount="0.03",
                )
            )
            waiting = await db.get(Session, child)
            assert waiting is not None
            waiting.status = SessionStatus.awaiting_confirmation.value
            waiting.parent_id = parent
            await db.commit()

    asyncio.run(apply())


def test_monitor_lists_agents_with_step_project_and_cost(tmp_path: Path) -> None:
    with authed_client() as client:
        project = client.post(
            "/api/projects", json={"name": "Monitor project", "root_path": str(tmp_path)}
        ).json()["id"]
        working = _session(client, project, "Working agent")
        parent = _session(client, project, "Parent agent")
        child = _session(client, project, "Waiting child")
        _prepare(working, child, parent)

        response = client.get("/api/monitor")
        assert response.status_code == 200, response.text
        rows = {row["session_id"]: row for row in response.json()}

        assert rows[working] | {"duration_seconds": 0, "idle_seconds": 0} == {
            "session_id": working,
            "parent_id": None,
            "project_id": project,
            "project": "Monitor project",
            "title": "Working agent",
            "status": "running",
            "model": rows[working]["model"],
            "mode": "dev",
            "action_kind": "tool",
            "action_text": "write_file",
            "duration_seconds": 0,
            "idle_seconds": 0,
            "cost": "0.03",
            "has_unknown_cost": False,
        }
        assert rows[child]["status"] == "awaiting_confirmation"
        assert rows[child]["parent_id"] == parent
        order = [row["session_id"] for row in client.get("/api/monitor").json()]
        assert order.index(working) < order.index(child)


def test_monitor_keeps_agents_from_all_projects_beyond_the_cli_limit(tmp_path: Path) -> None:
    with authed_client() as client:
        for index in range(2):
            (tmp_path / str(index)).mkdir()
        projects = [
            client.post(
                "/api/projects",
                json={
                    "name": f"Project {index}",
                    "root_path": str(tmp_path / str(index)),
                },
            ).json()["id"]
            for index in range(2)
        ]
        ids = {_session(client, projects[index % 2], f"Agent {index}") for index in range(51)}
        response = client.get("/api/monitor")
        assert response.status_code == 200
        assert {row["session_id"] for row in response.json()} == ids
