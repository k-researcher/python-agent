"""The model of an existing session can be changed between runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from agent import api
from agent.database import session_factory
from agent.models import Event, Session
from agent.schemas import SessionModelChange
from tests.integration.test_session_deletion import run, seed_project, seed_session
from tests.integration.test_settings_api import client

__all__ = ["client"]


def model_of(client: TestClient, session_id: str) -> str | None:
    async def read() -> str | None:
        async with session_factory() as db:
            return await db.scalar(select(Session.llm_profile).where(Session.id == session_id))

    return run(client, read)


def change_events(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    async def read() -> list[dict[str, Any]]:
        async with session_factory() as db:
            rows = await db.scalars(
                select(Event.payload).where(
                    Event.session_id == session_id, Event.type == "session.model_changed"
                )
            )
            return [dict(row) for row in rows]

    return run(client, read)


def test_a_stopped_session_continues_on_another_model(client: TestClient, tmp_path: Path) -> None:
    project = seed_project(client, tmp_path / "demo")
    session = seed_session(client, project, "stopped")

    response = client.post(f"/api/sessions/{session}/model", json={"llm_profile": "spare"})
    assert response.status_code == 200
    assert response.json()["llm_profile"] == "spare"
    assert response.json()["status"] == "stopped"
    assert model_of(client, session) == "spare"
    assert change_events(client, session) == [{"llm_profile": "spare", "previous": "fast"}]

    same = client.post(f"/api/sessions/{session}/model", json={"llm_profile": "spare"})
    assert same.status_code == 200
    assert len(change_events(client, session)) == 1


def test_a_session_whose_model_was_removed_can_pick_a_working_one(
    client: TestClient, tmp_path: Path
) -> None:
    project = seed_project(client, tmp_path / "demo")
    session = seed_session(client, project)

    async def remove_model() -> None:
        async with session_factory() as db:
            await db.execute(
                update(Session).where(Session.id == session).values(llm_profile="gone")
            )
            await db.commit()

    run(client, remove_model)
    refused = client.post(f"/api/sessions/{session}/messages", json={"content": "hi"})
    assert refused.status_code == 409

    response = client.post(f"/api/sessions/{session}/model", json={"llm_profile": "fast"})
    assert response.status_code == 200
    assert model_of(client, session) == "fast"


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
def test_an_active_session_keeps_its_model(client: TestClient, tmp_path: Path, status: str) -> None:
    project = seed_project(client, tmp_path / "demo")
    session = seed_session(client, project, status)
    response = client.post(f"/api/sessions/{session}/model", json={"llm_profile": "spare"})
    assert response.status_code == 409
    assert "Остановите сессию" in response.json()["detail"]
    assert model_of(client, session) == "fast"
    assert change_events(client, session) == []


def test_an_unknown_model_or_session_is_refused(client: TestClient, tmp_path: Path) -> None:
    project = seed_project(client, tmp_path / "demo")
    session = seed_session(client, project)
    unknown = client.post(f"/api/sessions/{session}/model", json={"llm_profile": "absent"})
    assert unknown.status_code == 422
    assert model_of(client, session) == "fast"
    assert client.post(f"/api/sessions/{session}/model", json={}).status_code == 422
    missing = client.post("/api/sessions/absent/model", json={"llm_profile": "spare"})
    assert missing.status_code == 404


def test_an_external_agent_session_keeps_its_backend(client: TestClient, tmp_path: Path) -> None:
    project = seed_project(client, tmp_path / "demo")

    async def seed() -> str:
        async with session_factory() as db:
            row = Session(
                project_id=project,
                title="external",
                status="completed",
                llm_profile=None,
                engine_profile="cli",
                run_generation=0,
            )
            db.add(row)
            await db.commit()
            return row.id

    session = run(client, seed)
    response = client.post(f"/api/sessions/{session}/model", json={"llm_profile": "spare"})
    assert response.status_code in {403, 409}
    assert model_of(client, session) is None

    # The operator passes the route access check, so the handler must refuse by itself.
    async def change() -> int:
        async with session_factory() as db:
            with pytest.raises(HTTPException) as refused:
                await api.change_session_model(session, SessionModelChange(llm_profile="spare"), db)
            return refused.value.status_code

    assert run(client, change) == 409
    assert model_of(client, session) is None
