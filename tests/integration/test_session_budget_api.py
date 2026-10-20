"""HTTP surface of session budgets: validation, snapshot, and restart blocks."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from agent.api import settings, supervisor
from agent.database import session_factory
from agent.models import Session
from tests.integration.helpers import authed_client


def _create_project(client: TestClient, tmp_path: Path, name: str = "budget project") -> str:
    response = client.post("/api/projects", json={"name": name, "root_path": str(tmp_path)})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _create_session(
    client: TestClient, project_id: str, *, budget: dict[str, object] | None = None
) -> tuple[int, dict[str, object]]:
    payload: dict[str, object] = {
        "project_id": project_id,
        "prompt": "budget test",
        "mode": "dev",
        "auto_start": False,
    }
    if budget is not None:
        payload["budget"] = budget
    response = client.post("/api/sessions", json=payload)
    return response.status_code, response.json()


def _set_budget_blocked(session_id: str, *, reason: str) -> None:
    async def apply() -> None:
        async with session_factory() as db:
            row = await db.get(Session, session_id)
            assert row is not None
            state = dict(row.budget_state or {})
            state["run_generation"] = row.run_generation or 0
            state["steps"] = 999
            state["stop_reason"] = reason
            row.budget_state = state
            await db.commit()

    asyncio.run(apply())


def _set_cost_spent(session_id: str, *, amount: str) -> None:
    async def apply() -> None:
        async with session_factory() as db:
            from agent.models import OutboundAudit

            db.add(
                OutboundAudit(
                    session_id=session_id,
                    category="llm",
                    destination="https://llm.example",
                    operation="chat:test",
                    status="completed",
                    prompt_tokens=10,
                    completion_tokens=20,
                    cost_amount=amount,
                )
            )
            await db.commit()

    asyncio.run(apply())


def _set_cost_limit(session_id: str, amount: str) -> None:
    async def apply() -> None:
        async with session_factory() as db:
            row = await db.get(Session, session_id)
            assert row is not None
            state = dict(row.budget_state or {})
            state["limits"] = {**state.get("limits", {}), "max_cost": amount}
            row.budget_state = state
            await db.commit()

    asyncio.run(apply())


def _advance_run_generation(session_id: str) -> None:
    async def apply() -> None:
        async with session_factory() as db:
            row = await db.get(Session, session_id)
            assert row is not None
            row.run_generation = (row.run_generation or 0) + 1
            await db.commit()

    asyncio.run(apply())


def test_stop_reason_only_for_current_generation(tmp_path: Path) -> None:
    with authed_client() as client:
        project_id = _create_project(client, tmp_path)
        status, body = _create_session(client, project_id, budget={"max_steps": 3})
        assert status == 201
        session_id = body["id"]
        _set_budget_blocked(session_id, reason="steps_exhausted")

        read = client.get(f"/api/sessions/{session_id}").json()
        assert read["budget"]["stop_reason"] == "steps_exhausted"

        # A new generation hides the stale stop reason of the previous run.
        _advance_run_generation(session_id)
        read = client.get(f"/api/sessions/{session_id}").json()
        assert read["budget"]["stop_reason"] is None
        assert read["budget"]["steps"] == 0


def test_create_session_writes_budget_snapshot_and_reads_it_back(tmp_path: Path) -> None:
    with authed_client() as client:
        project_id = _create_project(client, tmp_path)
        status, body = _create_session(
            client,
            project_id,
            budget={
                "max_steps": 7,
                "max_tokens": 5000,
                "max_cost": "1.25",
                "deadline_seconds": 60.0,
            },
        )
        assert status == 201, body
        session_id = body["id"]
        assert isinstance(body["budget"], dict)
        budget = body["budget"]
        assert budget["max_steps"] == 7
        assert budget["max_tokens"] == 5000
        assert budget["max_cost"] == "1.25"
        assert budget["deadline_seconds"] == 60.0
        assert budget["steps"] == 0
        assert budget["known_tokens"] == 0
        assert budget["known_cost"] == "0"
        assert budget["has_unknown_tokens"] is False
        assert budget["has_unknown_cost"] is False
        assert budget["started_at"] is not None
        assert budget["stop_reason"] is None

        read = client.get(f"/api/sessions/{session_id}")
        assert read.status_code == 200
        read_budget = read.json()["budget"]
        assert read_budget["started_at"] is not None
        for key, expected in budget.items():
            if key == "started_at":
                continue
            assert read_budget[key] == expected, (key, read_budget[key], expected)
        assert read.json()["configuration"]["budget"]["max_tokens"] == 5000


def test_create_session_without_budget_uses_server_defaults(tmp_path: Path) -> None:
    original_tokens = settings.max_session_tokens
    original_steps = settings.max_inference_steps
    try:
        settings.max_session_tokens = 4321
        settings.max_inference_steps = 55
        with authed_client() as client:
            project_id = _create_project(client, tmp_path)
            status, body = _create_session(client, project_id)
            assert status == 201, body
            budget = body["budget"]
            assert budget["max_steps"] == 55
            assert budget["max_tokens"] == 4321
            assert budget["max_cost"] is None
            assert budget["deadline_seconds"] is None
    finally:
        settings.max_session_tokens = original_tokens
        settings.max_inference_steps = original_steps


def test_create_session_rejects_budget_above_server_ceiling(tmp_path: Path) -> None:
    original = settings.max_session_tokens, settings.max_session_cost
    try:
        settings.max_session_tokens = 1000
        settings.max_session_cost = Decimal("1")
        with authed_client() as client:
            project_id = _create_project(client, tmp_path)
            for bad in ({"max_tokens": 5000}, {"max_cost": "1.01"}):
                status, body = _create_session(client, project_id, budget=bad)
                assert status == 422, (bad, body)
    finally:
        settings.max_session_tokens, settings.max_session_cost = original


def test_create_session_rejects_non_positive_budget_values(tmp_path: Path) -> None:
    with authed_client() as client:
        project_id = _create_project(client, tmp_path)
        for bad in (
            {"max_steps": 0},
            {"max_tokens": -1},
            {"max_cost": "0"},
            {"max_cost": "not-a-number"},
            {"deadline_seconds": 0},
        ):
            status, body = _create_session(client, project_id, budget=bad)
            assert status == 422, (bad, body)


def test_list_sessions_and_children_include_budget(tmp_path: Path) -> None:
    with authed_client() as client:
        project_id = _create_project(client, tmp_path)
        status, body = _create_session(client, project_id, budget={"max_steps": 3})
        assert status == 201
        listing = client.get("/api/sessions").json()
        assert any(item["id"] == body["id"] for item in listing)
        match = next(item for item in listing if item["id"] == body["id"])
        assert match["budget"]["max_steps"] == 3


class FakeApproval:
    """LLM stub that replies with a write_file tool call, creating an approval."""

    endpoint = "https://fake.example/v1/chat/completions"

    async def chat(
        self,
        _messages: list[dict[str, object]],
        _tools: list[dict[str, object]],
        **_options: object,
    ) -> object:
        from agent.llm import LLMResponse

        return LLMResponse(
            content=None,
            tool_calls=[
                {
                    "id": "blocked-call",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"x.txt","content":"x"}',
                    },
                }
            ],
            finish_reason="tool_calls",
            prompt_tokens=10,
            completion_tokens=5,
        )


def test_new_run_resets_steps_when_steps_were_exhausted(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = FakeApproval()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project_id = _create_project(client, tmp_path)
            status, body = _create_session(client, project_id, budget={"max_steps": 3})
            assert status == 201
            session_id = body["id"]
            _set_budget_blocked(session_id, reason="steps_exhausted")

            # Sending a message starts a new run: the exhausted step counter belongs
            # to the previous run, so the fresh run is allowed.
            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "continue"},
            )
            assert response.status_code == 201, response.text
            read = client.get(f"/api/sessions/{session_id}").json()
            assert read["budget"]["stop_reason"] is None
    finally:
        supervisor.llm = original_llm


def test_money_exhausted_blocks_new_run_with_409(tmp_path: Path) -> None:
    with authed_client() as client:
        project_id = _create_project(client, tmp_path)
        status, body = _create_session(client, project_id, budget={"max_cost": "0.01"})
        assert status == 201
        session_id = body["id"]
        _set_cost_spent(session_id, amount="0.05")

        # Money is session-wide: a new run still counts the spent cost.
        response = client.post(f"/api/sessions/{session_id}/start", json={})
        assert response.status_code == 409
        assert "cost_exhausted" in response.json()["detail"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "continue"},
        )
        assert response.status_code == 409
        assert "cost_exhausted" in response.json()["detail"]


def test_approval_on_cost_exhausted_session_returns_409(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = FakeApproval()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project_id = _create_project(client, tmp_path)
            status, body = _create_session(client, project_id, budget={"max_steps": 10})
            assert status == 201
            session_id = body["id"]
            response = client.post(f"/api/sessions/{session_id}/start", json={})
            # The fake LLM does not drive the await loop synchronously; stop it cleanly.
            import time

            deadline = time.monotonic() + 2
            approvals = []
            while time.monotonic() < deadline:
                approvals = client.get(f"/api/sessions/{session_id}/approvals").json()
                if approvals:
                    break
                time.sleep(0.02)
            assert approvals, "no approval was created"
            _set_cost_limit(session_id, "0.5")
            _set_cost_spent(session_id, amount="0.5")
            response = client.post(
                f"/api/approvals/{approvals[0]['id']}",
                json={"decision": "approve"},
            )
            assert response.status_code == 409
            assert "cost_exhausted" in response.json()["detail"]
    finally:
        supervisor.llm = original_llm
