"""Engine snapshots and live delivery are personal even on a shared browser API."""

import asyncio

from fastapi.testclient import TestClient

from agent.api import settings
from agent.database import session_factory
from agent.events import broker, emit_event
from agent.main import app
from tests.integration.helpers import ORIGIN
from tests.integration.test_engine_sessions import setup
from tests.integration.test_session_identity import bound_login
from tests.unit.test_engine import profile


def test_config_context_rest_and_ws_are_actor_filtered(tmp_path, monkeypatch):
    with TestClient(app) as client:
        actor, _, payload = setup(client, tmp_path, monkeypatch)
        monkeypatch.setattr(
            settings,
            "engine_profiles",
            {
                "personal": profile(enabled=True, operator_user_id=actor),
                "disabled": profile(operator_user_id=actor),
                "foreign": profile(enabled=True, operator_user_id="other"),
            },
        )
        sid = client.post("/api/sessions", json=payload).json()["id"]
        config = client.get("/api/config/public").json()
        assert [p["id"] for p in config["engine_profiles"]] == ["personal"]
        assert set(config["engine_profiles"][0]) == {"id", "status"}
        context = client.get(f"/api/sessions/{sid}/context").json()
        assert context["status"] == "unknown" and context["context_window"] is None

        async def broadcast():
            async with asyncio.timeout(3), session_factory() as db:
                await emit_event(db, "engine.usage", {"input_tokens": 7}, sid)
            await broker.broadcast({"type": "barrier", "payload": {}})

        with client.websocket_connect("/ws", headers={"Origin": ORIGIN}) as ws:
            assert ws.receive_json()["type"] == "connected"
            client.portal.call(broadcast)
            assert ws.receive_json()["type"] == "engine.usage"
            assert ws.receive_json()["type"] == "barrier"
        bound_login(client, bound=True)
        assert client.get("/api/config/public").json()["engine_profiles"] == []
        assert all(s["id"] != sid for s in client.get("/api/sessions").json())
        assert all(e["session_id"] != sid for e in client.get("/api/events").json())
        for path in [
            "",
            "/messages",
            "/approvals",
            "/children",
            "/context",
            "/outbound-audit",
            "/snapshot/session",
            "/snapshot/messages",
            "/snapshot/approvals",
            "/snapshot/context",
        ]:
            assert client.get(f"/api/sessions/{sid}{path}").status_code == 403
        assert client.post(f"/api/sessions/{sid}/stop").status_code == 403
        with client.websocket_connect("/ws", headers={"Origin": ORIGIN}) as ws:
            assert ws.receive_json()["type"] == "connected"
            client.portal.call(broadcast)
            assert ws.receive_json()["type"] == "barrier"


def test_native_approval_api_requires_identity_and_returns_scope(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from agent.models import Approval, Session, ToolCall

    with TestClient(app) as client:
        actor, _, payload = setup(client, tmp_path, monkeypatch)
        sid = client.post("/api/sessions", json=payload).json()["id"]

        async def pending():
            async with asyncio.timeout(3), session_factory() as db:
                session = await db.get(Session, sid)
                session.status = "awaiting_confirmation"
                session.run_generation = 1
                session.owner_token = "worker"
                session.lease_until = datetime.now(UTC) + timedelta(seconds=60)
                call = ToolCall(
                    id="native-call",
                    session_id=sid,
                    name="write",
                    arguments={"path": "a"},
                    risk_level="observed",
                    status="awaiting_confirmation",
                    executor="engine",
                    native_request_id="native",
                    native_item_id="item",
                )
                db.add(call)
                await db.flush()
                approval = Approval(
                    tool_call_id=call.id,
                    generation=1,
                    executor="engine",
                    scope="turn",
                    expires_at=datetime.now(UTC) + timedelta(seconds=60),
                )
                db.add(approval)
                await db.commit()
                return approval.id

        aid = client.portal.call(pending)
        listed = client.get(f"/api/sessions/{sid}/approvals").json()
        assert {
            key: listed[0][key] for key in ["generation", "executor", "scope", "request_id"]
        } == {"generation": 1, "executor": "engine", "scope": "turn", "request_id": "native"}
        for body in [
            {"decision": "approve"},
            {"decision": "approve", "generation": 0, "executor": "engine"},
            {"decision": "approve", "generation": 1, "executor": "python"},
        ]:
            assert client.post(f"/api/approvals/{aid}", json=body).status_code == 409
        body = {"decision": "reject", "generation": 1, "executor": "engine"}
        operator_cookies = dict(client.cookies)
        operator_csrf = client.headers["X-Agent-CSRF"]
        bound_login(client, bound=True)
        assert client.post(f"/api/approvals/{aid}", json=body).status_code == 403
        client.cookies.update(operator_cookies)
        client.headers["X-Agent-CSRF"] = operator_csrf
        assert client.post(f"/api/approvals/{aid}", json=body).status_code == 200
        assert client.post(f"/api/approvals/{aid}", json=body).status_code == 409


def test_ws_hides_deleted_engine_and_queued_payloads(tmp_path, monkeypatch):
    from agent.models import Session

    with TestClient(app) as client:
        _, _, payload = setup(client, tmp_path, monkeypatch)
        sid = client.post("/api/sessions", json=payload).json()["id"]
        operator_cookies = dict(client.cookies)
        operator_csrf = client.headers["X-Agent-CSRF"]

        async def inactive():
            async with asyncio.timeout(3), session_factory() as db:
                session = await db.get(Session, sid)
                session.status = "completed"
                await db.commit()

        client.portal.call(inactive)
        bound_login(client, bound=True)
        with client.websocket_connect("/ws", headers={"Origin": ORIGIN}) as ws:
            assert ws.receive_json()["type"] == "connected"
            client.cookies.update(operator_cookies)
            client.headers["X-Agent-CSRF"] = operator_csrf
            assert client.delete(f"/api/sessions/{sid}").status_code == 204

            async def queued():
                async with asyncio.timeout(3):
                    await broker.broadcast(
                        {"type": "engine.usage", "session_id": sid, "payload": {"input_tokens": 9}}
                    )
                    await broker.broadcast({"type": "barrier", "payload": {}})

            client.portal.call(queued)
            assert ws.receive_json()["type"] == "barrier"


def test_child_lists_hide_engine_data_when_parent_uses_model(tmp_path, monkeypatch):
    from agent.models import Session

    with TestClient(app) as client:
        _, project, payload = setup(client, tmp_path, monkeypatch)
        sid = client.post("/api/sessions", json=payload).json()["id"]
        parent = client.post(
            "/api/sessions",
            json={"project_id": project["id"], "prompt": "parent", "auto_start": False},
        ).json()["id"]

        async def attach():
            async with asyncio.timeout(3), session_factory() as db:
                session = await db.get(Session, sid)
                session.parent_id = parent
                await db.commit()

        client.portal.call(attach)
        assert [row["id"] for row in client.get(f"/api/sessions/{parent}/children").json()] == [sid]
        bound_login(client, bound=True)
        assert client.get(f"/api/sessions/{parent}/children").json() == []
        assert client.get(f"/api/sessions/{parent}/snapshot/children").json()["data"] == []
