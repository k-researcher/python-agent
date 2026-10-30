"""Personal engine selection, access and unavailable finish, independently of any CLI."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.api import settings, supervisor
from agent.config import Settings
from agent.database import session_factory
from agent.main import app
from agent.message_tree import append_message
from agent.models import Event, Message, Project, Session, SessionRunOutcome, User
from agent.runtime import AgentSupervisor
from agent.session_service import create_session_record
from tests.integration.test_session_identity import bound_login
from tests.unit.test_engine import profile


def setup(client, tmp_path, monkeypatch):
    actor = bound_login(client, bound=True)
    monkeypatch.setattr(
        settings, "engine_profiles", {"personal": profile(enabled=True, operator_user_id=actor)}
    )
    project = client.post(
        "/api/projects", json={"name": "engine", "root_path": str(tmp_path)}
    ).json()
    payload = {
        "project_id": project["id"],
        "prompt": "test",
        "engine_profile": "personal",
        "auto_start": False,
    }
    return actor, project, payload


async def snapshot(session_id):
    async with asyncio.timeout(3), session_factory() as db:
        session = await db.get(Session, session_id)
        return (
            {column.name: getattr(session, column.name) for column in Session.__table__.columns},
            [
                (m.id, m.content, m.skipped, m.parent_id)
                for m in await db.scalars(
                    select(Message).where(Message.session_id == session_id).order_by(Message.id)
                )
            ],
            list(await db.scalars(select(Session.id).order_by(Session.id))),
            [
                (e.id, e.type)
                for e in await db.scalars(
                    select(Event).where(Event.session_id == session_id).order_by(Event.id)
                )
            ],
        )


def test_create_read_and_profile_validation(tmp_path, monkeypatch):
    with TestClient(app) as client:
        actor, project, payload = setup(client, tmp_path, monkeypatch)
        created = client.post("/api/sessions", json=payload)
        assert created.status_code == 201, created.text
        result = created.json()
        assert result["engine_profile"] == "personal" and result["llm_profile"] is None
        assert client.get(f"/api/sessions/{result['id']}").json()["engine_profile"] == "personal"
        for changes in [
            {"llm_profile": "default"},
            {"engine_profile": "missing"},
            {"engine_profile": ""},
        ]:
            assert client.post("/api/sessions", json=payload | changes).status_code in {400, 422}
        monkeypatch.setattr(
            settings, "engine_profiles", {"personal": profile(operator_user_id=actor)}
        )
        assert client.post("/api/sessions", json=payload).status_code == 400


@pytest.mark.parametrize("operation", ["edit", "regenerate", "branch", "fork"])
def test_history_409_has_no_side_effects(tmp_path, monkeypatch, operation):
    dispatch = AsyncMock(side_effect=AssertionError("dispatch"))
    with TestClient(app) as client:
        _, _, payload = setup(client, tmp_path, monkeypatch)
        created = client.post("/api/sessions", json=payload)
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]

        async def prepare():
            async with asyncio.timeout(3), session_factory() as db:
                session = await db.get(Session, session_id)
                session.status = "completed"
                session.engine_state = {"thread_id": "native", "history_cursor": 7}
                answer = await append_message(db, session_id, role="assistant", content="answer")
                await db.commit()
                return answer.id

        answer_id = client.portal.call(prepare)
        before = client.portal.call(snapshot, session_id)
        user_id = before[1][1][0]
        monkeypatch.setattr(supervisor, "start", dispatch)
        if operation == "branch":
            url, body = f"/api/sessions/{session_id}/branch", {"leaf_id": user_id}
        else:
            target = answer_id if operation == "regenerate" else user_id
            url = f"/api/sessions/{session_id}/messages/{target}/{operation}"
            body = {"content": "replacement"} if operation == "edit" else {}
        response = client.post(url, json=body)
        assert response.status_code == 409, response.text
        assert "для внешнего агента пока недоступно" in response.json()["detail"]
        assert client.portal.call(snapshot, session_id) == before
        dispatch.assert_not_called()


@pytest.mark.parametrize("refusal", ["operator", "creator", "unbound", "workspace"])
@pytest.mark.parametrize("operation", ["messages", "start", "fork", "approval"])
def test_personal_access_before_changes(tmp_path, monkeypatch, refusal, operation):
    with TestClient(app) as client:
        actor, project, payload = setup(client, tmp_path, monkeypatch)
        created = client.post("/api/sessions", json=payload)
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]

        async def prepare():
            from agent.models import Approval, ToolCall

            async with asyncio.timeout(3), session_factory() as db:
                session = await db.get(Session, session_id)
                if refusal == "creator":
                    other = User(display_name="other")
                    db.add(other)
                    await db.flush()
                    session.created_by_user_id = other.id
                if refusal == "workspace":
                    from agent.models import Workspace

                    workspace = Workspace(name="team")
                    db.add(workspace)
                    await db.flush()
                    (await db.get(Project, project["id"])).workspace_id = workspace.id
                call = ToolCall(
                    id="engine-test-call",
                    session_id=session_id,
                    name="test",
                    arguments={},
                    risk_level="high",
                    status="pending",
                )
                db.add(call)
                await db.flush()
                approval = Approval(tool_call_id=call.id)
                db.add(approval)
                await db.commit()
                return approval.id

        approval_id = client.portal.call(prepare)
        if refusal == "operator":
            monkeypatch.setattr(
                settings,
                "engine_profiles",
                {"personal": profile(enabled=True, operator_user_id="other")},
            )
        if refusal == "unbound":
            monkeypatch.setattr(settings, "api_token", "test-token")
            client.headers["X-Agent-Token"] = "test-token"
        before = client.portal.call(snapshot, session_id)
        forbidden = AsyncMock(side_effect=AssertionError("dispatch"))
        monkeypatch.setattr(supervisor, "start", forbidden)
        monkeypatch.setattr(supervisor, "resolve_approval", forbidden)
        if operation == "fork":
            url, body = f"/api/sessions/{session_id}/messages/{before[1][1][0]}/fork", {}
        elif operation == "approval":
            url, body = f"/api/approvals/{approval_id}", {"decision": "approve"}
        else:
            url, body = (
                f"/api/sessions/{session_id}/{operation}",
                {"content": "next"} if operation == "messages" else {},
            )
        response = client.post(url, json=body)
        assert response.status_code == 403, response.text
        assert client.portal.call(snapshot, session_id) == before
        forbidden.assert_not_called()


async def test_service_xor_engine_skips_model_registry_and_model_default_stays(
    tmp_path, monkeypatch
):
    async with asyncio.timeout(4):
        (tmp_path / "dev.md").write_text("system")
        config = Settings(
            models_file="",
            prompts_dir=tmp_path,
            engine_profiles={"personal": profile(enabled=True)},
        )
        async with session_factory() as db:
            user = User(id="operator", display_name="operator")
            project = Project(name="service", root_path=str(tmp_path))
            db.add_all([user, project])
            await db.flush()
            ordinary = await create_session_record(db, config, project, "model", "dev")
            assert ordinary.llm_profile == "default" and ordinary.engine_profile is None
            forbidden = AsyncMock(side_effect=AssertionError("model registry"))
            monkeypatch.setattr("agent.session_service.effective_registry", forbidden)
            with pytest.raises(ValueError, match="backend|llm_profile"):
                await create_session_record(
                    db,
                    config,
                    project,
                    "bad",
                    "dev",
                    llm_profile="default",
                    engine_profile="personal",
                    actor_user_id="operator",
                )
            engine = await create_session_record(
                db,
                config,
                project,
                "engine",
                "dev",
                engine_profile="personal",
                actor_user_id="operator",
                configuration={"reasoning_effort": "engine-specific"},
            )
            assert engine.llm_profile is None
            forbidden.assert_not_called()


@pytest.mark.parametrize("availability", ["enabled", "disabled", "removed"])
async def test_message_finishes_unavailable_without_model_or_process(
    tmp_path, monkeypatch, availability
):
    async with asyncio.timeout(5):
        from agent.engine_adapters import EngineRegistry

        (tmp_path / "dev.md").write_text("system")
        config = Settings(
            models_file="",
            prompts_dir=tmp_path,
            engine_profiles={"personal": profile(enabled=True)},
        )
        async with session_factory() as db:
            db.add(User(id="operator", display_name="operator"))
            project = Project(name="finish", root_path=str(tmp_path))
            db.add(project)
            await db.flush()
            session = await create_session_record(
                db,
                config,
                project,
                "engine",
                "dev",
                engine_profile="personal",
                actor_user_id="operator",
            )
            session_id = session.id
        if availability == "disabled":
            config.engine_profiles["personal"] = profile()
        elif availability == "removed":
            config.engine_profiles.clear()
        runner = AgentSupervisor(config)
        forbidden = Mock(
            side_effect=AssertionError("engine cannot call models, tools or subprocess")
        )
        monkeypatch.setattr(runner, "_llm_for", forbidden)
        monkeypatch.setattr(runner, "_models", forbidden)
        monkeypatch.setattr("agent.runtime.source_registry", forbidden)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
        # Model a genuinely uninstalled adapter explicitly: both optional
        # adapters may now be present in the source tree. Keep all no-call and
        # durable-unavailable assertions independent of installed modules.
        monkeypatch.setattr("agent.engine_adapters.importlib.util.find_spec", lambda _: None)
        monkeypatch.setattr(EngineRegistry, "get", forbidden)
        await runner.run_job(session_id)
        async with session_factory() as db:
            record = await db.get(Session, session_id)
            assert record.status == "error"
            assert "engine недоступен" in record.error
            outcome = await db.get(SessionRunOutcome, (session_id, record.run_generation))
            assert outcome is not None and outcome.status == "error"
            events = list(
                await db.scalars(select(Event.type).where(Event.session_id == session_id))
            )
            assert "session.error" in events
        forbidden.assert_not_called()
        await runner.shutdown()


@pytest.mark.parametrize("refusal", ["operator", "unbound", "workspace"])
def test_create_refuses_nonpersonal_actor_or_project(tmp_path, monkeypatch, refusal):
    with TestClient(app) as client:
        _, project, payload = setup(client, tmp_path, monkeypatch)
        if refusal == "operator":
            monkeypatch.setattr(settings, "engine_profiles", {"personal": profile(enabled=True)})
        elif refusal == "unbound":
            monkeypatch.setattr(settings, "api_token", "test-token")
            client.headers["X-Agent-Token"] = "test-token"
        else:

            async def bind_workspace():
                from agent.models import Workspace

                async with asyncio.timeout(3), session_factory() as db:
                    workspace = Workspace(name="team")
                    db.add(workspace)
                    await db.flush()
                    (await db.get(Project, project["id"])).workspace_id = workspace.id
                    await db.commit()

            client.portal.call(bind_workspace)
        assert client.post("/api/sessions", json=payload).status_code == 403


@pytest.mark.parametrize("availability", ["enabled", "disabled", "removed"])
def test_http_message_finishes_without_model_fallback(tmp_path, monkeypatch, availability):
    with TestClient(app) as client:
        actor, _, payload = setup(client, tmp_path, monkeypatch)
        response = client.post("/api/sessions", json=payload)
        assert response.status_code == 201
        session_id = response.json()["id"]
        if availability == "disabled":
            monkeypatch.setattr(
                settings, "engine_profiles", {"personal": profile(operator_user_id=actor)}
            )
        elif availability == "removed":
            monkeypatch.setattr(settings, "engine_profiles", {})
        forbidden = Mock(side_effect=AssertionError("model or tool path"))
        monkeypatch.setattr(supervisor, "_llm_for", forbidden)
        monkeypatch.setattr(supervisor, "_models", forbidden)
        monkeypatch.setattr("agent.api.effective_registry", forbidden)
        monkeypatch.setattr("agent.runtime.source_registry", forbidden)
        sent = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "next", "reasoning_effort": "native"},
        )
        assert sent.status_code == 201, sent.text

        async def settle():
            async with asyncio.timeout(4):
                await supervisor._tasks[session_id]

        client.portal.call(settle)
        read = client.get(f"/api/sessions/{session_id}").json()
        assert read["status"] == "error" and "engine недоступен" in read["error"]
        forbidden.assert_not_called()


async def test_direct_start_and_approval_check_actor_before_changes(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        from agent.models import Approval, ToolCall

        (tmp_path / "dev.md").write_text("system")
        config = Settings(
            models_file="",
            prompts_dir=tmp_path,
            engine_profiles={"personal": profile(enabled=True)},
        )
        async with session_factory() as db:
            db.add(User(id="operator", display_name="operator"))
            project = Project(name="direct", root_path=str(tmp_path))
            db.add(project)
            await db.flush()
            session = await create_session_record(
                db,
                config,
                project,
                "test",
                "dev",
                engine_profile="personal",
                actor_user_id="operator",
            )
            call = ToolCall(
                id="native-call",
                session_id=session.id,
                name="test",
                arguments={},
                risk_level="high",
                status="pending",
            )
            db.add(call)
            await db.flush()
            approval = Approval(tool_call_id=call.id)
            db.add(approval)
            await db.commit()
            session_id, approval_id = session.id, approval.id
        runner = AgentSupervisor(config)
        before = await snapshot(session_id)
        with pytest.raises(PermissionError):
            await runner.start(session_id)
        with pytest.raises(PermissionError):
            await runner.resolve_approval(approval_id, "approve", None)
        assert await snapshot(session_id) == before
        assert runner._tasks == {}
        async with session_factory() as db:
            assert (await db.get(Approval, approval_id)).status == "pending"
        await runner.shutdown()
