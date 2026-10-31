"""The status tool reports bounded facts without reading conversation content."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import (
    Approval,
    Message,
    OutboundAudit,
    Project,
    Session,
    ToolCall,
    User,
    Workspace,
    WorkspaceMembership,
)
from agent.tools.base import RiskLevel, ToolContext, ToolError
from agent.tools.registry import ToolRegistry
from tests.integration.test_run_finish_races import run_store
from tests.unit.test_engine import profile

__all__ = ["run_store"]


async def setup(tmp_path: Path, *, workspace: bool = False):
    await init_database(recover_running=False)
    async with session_factory() as db:
        user = User(display_name="reader")
        db.add(user)
        await db.flush()
        space = Workspace(name="team") if workspace else None
        if space:
            db.add(space)
            await db.flush()
            db.add(WorkspaceMembership(workspace_id=space.id, user_id=user.id, role="member"))
        project = Project(
            name="local", root_path=str(tmp_path), workspace_id=space.id if space else None
        )
        db.add(project)
        await db.flush()
        current = Session(project_id=project.id, title="current", created_by_user_id=user.id)
        target = Session(
            project_id=project.id,
            parent_id=None,
            title="private title",
            created_by_user_id=user.id,
            status="running",
            run_generation=2,
            budget_state={
                "run_generation": 2,
                "steps": 3,
                "started_at": (datetime.now(UTC) - timedelta(seconds=60)).isoformat(),
                "limits": {"max_steps": 9, "max_tokens": 1000, "max_cost": "0.5"},
            },
        )
        db.add_all([current, target])
        await db.commit()
        return current.id, target.id, user.id, project.id


async def status(current: str, target: str, tmp_path: Path, settings=None):
    return await ToolRegistry(settings=settings or Settings()).execute(
        "agent_status", "dev", ToolContext(current, tmp_path), {"session_id": target}
    )


def test_status_is_registered_read_only_with_strict_arguments():
    registry = ToolRegistry()
    for mode in ("dev", "pm", "ask", "analyst"):
        assert registry.get("agent_status", mode).risk_level == RiskLevel.read_only
    assert not registry.requires_confirmation("agent_status", "dev")
    for arguments in ({}, {"session_id": ""}, {"session_id": "s", "logs": True}):
        with pytest.raises(ToolError, match="Invalid arguments"):
            registry.validate("agent_status", "dev", arguments)


async def test_reports_budget_and_last_step_without_private_payloads(tmp_path):
    current, target, _, _ = await setup(tmp_path)
    secret = "PRIVATE-CONTENT-MUST-NOT-ESCAPE"
    async with session_factory() as db:
        db.add(Message(session_id=target, role="assistant", content=secret))
        db.add(
            ToolCall(
                id="work",
                session_id=target,
                name="read_file",
                risk_level="read_only",
                status="running",
                arguments={"path": secret},
                result={"text": secret},
            )
        )
        db.add(
            OutboundAudit(
                session_id=target,
                category="llm",
                destination=secret,
                operation="chat",
                detail=secret,
                status="completed",
                prompt_tokens=12,
                completion_tokens=8,
                cost_amount="0.02",
            )
        )
        await db.commit()
    result = await status(current, target, tmp_path)
    assert result["session_id"] == target
    assert result["status"] == "running" and result["waiting_for"] == "tool"
    assert result["latest_tool_call"]["name"] == "read_file"
    assert result["budget"]["steps"] == 3 and result["budget"]["max_steps"] == 9
    assert result["budget"]["known_tokens"] == 20
    assert result["budget"]["known_cost"] == "0.02"
    assert result["budget"]["has_unknown_cost"] is False
    assert result["latest_inference"]["status"] == "completed"
    assert result["observed_at"] and result["last_activity_at"]
    assert secret not in json.dumps(result)
    assert "private title" not in json.dumps(result)


async def test_current_approvals_are_counted_but_output_is_bounded(tmp_path):
    current, target, _, _ = await setup(tmp_path)
    async with session_factory() as db:
        row = await db.get(Session, target)
        row.status = "awaiting_confirmation"
        for index in range(15):
            call = ToolCall(
                id=f"approval-{index}",
                session_id=target,
                name="write_file",
                risk_level="local_write",
                status="awaiting_confirmation",
            )
            db.add(call)
            await db.flush()
            db.add(Approval(tool_call_id=call.id, generation=2 if index < 13 else 1))
        await db.commit()
    result = await status(current, target, tmp_path)
    assert result["waiting_for"] == "approval"
    assert result["approvals"]["pending_count"] == 13
    assert len(result["approvals"]["items"]) == 10
    assert result["approvals"]["truncated"] is True
    assert len(json.dumps(result)) < 6000


@pytest.mark.parametrize("state", ["running", "completed", "stopped", "error"])
async def test_silence_never_becomes_a_hang_diagnosis(tmp_path, state):
    current, target, _, _ = await setup(tmp_path)
    async with session_factory() as db:
        row = await db.get(Session, target)
        row.status = state
        row.updated_at = datetime.now(UTC) - timedelta(days=3)
        row.error = "private failure"
        row.budget_state = {"run_generation": 1, "steps": 99, "stop_reason": "cost_exhausted"}
        await db.commit()
    result = await status(current, target, tmp_path)
    assert result["status"] == state and result["waiting_for"] is None
    assert result["seconds_since_activity"] > 86400
    assert result["budget"]["steps"] == 0
    assert result["budget"]["started_at"] is None
    assert result["budget"]["stop_reason"] is None
    assert "hang" not in json.dumps(result) and "private failure" not in json.dumps(result)


async def test_waited_child_is_reported_from_running_tool_not_from_silence(tmp_path):
    current, target, _, _ = await setup(tmp_path)
    async with session_factory() as db:
        db.add(
            ToolCall(
                id="wait",
                session_id=target,
                name="run_agent",
                risk_level="local_write",
                status="running",
                arguments={"wait": True, "prompt": "private"},
            )
        )
        await db.commit()
    assert (await status(current, target, tmp_path))["waiting_for"] == "child"


async def test_parallel_completion_does_not_hide_a_running_tool(tmp_path):
    current, target, _, _ = await setup(tmp_path)
    now = datetime.now(UTC)
    async with session_factory() as db:
        db.add_all(
            [
                ToolCall(
                    id="active",
                    session_id=target,
                    name="search_files",
                    risk_level="read_only",
                    status="running",
                    created_at=now - timedelta(seconds=10),
                    updated_at=now - timedelta(seconds=10),
                ),
                ToolCall(
                    id="done",
                    session_id=target,
                    name="read_file",
                    risk_level="read_only",
                    status="completed",
                    created_at=now,
                    updated_at=now,
                ),
            ]
        )
        await db.commit()
    result = await status(current, target, tmp_path)
    assert result["latest_tool_call"]["id"] == "done"
    assert result["waiting_for"] == "tool"


async def test_old_running_records_do_not_describe_the_new_run(tmp_path):
    current, target, _, _ = await setup(tmp_path)
    old = datetime.now(UTC) - timedelta(days=1)
    async with session_factory() as db:
        db.add(
            ToolCall(
                id="old",
                session_id=target,
                name="run_agent",
                risk_level="local_write",
                status="running",
                created_at=old,
                updated_at=old,
            )
        )
        db.add(
            OutboundAudit(
                session_id=target,
                category="llm",
                destination="private",
                operation="chat",
                status="started",
                created_at=old,
            )
        )
        await db.commit()
    result = await status(current, target, tmp_path)
    assert result["latest_tool_call"]["id"] == "old"
    assert result["waiting_for"] is None


async def test_no_writes_and_budget_stop_reason_is_allowlisted(tmp_path):
    from sqlalchemy import event

    from agent.database import engine

    current, target, _, _ = await setup(tmp_path)
    async with session_factory() as db:
        row = await db.get(Session, target)
        row.budget_state = dict(row.budget_state, stop_reason="tokens_exhausted")
        row.stop_requested = True
        await db.commit()
    statements = []

    def record(_conn, _cursor, statement, *_args):
        statements.append(statement.lstrip().split()[0].upper())

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        result = await status(current, target, tmp_path)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)
    assert statements and set(statements) == {"SELECT"}
    assert result["stop_requested"] is True
    assert result["budget"]["stop_reason"] == "tokens_exhausted"
    async with session_factory() as db:
        row = await db.get(Session, target)
        row.budget_state = dict(row.budget_state, stop_reason="private arbitrary text")
        await db.commit()
    assert (await status(current, target, tmp_path))["budget"]["stop_reason"] is None


@pytest.mark.parametrize("foreign", ["missing", "project", "workspace"])
async def test_missing_and_foreign_targets_have_the_same_error(tmp_path, foreign):
    current, target, _, _ = await setup(tmp_path)
    async with session_factory() as db:
        if foreign == "missing":
            target = "not-found"
        else:
            space = Workspace(name="other")
            db.add(space)
            await db.flush()
            project = Project(
                name="foreign",
                root_path=str(tmp_path / "other"),
                workspace_id=space.id if foreign == "workspace" else None,
            )
            db.add(project)
            await db.flush()
            row = await db.get(Session, target)
            row.project_id = project.id
            await db.commit()
    with pytest.raises(ToolError, match="^Session not found$"):
        await status(current, target, tmp_path)


@pytest.mark.parametrize("denial", ["disabled", "revoked", "missing", "unbound"])
async def test_workspace_access_uses_current_identity_and_membership(tmp_path, denial):
    current, target, user, project = await setup(tmp_path, workspace=True)
    assert (await status(current, target, tmp_path))["session_id"] == target
    async with session_factory() as db:
        if denial == "disabled":
            (await db.get(User, user)).enabled = False
        elif denial == "unbound":
            (await db.get(Session, current)).created_by_user_id = None
        else:
            space = (await db.get(Project, project)).workspace_id
            membership = await db.get(WorkspaceMembership, (space, user))
            if denial == "revoked":
                membership.revoked_at = datetime.now(UTC)
            else:
                await db.delete(membership)
        await db.commit()
    with pytest.raises(ToolError, match="^Session not found$"):
        await status(current, target, tmp_path)


async def test_personal_engine_requires_creator_and_current_operator(tmp_path):
    current, target, user, _ = await setup(tmp_path)
    settings = Settings(engine_profiles={"personal": profile(operator_user_id=user)})
    async with session_factory() as db:
        row = await db.get(Session, target)
        row.llm_profile = None
        row.engine_profile = "personal"
        row.engine_state = {"operator_user_id": user, "secret": "private engine state"}
        await db.commit()
    result = await status(current, target, tmp_path, settings)
    assert result["session_id"] == target
    assert result["budget"] is None
    settings.engine_profiles = {"personal": profile(operator_user_id="someone-else")}
    with pytest.raises(ToolError, match="^Session not found$"):
        await status(current, target, tmp_path, settings)


@pytest.mark.parametrize("location", ["current", "target"])
async def test_another_users_engine_is_hidden_in_both_directions(tmp_path, location):
    current, target, user, _ = await setup(tmp_path)
    settings = Settings(engine_profiles={"personal": profile(operator_user_id=user)})
    async with session_factory() as db:
        other = User(display_name="other")
        db.add(other)
        await db.flush()
        row = await db.get(Session, current if location == "current" else target)
        row.llm_profile = None
        row.engine_profile = "personal"
        row.created_by_user_id = other.id if location == "target" else user
        if location == "current":
            settings.engine_profiles = {"personal": profile(operator_user_id=other.id)}
        await db.commit()
    with pytest.raises(ToolError, match="^Session not found$"):
        await status(current, target, tmp_path, settings)


async def test_legacy_personal_sessions_and_unknown_usage(tmp_path):
    current, target, _, _ = await setup(tmp_path)
    async with session_factory() as db:
        (await db.get(Session, current)).created_by_user_id = None
        db.add(
            OutboundAudit(
                session_id=target,
                category="llm",
                destination="private",
                operation="chat",
                status="started",
            )
        )
        await db.commit()
    result = await status(current, target, tmp_path)
    assert result["budget"]["has_unknown_tokens"] is True
    assert result["budget"]["has_unknown_cost"] is True
    assert result["latest_inference"]["status"] == "started"
    assert result["waiting_for"] == "model"


async def test_status_queries_on_each_database(run_store, tmp_path, monkeypatch):
    from agent.tools import agent_status

    monkeypatch.setattr(agent_status, "session_factory", run_store)
    async with run_store() as db:
        project = Project(name="database", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        current = Session(project_id=project.id, title="reader")
        target = Session(
            project_id=project.id,
            title="target",
            status="awaiting_confirmation",
            run_generation=1,
            budget_state={
                "run_generation": 1,
                "steps": 1,
                "started_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
            },
        )
        db.add_all([current, target])
        await db.flush()
        db.add(
            ToolCall(
                id="database-call",
                session_id=target.id,
                name="write_file",
                risk_level="local_write",
                status="awaiting_confirmation",
            )
        )
        await db.flush()
        db.add(Approval(tool_call_id="database-call", generation=1))
        db.add(
            OutboundAudit(
                session_id=target.id,
                category="llm",
                destination="private",
                operation="chat",
                status="started",
            )
        )
        await db.commit()
        current_id, target_id = current.id, target.id
    result = await status(current_id, target_id, tmp_path)
    assert result["approvals"]["pending_count"] == 1
    assert result["waiting_for"] == "approval"
    assert result["budget"]["has_unknown_cost"] is True
    async with run_store() as db:
        (await db.get(Session, target_id)).status = "running"
        await db.commit()
    assert (await status(current_id, target_id, tmp_path))["waiting_for"] == "model"
