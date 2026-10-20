from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import agent.runtime as runtime
from agent.config import Settings
from agent.database import init_database, session_factory
from agent.llm import LLMResponse
from agent.models import (
    Approval,
    Message,
    Project,
    Session,
    ToolCall,
    User,
    Workspace,
    WorkspaceMembership,
)
from agent.runtime import AgentSupervisor
from agent.session_fence import CURRENT_FENCE, FenceLostError, claim, stop_generation
from agent.source_binding import mcp_tool_policy
from agent.tool_sources import ToolSource
from agent.tools.base import RiskLevel, Tool, ToolContext
from agent.tools.mcp_tools import McpBackedTool
from agent.tools.registry import ToolRegistry


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"content": [{"type": "text", "text": "done"}]}


class CountingBuiltin(Tool):
    description = "Test builtin"
    risk_level = RiskLevel.read_only
    input_schema = {"type": "object"}
    parallel_safe = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"done": True}


@dataclass
class RuntimeCase:
    session_id: str
    user_id: str
    source: ToolSource
    client: CountingClient
    tool: McpBackedTool
    supervisor: AgentSupervisor


def bound_tool(client: CountingClient, source: ToolSource) -> McpBackedTool:
    return McpBackedTool(
        client, mcp_tool_policy(source), "read", "Read", {"type": "object"}, source=source
    )


@pytest.fixture
async def runtime_case(tmp_path: Path) -> AsyncIterator[RuntimeCase]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        user = User(display_name="Runtime actor")
        workspace = Workspace(name="Runtime workspace")
        db.add_all([user, workspace])
        await db.flush()
        project = Project(
            name="Runtime project", root_path=str(tmp_path), workspace_id=workspace.id
        )
        db.add_all(
            [
                project,
                WorkspaceMembership(user_id=user.id, workspace_id=workspace.id, role="member"),
            ]
        )
        await db.flush()
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Source run",
            status="running",
            configuration={},
        )
        db.add(session)
        await db.commit()
        source = ToolSource(
            source_id="bound",
            workspace_id=workspace.id,
            project_ids=frozenset({project.id}),
            command=("server",),
            enabled=True,
        )
        client = CountingClient()
        tool = bound_tool(client, source)
        supervisor = AgentSupervisor(Settings(), ToolRegistry(tools=(tool,)))
        yield RuntimeCase(session.id, user.id, source, client, tool, supervisor)


async def load_session(db: AsyncSession, session_id: str) -> Session:
    return (
        await db.scalars(
            select(Session)
            .where(Session.id == session_id)
            .options(
                selectinload(Session.project),
                selectinload(Session.tool_calls).selectinload(ToolCall.approval),
            )
        )
    ).one()


async def add_call(
    case: RuntimeCase,
    *,
    revision: str | None,
    name: str | None = None,
    approval: str | None = "approved",
    status: str = "pending",
) -> str:
    async with session_factory() as db:
        call = ToolCall(
            id=uuid.uuid4().hex,
            session_id=case.session_id,
            name=name or case.tool.name,
            arguments={},
            risk_level="network_access",
            source_revision=revision,
            status=status,
        )
        db.add(call)
        await db.flush()
        if approval is not None:
            db.add(Approval(tool_call_id=call.id, status=approval))
        await db.commit()
        return call.id


async def assert_call_error(case: RuntimeCase, call_id: str) -> None:
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.status == "error"
        assert call.result is not None and call.result["success"] is False
        messages = list(await db.scalars(select(Message).where(Message.tool_call_id == call_id)))
        assert len(messages) == 1 and messages[0].role == "tool"
    assert case.client.calls == 0


@pytest.mark.parametrize(
    "change",
    ["null", "schema", "config", "membership", "builtin", "builtin-null", "unbound", "missing"],
)
async def test_approved_source_call_fails_closed(runtime_case: RuntimeCase, change: str) -> None:
    case = runtime_case
    revision = None if change in {"null", "builtin-null"} else case.tool.approval_revision()
    call_id = await add_call(case, revision=revision)
    if change == "schema":
        case.tool.input_schema["required"] = ["changed"]
    elif change == "config":
        replacement = bound_tool(
            case.client, case.source.model_copy(update={"command": ("other-server",)})
        )
        case.supervisor.registry = ToolRegistry(tools=(replacement,))
    elif change == "membership":
        async with session_factory() as other:
            await other.execute(
                update(WorkspaceMembership)
                .where(WorkspaceMembership.user_id == case.user_id)
                .values(role="viewer")
            )
            await other.commit()
    elif change in {"builtin", "builtin-null"}:
        builtin = CountingBuiltin(case.tool.name)
        case.supervisor.registry = ToolRegistry(tools=(builtin,))
    elif change == "unbound":
        unbound = McpBackedTool(
            case.client, mcp_tool_policy(case.source), "read", "Read", {"type": "object"}
        )
        case.supervisor.registry = ToolRegistry(tools=(unbound,))
    elif change == "missing":
        case.supervisor.registry = ToolRegistry(tools=())
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert await case.supervisor._resume_tool_calls(db, session) == "processed"
    await assert_call_error(case, call_id)
    if change in {"builtin", "builtin-null"}:
        assert builtin.calls == 0


async def test_valid_approved_source_executes_once(runtime_case: RuntimeCase) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert await case.supervisor._resume_tool_calls(db, session) == "processed"
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.status == "completed"
        session = await load_session(db, case.session_id)
        assert await case.supervisor._resume_tool_calls(db, session) == "ready"
    assert case.client.calls == 1


async def test_revocation_during_approval_wait_is_rechecked(runtime_case: RuntimeCase) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision(), approval="pending")
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert await case.supervisor._resume_tool_calls(db, session) == "waiting"
    assert case.client.calls == 0
    async with session_factory() as db:
        await db.execute(update(User).where(User.id == case.user_id).values(enabled=False))
        await db.execute(
            update(Approval).where(Approval.tool_call_id == call_id).values(status="approved")
        )
        await db.commit()
        session = await load_session(db, case.session_id)
        assert await case.supervisor._resume_tool_calls(db, session) == "processed"
    await assert_call_error(case, call_id)


async def test_source_revision_cannot_bypass_guard_through_parallel_batch(
    runtime_case: RuntimeCase,
) -> None:
    case = runtime_case
    replacement = CountingBuiltin(case.tool.name)
    sibling = CountingBuiltin("sibling")
    case.supervisor.registry = ToolRegistry(tools=(replacement, sibling))
    blocked_id = await add_call(case, revision=case.tool.approval_revision(), approval=None)
    await add_call(case, revision=None, name=sibling.name, approval=None)
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        calls = sorted(session.tool_calls, key=lambda call: call.id != blocked_id)
        assert not case.supervisor._may_run_in_parallel(session, calls[0])
        assert await case.supervisor._execute_parallel(db, session, calls)
    assert replacement.calls == 0 and sibling.calls == 1
    await assert_call_error(case, blocked_id)


@pytest.mark.parametrize("status", ["running", "stopped"])
async def test_interrupted_or_stopped_source_is_not_executed(
    runtime_case: RuntimeCase, status: str
) -> None:
    case = runtime_case
    call_id = await add_call(
        case,
        revision=case.tool.approval_revision(),
        status="running" if status == "running" else "pending",
    )
    async with session_factory() as db:
        if status == "stopped":
            await db.execute(
                update(Session)
                .where(Session.id == case.session_id)
                .values(stop_requested=True, status="stopped")
            )
            await db.commit()
        session = await load_session(db, case.session_id)
        result = await case.supervisor._resume_tool_calls(db, session)
        assert result == ("processed" if status == "running" else "stopped")
    assert case.client.calls == 0
    if status == "running":
        await assert_call_error(case, call_id)


class SourceCallLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any
    ) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            return LLMResponse("Done", [], "stop", 1, 1)
        return LLMResponse(
            None,
            [
                {
                    "id": "source-call",
                    "type": "function",
                    "function": {"name": self.name, "arguments": "{}"},
                }
            ],
            "tool_calls",
            1,
            1,
        )


async def test_runtime_persists_source_revision_before_approval(runtime_case: RuntimeCase) -> None:
    case = runtime_case
    case.supervisor.llm = SourceCallLLM(case.tool.name)
    async with session_factory() as db:
        db.add(Message(session_id=case.session_id, role="user", content="Read"))
        await db.commit()
    await case.supervisor._run(case.session_id)
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert session.status == "awaiting_confirmation"
        assert len(session.tool_calls) == 1
        call = session.tool_calls[0]
        assert call.source_revision == case.tool.approval_revision()
        assert call.approval is not None and call.approval.status == "pending"
        assert call.status == "awaiting_confirmation"
    assert case.client.calls == 0


@pytest.mark.parametrize("change", ["none", "schema", "membership"])
async def test_actual_approval_resume_keeps_original_revision(
    runtime_case: RuntimeCase, change: str
) -> None:
    case = runtime_case
    case.supervisor.llm = SourceCallLLM(case.tool.name)
    case.supervisor.start = AsyncMock(return_value=True)
    async with session_factory() as db:
        db.add(Message(session_id=case.session_id, role="user", content="Read"))
        await db.commit()
    await case.supervisor._run(case.session_id)
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        call = session.tool_calls[0]
        assert call.approval is not None
        approval_id = call.approval.id
        call_id = call.id
        approved_revision = call.source_revision
    if change == "schema":
        case.tool.input_schema["required"] = ["changed"]
    elif change == "membership":
        async with session_factory() as db:
            await db.execute(
                update(WorkspaceMembership)
                .where(WorkspaceMembership.user_id == case.user_id)
                .values(role="viewer")
            )
            await db.commit()
    assert await case.supervisor.resolve_approval(approval_id, "approve", None) == case.session_id
    case.supervisor.start.assert_awaited_once_with(case.session_id, new_run=False)
    await case.supervisor._run(case.session_id)
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.source_revision == approved_revision
        assert call.status == ("completed" if change == "none" else "error")
        approval = await db.get(Approval, approval_id)
        assert approval is not None and approval.status == "approved"
        assert approval.resolved_at is not None
    assert case.client.calls == (1 if change == "none" else 0)


async def test_source_cannot_be_executed_by_direct_parallel_batch(
    runtime_case: RuntimeCase,
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision(), approval=None)
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert await case.supervisor._execute_parallel(db, session, session.tool_calls)
    await assert_call_error(case, call_id)


@pytest.mark.parametrize("owned", [False, True])
async def test_stop_during_source_guard_prevents_new_execution(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch, owned: bool
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    original_guard = runtime.validate_source_execution

    async def stop_after_guard(
        db: AsyncSession, session_id: str, tool: Tool, approved_revision: str | None
    ) -> None:
        await original_guard(db, session_id, tool, approved_revision)
        token = CURRENT_FENCE.set(None)
        try:
            async with session_factory() as other:
                assert await stop_generation(other, session_id)
                await other.commit()
        finally:
            CURRENT_FENCE.reset(token)

    monkeypatch.setattr(runtime, "validate_source_execution", stop_after_guard)
    async with session_factory() as db:
        fence = await claim(db, case.session_id, lease_seconds=30) if owned else None
        if owned:
            assert fence is not None
        session = await load_session(db, case.session_id)
        call = next(item for item in session.tool_calls if item.id == call_id)
        token = CURRENT_FENCE.set(fence)
        try:
            if owned:
                with pytest.raises(FenceLostError):
                    await case.supervisor._execute_tool_call(db, session, call)
                await db.rollback()
            else:
                assert not await case.supervisor._execute_tool_call(db, session, call)
        finally:
            CURRENT_FENCE.reset(token)
    assert case.client.calls == 0
