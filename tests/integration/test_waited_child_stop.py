"""Verify that a Stop of a waited child cancels its run and blocks its MCP egress."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agent import runtime
from agent.config import Settings
from agent.database import init_database, session_factory
from agent.events import broker
from agent.message_tree import append_message
from agent.models import Approval, Project, Session, ToolCall
from agent.run_finish import finish_run
from agent.runtime import AgentSupervisor
from agent.session_budget_store import BudgetExceeded
from agent.session_fence import CURRENT_FENCE, claim, stop_generation
from agent.session_service import create_session_record
from agent.tools.base import Tool
from tests.integration.test_runtime_mcp_audit import audit_rows, execute
from tests.integration.test_runtime_source_revision import RuntimeCase, add_call
from tests.integration.test_runtime_source_revision import runtime_case as runtime_case


async def stop_child(case: RuntimeCase) -> None:
    token = CURRENT_FENCE.set(None)
    try:
        async with session_factory() as db:
            assert await stop_generation(db, case.session_id)
            await db.commit()
    finally:
        CURRENT_FENCE.reset(token)


@pytest.mark.parametrize("stage", ["guard", "started"])
async def test_child_stop_under_parent_fence_blocks_mcp_egress(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    async with session_factory() as db:
        child = await db.get(Session, case.session_id)
        assert child is not None
        parent = Session(
            project_id=child.project_id,
            created_by_user_id=case.user_id,
            mode="dev",
            title="Parent run",
            status="running",
            configuration={},
        )
        db.add(parent)
        await db.commit()
        fence = await claim(db, parent.id, lease_seconds=60)
    assert fence is not None
    if stage == "guard":
        original_guard = runtime.validate_source_execution

        async def stop_after_guard(
            db: AsyncSession, session_id: str, tool: Tool, approved_revision: str | None
        ) -> None:
            await original_guard(db, session_id, tool, approved_revision)
            await stop_child(case)

        monkeypatch.setattr(runtime, "validate_source_execution", stop_after_guard)
    else:
        original = case.supervisor._audit_egress

        async def audit(*args: Any) -> None:
            await original(*args)
            if args[5] == "started":
                await stop_child(case)

        monkeypatch.setattr(case.supervisor, "_audit_egress", audit)
    token = CURRENT_FENCE.set(fence)
    try:
        executed = await asyncio.wait_for(execute(case, call_id), timeout=3)
    finally:
        CURRENT_FENCE.reset(token)
    assert case.client.calls == 0
    rows = await audit_rows(case)
    if stage == "guard":
        assert not executed
        assert rows == []
    else:
        assert executed
        assert [row.status for row in rows] == ["started", "error"]
        assert json.loads(rows[-1].detail)["outcome"] == "guard_denied"


class BlockingRun:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.session_id: str | None = None

    async def __call__(self, session_id: str) -> None:
        self.session_id = session_id
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def waited_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[AgentSupervisor, Session, BlockingRun]:
    await init_database(recover_running=False)
    (tmp_path / "dev.md").write_text("Test system prompt")
    config = Settings(prompts_dir=tmp_path).model_copy(update=overrides)
    async with session_factory() as db:
        project = Project(name="waited-child", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        parent = await create_session_record(db, config, project, "Parent", "dev")
    supervisor = AgentSupervisor(config)
    run = BlockingRun()
    monkeypatch.setattr(supervisor, "_run", run)
    return supervisor, parent, run


async def test_stop_of_waited_child_cancels_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, parent, run = await waited_case(tmp_path, monkeypatch)
    spawn = asyncio.create_task(supervisor._spawn_child(parent, "dev", "Child", True, None))
    try:
        await asyncio.wait_for(run.started.wait(), timeout=3)
        assert run.session_id is not None
        await asyncio.wait_for(supervisor.stop(run.session_id), timeout=3)
        result = await asyncio.wait_for(spawn, timeout=3)
    finally:
        spawn.cancel()
        await asyncio.gather(spawn, return_exceptions=True)
    assert run.cancelled.is_set()
    assert result["session_id"] == run.session_id
    assert result["status"] == "stopped"
    assert run.session_id not in supervisor._tasks


async def test_cancelled_parent_cancels_waited_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, parent, run = await waited_case(tmp_path, monkeypatch)
    spawn = asyncio.create_task(supervisor._spawn_child(parent, "dev", "Child", True, None))
    try:
        await asyncio.wait_for(run.started.wait(), timeout=3)
        spawn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(spawn, timeout=3)
    finally:
        spawn.cancel()
        await asyncio.gather(spawn, return_exceptions=True)
    assert run.cancelled.is_set()
    assert run.session_id not in supervisor._tasks


async def test_waited_child_timeout_stops_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, parent, run = await waited_case(tmp_path, monkeypatch, max_child_wait_seconds=0.05)
    with pytest.raises(RuntimeError, match="^Child agent exceeded its wait timeout$"):
        await asyncio.wait_for(
            supervisor._spawn_child(parent, "dev", "Child", True, None), timeout=3
        )
    assert run.cancelled.is_set()
    assert run.session_id is not None
    assert run.session_id not in supervisor._tasks
    async with session_factory() as db:
        child = await db.get(Session, run.session_id)
        assert child is not None and child.status == "stopped"


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_waited_child_keeps_waiting_across_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    supervisor, parent, _ = await waited_case(
        tmp_path, monkeypatch, max_parallel_sessions=1, max_child_wait_seconds=0.5
    )
    paused = asyncio.Event()
    approval_id = "child-write-approval"
    child_id = ""
    budget_checks = 0
    check_budget = supervisor._check_tool_budget

    async def budget(db: Any, session_id: str) -> None:
        nonlocal budget_checks
        budget_checks += 1
        await check_budget(db, session_id)

    monkeypatch.setattr(supervisor, "_check_tool_budget", budget)

    async def run(session_id: str) -> None:
        nonlocal child_id
        child_id = session_id
        async with session_factory() as db:
            child = await db.get(Session, session_id)
            assert child is not None
            if not paused.is_set():
                db.add(
                    ToolCall(
                        id="child-write",
                        session_id=session_id,
                        name="write_file",
                        arguments={"path": "post.md", "content": "Post"},
                        risk_level="write",
                        status="awaiting_confirmation",
                    )
                )
                await db.flush()
                db.add(Approval(id=approval_id, tool_call_id="child-write"))
                await append_message(
                    db,
                    session_id,
                    role="assistant",
                    content="\n\n",
                    tool_calls=[{"id": "child-write"}],
                )
                child.status = "awaiting_confirmation"
                await db.commit()
                paused.set()
                return
            await append_message(db, session_id, role="assistant", content=f"Post: {decision}")
            await finish_run(db, session_id, 1, "completed")
            await db.commit()

    monkeypatch.setattr(supervisor, "_run", run)
    async with asyncio.timeout(5), supervisor._semaphore:
        spawn = asyncio.create_task(supervisor._spawn_child(parent, "dev", "Child", True, None))
        try:
            await paused.wait()
            # Approval time must not spend the child's execution timeout.
            await asyncio.sleep(0.65)
            assert not spawn.done(), "A pending child approval is not a final response"
            for _ in range(100):
                await broker.broadcast(
                    {"session_id": child_id, "type": "reasoning.delta", "payload": {"delta": "x"}}
                )
            await supervisor.resolve_approval(approval_id, decision, None)
            result = await spawn
        finally:
            spawn.cancel()
            await asyncio.gather(spawn, return_exceptions=True)
    assert result["status"] == "completed"
    assert result["response"] == f"Post: {decision}"
    assert budget_checks < 10, "Stream deltas must not trigger repeated database reads"
    assert child_id not in supervisor._waited_children
    assert child_id not in supervisor._tasks or supervisor._tasks[child_id].done()


@pytest.mark.parametrize("reason", ["cancel", "budget"])
async def test_ended_waiter_withdraws_the_childs_pending_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    supervisor, parent, _ = await waited_case(tmp_path, monkeypatch)
    paused = asyncio.Event()
    child_id = ""

    async def run(session_id: str) -> None:
        nonlocal child_id
        child_id = session_id
        async with session_factory() as db:
            child = await db.get(Session, session_id)
            assert child is not None
            child.status = "awaiting_confirmation"
            db.add(
                ToolCall(
                    id="pending-write",
                    session_id=session_id,
                    name="write_file",
                    arguments={"path": "post.md", "content": "Post"},
                    risk_level="write",
                    status="awaiting_confirmation",
                )
            )
            await db.flush()
            db.add(Approval(id="pending-approval", tool_call_id="pending-write"))
            await db.commit()
        paused.set()

    monkeypatch.setattr(supervisor, "_run", run)
    check_budget = supervisor._check_tool_budget

    async def budget(db: Any, session_id: str) -> None:
        if reason == "budget" and paused.is_set():
            raise BudgetExceeded("deadline_exceeded")
        await check_budget(db, session_id)

    monkeypatch.setattr(supervisor, "_check_tool_budget", budget)
    async with asyncio.timeout(5):
        spawn = asyncio.create_task(supervisor._spawn_child(parent, "dev", "Child", True, None))
        try:
            await paused.wait()
            if reason == "cancel":
                spawn.cancel()
            with pytest.raises(asyncio.CancelledError if reason == "cancel" else BudgetExceeded):
                await spawn
            async with session_factory() as db:
                child = await db.get(Session, child_id)
                approval = await db.get(Approval, "pending-approval")
                assert child is not None and child.status == "stopped"
                assert approval is not None and approval.status == "cancelled"
        finally:
            spawn.cancel()
            await asyncio.gather(spawn, return_exceptions=True)
