from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event as sa_event
from sqlalchemy import func, select, text, update

from agent import events, runtime
from agent.config import get_settings
from agent.database import init_database, session_factory
from agent.models import Event, Session
from agent.runtime import AgentSupervisor
from agent.session_fence import CURRENT_FENCE, FenceLostError, claim, install_commit_guard
from tests.integration.test_runtime_start import make_session


async def outcome_rows(session_id: str) -> list[Any]:
    from agent.models import SessionRunOutcome

    async with session_factory() as db:
        return list(
            await db.scalars(
                select(SessionRunOutcome)
                .where(SessionRunOutcome.session_id == session_id)
                .order_by(SessionRunOutcome.generation)
            )
        )


async def status_events(session_id: str) -> list[Event]:
    async with session_factory() as db:
        return list(await db.scalars(select(Event).where(Event.session_id == session_id)))


@pytest.mark.parametrize("status", ["completed", "error", "stopped", "interrupted"])
@pytest.mark.parametrize("write", ["commit", "flush", "autoflush"])
async def test_fenced_orm_terminal_transition_requires_finish(
    tmp_path: Path, status: str, write: str
) -> None:
    install_commit_guard()
    session_id = await make_session(tmp_path, run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            session.status = status
            with pytest.raises(FenceLostError, match="finish_run"):
                if write == "autoflush":
                    await db.execute(select(Session))
                else:
                    await getattr(db, write)()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == "pending"
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []


async def test_finish_permission_does_not_authorize_different_orm_outcome(tmp_path: Path) -> None:
    from agent.run_finish import finish_run

    install_commit_guard()
    session_id = await make_session(tmp_path, run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            await finish_run(db, session_id, 1, "completed")
            session.status = "error"
            with pytest.raises(FenceLostError, match="finish_run"):
                await db.commit()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    assert await outcome_rows(session_id) == []


@pytest.mark.parametrize(
    "status", ["pending", "running", "awaiting_confirmation", "error", "stopped", "interrupted"]
)
@pytest.mark.parametrize("write", ["commit", "flush", "autoflush"])
async def test_finish_rejects_every_different_orm_status(
    tmp_path: Path, status: str, write: str
) -> None:
    from agent.run_finish import finish_run

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            await finish_run(db, session_id, 1, "completed")
            await db.refresh(session)
            session.status = status
            with pytest.raises(FenceLostError, match="finish_run"):
                if write == "autoflush":
                    await db.execute(select(Session))
                else:
                    await getattr(db, write)()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == "running"
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []


@pytest.mark.parametrize(
    "status", ["pending", "running", "awaiting_confirmation", "error", "stopped", "interrupted"]
)
@pytest.mark.parametrize("write", ["orm", "core", "text", "connection", "guarded"])
async def test_finish_commit_rejects_every_different_bulk_status(
    tmp_path: Path, status: str, write: str
) -> None:
    from agent.run_finish import finish_run
    from agent.session_fence import guarded_update

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            await finish_run(db, session_id, 1, "completed")
            if write == "guarded":
                await guarded_update(db, fence, {"status": status})
            elif write == "text":
                await db.execute(
                    text("UPDATE sessions SET status = :status WHERE id = :session_id"),
                    {"status": status, "session_id": session_id},
                )
            else:
                statement = (
                    update(Session if write == "orm" else Session.__table__)
                    .where(Session.id == session_id)
                    .values(status=status)
                    .execution_options(synchronize_session=False)
                )
                if write == "connection":
                    connection = await db.connection()
                    await connection.execute(statement)
                else:
                    await db.execute(statement)
            with pytest.raises(FenceLostError):
                await db.commit()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == "running"
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []


@pytest.mark.parametrize("status", ["pending", "error"])
async def test_finish_commit_checks_status_after_flush(tmp_path: Path, status: str) -> None:
    from agent.run_finish import finish_run

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            await finish_run(db, session_id, 1, "completed")

            def overwrite_status(sync_session: Any, flush_context: Any) -> None:
                sync_session.connection().execute(
                    update(Session.__table__).where(Session.id == session_id).values(status=status)
                )

            sa_event.listen(db.sync_session, "after_flush_postexec", overwrite_status)
            try:
                session.title = "force commit-time flush"
                with pytest.raises(FenceLostError):
                    await db.commit()
                await db.rollback()
            finally:
                sa_event.remove(db.sync_session, "after_flush_postexec", overwrite_status)
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == "running"
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []


@pytest.mark.parametrize("end_savepoint", ["commit", "rollback"])
async def test_savepoint_end_does_not_erase_outer_finish_guard(
    tmp_path: Path, end_savepoint: str
) -> None:
    from agent.run_finish import finish_run

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            await finish_run(db, session_id, 1, "completed")
            savepoint = await db.begin_nested()
            await getattr(savepoint, end_savepoint)()
            connection = await db.connection()
            await connection.execute(
                update(Session.__table__).where(Session.id == session_id).values(status="pending")
            )
            with pytest.raises(FenceLostError):
                await db.commit()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
async def test_committed_finish_cannot_be_reopened_by_connection(
    tmp_path: Path, status: str
) -> None:
    from agent.run_finish import finish_run
    from agent.session_fence import release

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            await finish_run(db, session_id, 1, "completed")
            await db.commit()
            connection = await db.connection()
            await connection.execute(
                update(Session.__table__).where(Session.id == session_id).values(status=status)
            )
            with pytest.raises(FenceLostError):
                await db.commit()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == "completed"
        await release(db, fence)
        assert await claim(db, session_id, lease_seconds=60, expected_generation=1) is None
    assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [
        (1, "completed")
    ]
    assert len(await status_events(session_id)) == 1


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
@pytest.mark.parametrize("operation", ["claim", "renew", "fence_check", "guarded_update"])
async def test_recorded_outcome_prevents_ownership_of_reopened_generation(
    tmp_path: Path, status: str, operation: str
) -> None:
    from agent.run_finish import finish_run
    from agent.session_fence import fence_check, guarded_update, release, renew

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
        assert fence is not None
        await finish_run(db, session_id, 1, "completed")
        await db.commit()
        await db.execute(update(Session).where(Session.id == session_id).values(status=status))
        await db.commit()
    async with session_factory() as db:
        if operation == "claim":
            await release(db, fence)
            assert await claim(db, session_id, lease_seconds=60, expected_generation=1) is None
        elif operation == "renew":
            assert not await renew(db, fence, lease_seconds=60)
        else:
            with pytest.raises(FenceLostError):
                if operation == "fence_check":
                    await fence_check(db, fence)
                else:
                    await guarded_update(db, fence, {"title": "stale write"})
            await db.rollback()
    assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [
        (1, "completed")
    ]
    assert len(await status_events(session_id)) == 1


@pytest.mark.parametrize("end_transaction", ["rollback", "close"])
async def test_finish_permission_does_not_survive_transaction_end(
    tmp_path: Path, end_transaction: str
) -> None:
    from agent.run_finish import finish_run

    install_commit_guard()
    session_id = await make_session(tmp_path, run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            await finish_run(db, session_id, 1, "completed")
            await getattr(db, end_transaction)()
            session = await db.get(Session, session_id)
            session.status = "completed"
            with pytest.raises(FenceLostError, match="finish_run"):
                await db.flush()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []


@pytest.mark.parametrize("status", ["completed", "error", "stopped", "interrupted"])
async def test_matching_orm_terminal_change_is_allowed_in_finish_transaction(
    tmp_path: Path, status: str
) -> None:
    from agent.run_finish import finish_run
    from agent.session_fence import release

    install_commit_guard()
    session_id = await make_session(tmp_path, run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            await finish_run(db, session_id, 1, status)
            session.status = status
            await db.commit()
    finally:
        CURRENT_FENCE.reset(token)
    assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [(1, status)]
    assert len(await status_events(session_id)) == 1
    async with session_factory() as db:
        await release(db, fence)
        assert await claim(db, session_id, lease_seconds=60, expected_generation=1) is None


@pytest.mark.parametrize("finish_inside", [False, True])
@pytest.mark.parametrize("end_savepoint", ["commit", "rollback"])
async def test_finish_permission_follows_savepoint_boundary(
    tmp_path: Path, finish_inside: bool, end_savepoint: str
) -> None:
    from agent.run_finish import finish_run
    from agent.session_fence import release

    install_commit_guard()
    session_id = await make_session(tmp_path, status="running", run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    status = "error" if finish_inside and end_savepoint == "rollback" else "completed"
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            if not finish_inside:
                await finish_run(db, session_id, 1, "completed")
            savepoint = await db.begin_nested()
            if finish_inside:
                await finish_run(db, session_id, 1, "completed")
            await getattr(savepoint, end_savepoint)()
            if status == "error":
                await finish_run(db, session_id, 1, status)
            await db.commit()
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == status
        await release(db, fence)
        assert await claim(db, session_id, lease_seconds=60, expected_generation=1) is None
    assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [(1, status)]
    assert len(await status_events(session_id)) == 1


@pytest.mark.parametrize("terminal_child", [False, True])
async def test_cascade_does_not_descend_after_generation_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_child: bool
) -> None:
    from agent.run_finish import finish_run
    from agent.session_fence import new_generation

    parent_id = await make_session(tmp_path, run_generation=1)
    child_id = await make_session(tmp_path, run_generation=1, parent_id=parent_id)
    if terminal_child:
        async with session_factory() as db:
            await finish_run(db, child_id, 1, "completed")
            await db.commit()
    grandchild_id = str(uuid.uuid4())

    async def start_before_scoped_finish(
        db: Any, session_id: str, *args: Any, **kwargs: Any
    ) -> Any:
        if session_id == child_id:
            await new_generation(db, child_id)
            child = await db.get(Session, child_id)
            child.status = "pending"
            db.add(
                Session(
                    id=grandchild_id,
                    project_id=child.project_id,
                    parent_id=child_id,
                    title="new branch",
                    configuration={},
                    run_generation=1,
                )
            )
            await db.flush()
        return await finish_run(db, session_id, *args, **kwargs)

    monkeypatch.setattr(runtime, "finish_run", start_before_scoped_finish)
    await AgentSupervisor(get_settings()).stop(parent_id)
    async with session_factory() as db:
        assert (await db.get(Session, child_id)).run_generation == 2
        assert (await db.get(Session, grandchild_id)).status == "pending"
    assert await outcome_rows(grandchild_id) == []


@pytest.mark.parametrize("status", ["completed", "error", "stopped", "interrupted"])
async def test_finish_is_atomic_and_first_outcome_wins(tmp_path: Path, status: str) -> None:
    from agent.run_finish import finish_run

    session_id = await make_session(tmp_path, run_generation=4)
    async with session_factory() as db:
        assert await finish_run(db, session_id, 4, status, reason="test")
        await db.rollback()
    assert await outcome_rows(session_id) == []
    assert await status_events(session_id) == []
    async with session_factory() as db:
        assert (await db.get(Session, session_id)).status == "pending"
        assert await finish_run(db, session_id, 4, status, reason="first")
        await db.commit()
        assert await finish_run(db, session_id, 4, "error", error="late") == []
        await db.commit()
        stored = await db.get(Session, session_id, populate_existing=True)
        assert stored.status == status and stored.error is None
    rows = await outcome_rows(session_id)
    assert [(row.generation, row.status, row.reason) for row in rows] == [(4, status, "first")]
    assert len(await status_events(session_id)) == 1


async def test_publish_failure_keeps_committed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.run_finish import finish_run

    session_id = await make_session(tmp_path, run_generation=1)
    sent: list[str] = []

    async def broken(payload: dict[str, Any]) -> None:
        sent.append(payload["type"])
        raise RuntimeError("delivery failed")

    monkeypatch.setattr(events.broker, "broadcast", broken)
    async with session_factory() as db:
        pending = await finish_run(db, session_id, 1, "completed")
        pending.append(await events.add_event(db, "extra", {}, session_id))
        await db.commit()
    await events.publish(pending)
    await AgentSupervisor(get_settings())._mark_error(session_id, RuntimeError("late"))
    assert sent == ["session.status", "extra"]
    assert (await outcome_rows(session_id))[0].status == "completed"
    assert len(await status_events(session_id)) == 2


async def test_stop_records_own_generation_once_and_fences_executor(tmp_path: Path) -> None:
    from agent.run_finish import finish_run

    session_id = await make_session(tmp_path, run_generation=6)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    await AgentSupervisor(get_settings()).stop(session_id)
    await AgentSupervisor(get_settings()).stop(session_id)
    async with session_factory() as db:
        assert await finish_run(db, session_id, 6, "completed") == []
        await db.commit()
        stored = await db.get(Session, session_id)
        assert stored.run_generation == 7 and stored.status == "stopped"
    assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [
        (6, "stopped")
    ]
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            db.add(Event(session_id=session_id, type="stale", payload={}))
            with pytest.raises(FenceLostError):
                await db.commit()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)


async def test_stop_after_completion_is_control_only(tmp_path: Path) -> None:
    from agent.run_finish import finish_run

    session_id = await make_session(tmp_path, run_generation=2)
    async with session_factory() as db:
        await finish_run(db, session_id, 2, "completed")
        await db.commit()
    supervisor = AgentSupervisor(get_settings())
    assert not await supervisor.stop_scoped(session_id, 2)
    await supervisor.stop(session_id)
    await supervisor.stop(session_id)
    assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [
        (2, "completed")
    ]
    assert len(await status_events(session_id)) == 2


async def test_stop_cascade_is_durable_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = await make_session(tmp_path, run_generation=2)
    child = await make_session(tmp_path, run_generation=3, parent_id=parent)
    grandchild = await make_session(tmp_path, run_generation=5, parent_id=child)

    async def fail_publish(*args: Any) -> None:
        raise RuntimeError("after commit")

    monkeypatch.setattr(runtime, "publish", fail_publish, raising=False)
    with pytest.raises(RuntimeError, match="after commit"):
        await AgentSupervisor(get_settings()).stop(parent)
    for session_id, generation in [(parent, 2), (child, 3), (grandchild, 5)]:
        assert [(row.generation, row.status) for row in await outcome_rows(session_id)] == [
            (generation, "stopped")
        ]


async def test_finish_under_guard_and_later_writes_are_rejected(tmp_path: Path) -> None:
    from agent.run_finish import finish_run

    install_commit_guard()
    session_id = await make_session(tmp_path, run_generation=1)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    token = CURRENT_FENCE.set(fence)
    try:
        async with session_factory() as db:
            await finish_run(db, session_id, 1, "completed")
            await db.rollback()
            await finish_run(db, session_id, 1, "completed")
            await db.commit()
            db.add(Event(session_id=session_id, type="late", payload={}))
            with pytest.raises(FenceLostError):
                await db.commit()
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    assert len(await status_events(session_id)) == 1


async def test_interrupted_cannot_be_claimed(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path, run_generation=2, status="interrupted")
    async with session_factory() as db:
        assert await claim(db, session_id, lease_seconds=60) is None


class FinishingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__(get_settings())
        self.runs: list[str] = []

    async def _run(self, session_id: str) -> None:
        from agent.run_finish import finish_run

        self.runs.append(session_id)
        fence = CURRENT_FENCE.get()
        assert fence is not None and fence.session_id == session_id
        async with session_factory() as db:
            await finish_run(db, session_id, fence.generation, "completed")
            await db.commit()


async def test_competing_finishers_produce_one_outcome(tmp_path: Path) -> None:
    from agent.run_finish import finish_run

    session_id = await make_session(tmp_path, run_generation=3)

    async def finish(status: str) -> int:
        async with session_factory() as db:
            rows = await finish_run(db, session_id, 3, status)
            await db.commit()
            return len(rows)

    assert sum(await asyncio.gather(finish("completed"), finish("error"))) == 1
    assert len(await outcome_rows(session_id)) == 1
    async with session_factory() as db:
        assert (
            await db.scalar(
                select(func.count()).select_from(Event).where(Event.session_id == session_id)
            )
            == 1
        )


async def test_waited_child_owns_fence_without_second_slot(tmp_path: Path) -> None:
    from agent.models import Project
    from agent.session_service import create_session_record

    supervisor = FinishingSupervisor()
    supervisor._semaphore = asyncio.Semaphore(1)
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name="children", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        parent = await create_session_record(db, get_settings(), project, "parent", "ask")
        parent_id = parent.id
        fence = await claim(db, parent_id, lease_seconds=60)
    token = CURRENT_FENCE.set(fence)
    try:
        async with supervisor._semaphore:
            async with asyncio.timeout(3):
                result = await supervisor._spawn_child(parent, "ask", "child", True, None)
        rows = await outcome_rows(result["session_id"])
        assert [(row.generation, row.status) for row in rows] == [(1, "completed")]
        assert CURRENT_FENCE.get() == fence
    finally:
        CURRENT_FENCE.reset(token)
        await supervisor.shutdown()


async def test_cascade_rolls_back_as_a_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = await make_session(tmp_path, run_generation=1)
    child = await make_session(tmp_path, run_generation=1, parent_id=parent)
    supervisor = AgentSupervisor(get_settings())
    original = supervisor._cancel_unfinished_tool_calls

    async def fail_on_child(db: Any, session_id: str) -> list[str]:
        if session_id == child:
            raise RuntimeError("cascade interrupted")
        return await original(db, session_id)

    monkeypatch.setattr(supervisor, "_cancel_unfinished_tool_calls", fail_on_child)
    with pytest.raises(RuntimeError, match="cascade interrupted"):
        await supervisor.stop(parent)
    for session_id in (parent, child):
        async with session_factory() as db:
            stored = await db.get(Session, session_id)
            assert stored.status == "pending" and stored.run_generation == 1
        assert await outcome_rows(session_id) == []
        assert await status_events(session_id) == []


async def test_stop_prevents_late_child_creation(tmp_path: Path) -> None:
    supervisor = FinishingSupervisor()
    parent_id = await make_session(tmp_path, run_generation=1)
    async with session_factory() as db:
        parent = await db.get(Session, parent_id)
        fence = await claim(db, parent_id, lease_seconds=60)
    await supervisor.stop(parent_id)
    token = CURRENT_FENCE.set(fence)
    try:
        with pytest.raises(FenceLostError):
            await supervisor._spawn_child(parent, "ask", "child", True, None)
    finally:
        CURRENT_FENCE.reset(token)
    async with session_factory() as db:
        assert (
            await db.scalar(
                select(func.count()).select_from(Session).where(Session.parent_id == parent_id)
            )
            == 0
        )


async def test_waited_child_is_owned_before_creation_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.models import Project
    from agent.session_service import create_session_record

    supervisor = FinishingSupervisor()
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name="reserved-child", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        parent = await create_session_record(db, get_settings(), project, "parent", "ask")

    async def check_owner(pending: list[Any]) -> None:
        for event in pending:
            if event.payload["type"] == "session.child_created":
                async with session_factory() as db:
                    child = await db.get(Session, event.payload["payload"]["child_id"])
                    assert child.owner_token is not None
                    assert child.run_generation == 1

    monkeypatch.setattr(runtime, "publish", check_owner)
    monkeypatch.setattr(events, "publish", check_owner)
    result = await supervisor._spawn_child(parent, "ask", "child", True, None)
    assert result["status"] == "completed"


async def test_cascade_preserves_terminal_child_but_reaches_active_descendants(
    tmp_path: Path,
) -> None:
    from agent.run_finish import finish_run

    parent = await make_session(tmp_path, run_generation=1)
    child = await make_session(tmp_path, run_generation=1, parent_id=parent)
    grandchild = await make_session(tmp_path, run_generation=1, parent_id=child)
    async with session_factory() as db:
        await finish_run(db, child, 1, "completed")
        await db.commit()
    await AgentSupervisor(get_settings()).stop(parent)
    async with session_factory() as db:
        assert (await db.get(Session, child)).status == "completed"
        assert (await db.get(Session, grandchild)).status == "stopped"
    assert [(row.generation, row.status) for row in await outcome_rows(child)] == [(1, "completed")]
