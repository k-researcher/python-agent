from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, select, text, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent import runtime
from agent.config import get_settings
from agent.database import configure_sqlite_connection
from agent.models import Approval, Event, Project, Session, SessionRunOutcome, ToolCall
from agent.run_finish import finish_run
from agent.runtime import AgentSupervisor
from agent.session_fence import (
    CURRENT_FENCE,
    FenceLostError,
    claim,
    fence_check,
    install_commit_guard,
    new_generation,
)


def prepare_schema(connection: Connection) -> None:
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.attributes["connection"] = connection
    command.upgrade(config, "0015")
    command.upgrade(config, "0016")
    command.upgrade(config, "0017")
    command.upgrade(config, "0018")


@pytest.mark.parametrize("stop_kind", ["plain", "scoped", "cascade"])
async def test_stopped_message_call_is_never_recovered(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_kind: str,
) -> None:
    from sqlalchemy.orm import attributes, selectinload

    from agent import api
    from agent.models import Message, SessionInbox
    from agent.schemas import MessageCreate

    sender_id = await seed(run_store, tmp_path)
    parent_id = await seed(run_store, tmp_path / "parent")
    async with run_store() as db:
        sender = await db.get(Session, sender_id)
        assert sender is not None
        if stop_kind == "cascade":
            sender.parent_id = parent_id
        db.add(
            ToolCall(
                id="cancelled-message",
                session_id=sender_id,
                name="message_to_session",
                arguments={"session_id": parent_id, "message": "queued"},
                risk_level="network_access",
                status="running",
            )
        )
        await db.commit()
        old_fence = await claim(db, sender_id, lease_seconds=60)
    assert old_fence is not None
    supervisor = AgentSupervisor(get_settings())

    async def no_dispatch(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(supervisor, "start", no_dispatch)
    monkeypatch.setattr(supervisor, "_run_guarded", no_dispatch)
    monkeypatch.setattr(api, "_dispatch_session_run", no_dispatch)
    if stop_kind == "scoped":
        assert await supervisor.stop_scoped(sender_id, 1)
    else:
        await supervisor.stop(parent_id if stop_kind == "cascade" else sender_id)
    async with run_store() as db:
        stopped_call = await db.get(ToolCall, "cancelled-message")
        assert stopped_call is not None and stopped_call.status == "cancelled"
        assert stopped_call.result is not None and stopped_call.result["success"] is False
        await api.add_message(sender_id, MessageCreate(content="Continue"), db)
    assert await AgentSupervisor.start(supervisor, sender_id)
    await supervisor._tasks[sender_id]
    await supervisor.reconcile_pending()
    async with run_store() as db:
        sender = await db.get(Session, sender_id)
        assert sender is not None
        calls = list(
            await db.scalars(
                select(ToolCall)
                .where(ToolCall.session_id == sender_id)
                .options(selectinload(ToolCall.approval))
            )
        )
        attributes.set_committed_value(sender, "tool_calls", calls)
        assert await supervisor._resume_tool_calls(db, sender) == "ready"
        assert calls[0].status == "cancelled"
        closing = list(
            await db.scalars(select(Message).where(Message.tool_call_id == "cancelled-message"))
        )
        assert len(closing) == 1
        assert '"success": false' in (closing[0].content or "")
        assert await db.scalar(select(SessionInbox.id).limit(1)) is None
        token = CURRENT_FENCE.set(old_fence)
        try:
            with pytest.raises(FenceLostError):
                await supervisor._execute_message_call(db, sender, calls[0])
            await db.rollback()
        finally:
            CURRENT_FENCE.reset(token)


@pytest.fixture(params=["sqlite", "postgresql"])
async def run_store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    schema = f"run_test_{uuid.uuid4().hex}"
    if request.param == "postgresql":
        url = os.environ.get("RUN_FINISH_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("RUN_FINISH_TEST_POSTGRES_URL is not set")
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
        event.listen(
            engine.sync_engine,
            "connect",
            lambda db, _: configure_sqlite_connection(db, foreign_keys=False),
        )
    try:
        async with engine.begin() as connection:
            if request.param == "postgresql":
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.run_sync(prepare_schema)
        if request.param == "sqlite":
            await engine.dispose()
            event.listen(
                engine.sync_engine, "connect", lambda db, _: configure_sqlite_connection(db)
            )
        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(runtime, "session_factory", factory)
        from agent import attempt_audit, context_inference, context_state

        monkeypatch.setattr(attempt_audit, "session_factory", factory)
        monkeypatch.setattr(context_inference, "session_factory", factory)
        monkeypatch.setattr(context_state, "session_factory", factory)
        install_commit_guard()
        yield factory
    finally:
        if request.param == "postgresql":
            async with engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


async def seed(factory: async_sessionmaker[AsyncSession], tmp_path: Path) -> str:
    async with factory() as db:
        project = Project(name="races", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="run", configuration={}, run_generation=1)
        db.add(session)
        await db.commit()
        return session.id


async def seed_call(
    factory: async_sessionmaker[AsyncSession], session_id: str, *, approval: bool = False
) -> tuple[str, str | None]:
    async with factory() as db:
        call = ToolCall(
            id=uuid.uuid4().hex,
            session_id=session_id,
            name="read_file",
            arguments={},
            risk_level="low",
            status="awaiting_confirmation" if approval else "pending",
        )
        db.add(call)
        approval_row = Approval(tool_call=call) if approval else None
        if approval_row is not None:
            db.add(approval_row)
        await db.commit()
        return call.id, approval_row.id if approval_row else None


async def inbox_sessions(
    factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> tuple[str, str, str]:
    first = await seed(factory, tmp_path)
    async with factory() as db:
        project_id = await db.scalar(select(Session.project_id).where(Session.id == first))
        second = Session(project_id=project_id, title="second", run_generation=1)
        target = Session(
            project_id=project_id, title="target", run_generation=1, status="completed"
        )
        db.add_all([second, target])
        await db.commit()
        return first, second.id, target.id


async def test_postgres_inbox_capacity_serializes_competing_posts(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from sqlalchemy import func
    from sqlalchemy.exc import DBAPIError

    from agent.models import SessionInbox
    from agent.session_inbox import enqueue, retryable
    from agent.tools.base import ToolError

    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    first, second, target = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        for index in range(19):
            db.add(
                SessionInbox(
                    target_session_id=target,
                    sender_session_id=first,
                    kind="message",
                    content="seed",
                    hop=0,
                    dedupe_key=f"seed:{index}",
                )
            )
        await db.commit()
        first_fence = await claim(db, first, lease_seconds=60)
        second_fence = await claim(db, second, lease_seconds=60)
    inserted, collision, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def winner() -> None:
        token = CURRENT_FENCE.set(first_fence)
        try:
            async with run_store() as db:
                await enqueue(db, first, target, "winner", "winner")
                inserted.set()
                await collision.wait()
                await db.commit()
                committed.set()
        finally:
            CURRENT_FENCE.reset(token)

    async def contender() -> None:
        await inserted.wait()
        token = CURRENT_FENCE.set(second_fence)
        try:
            async with run_store() as db:
                with pytest.raises(DBAPIError) as caught:
                    await enqueue(db, second, target, "contender", "contender")
                assert retryable(caught.value)
                await db.rollback()
                collision.set()
                await committed.wait()
                with pytest.raises(ToolError, match="Inbox full"):
                    await enqueue(db, second, target, "contender", "contender")
        finally:
            CURRENT_FENCE.reset(token)

    async with asyncio.timeout(10):
        await asyncio.gather(winner(), contender())
    async with run_store() as db:
        assert await db.scalar(select(func.count()).select_from(SessionInbox)) == 20


async def test_postgres_mutual_messages_use_nowait_without_deadlock(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from sqlalchemy import func
    from sqlalchemy.exc import DBAPIError

    from agent.models import SessionInbox
    from agent.session_inbox import enqueue, retryable

    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    first, second, _ = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        fences = [await claim(db, session_id, lease_seconds=60) for session_id in (first, second)]
    barrier = asyncio.Barrier(2)

    async def send(sender: str, target: str, fence: Any) -> bool:
        token = CURRENT_FENCE.set(fence)
        try:
            async with run_store() as db:
                await fence_check(db, fence)
                await barrier.wait()
                try:
                    await enqueue(db, sender, target, "mutual", f"mutual:{sender}")
                    await db.commit()
                    return True
                except DBAPIError as error:
                    assert retryable(error)
                    await db.rollback()
                    return False
        finally:
            CURRENT_FENCE.reset(token)

    async with asyncio.timeout(10):
        accepted = await asyncio.gather(
            send(first, second, fences[0]), send(second, first, fences[1])
        )
    for sender, target, fence, done in zip(
        (first, second), (second, first), fences, accepted, strict=True
    ):
        if not done:
            token = CURRENT_FENCE.set(fence)
            try:
                async with run_store() as db:
                    await enqueue(db, sender, target, "mutual", f"mutual:{sender}")
                    await db.commit()
            finally:
                CURRENT_FENCE.reset(token)
    async with run_store() as db:
        assert await db.scalar(select(func.count()).select_from(SessionInbox)) == 2


async def test_two_inbox_auto_claims_increment_generation_once(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from agent.models import SessionInbox
    from agent.session_inbox import auto_claim

    first, _, target = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        db.add(
            SessionInbox(
                target_session_id=target,
                sender_session_id=first,
                kind="message",
                content="wake",
                hop=0,
                dedupe_key="wake",
            )
        )
        await db.commit()
    barrier = asyncio.Barrier(2)

    async def wake() -> int | None:
        await barrier.wait()
        async with run_store() as db:
            generation = await auto_claim(db, target, 1, get_settings())
            await db.commit()
            return generation

    async with asyncio.timeout(10):
        results = await asyncio.gather(wake(), wake())
    assert results.count(2) == 1 and results.count(None) == 1


async def test_postgres_locked_target_gives_sender_tool_error(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    async with asyncio.timeout(10):
        sender, _, target = await inbox_sessions(run_store, tmp_path)

        from sqlalchemy import func

        from agent.llm import LLMResponse
        from agent.message_tree import append_message
        from agent.models import Message, SessionInbox
        from agent.session_fence import release
        from agent.session_inbox import enqueue

        supervisor = AgentSupervisor(get_settings())
        seen: list[Any] = []

        class Reply:
            endpoint = "https://inference.invalid/chat"

            async def chat(self, messages: Any, tools: Any, **options: Any) -> Any:
                seen.append(messages)
                return LLMResponse("Sender continued", [], "stop", 1, 1)

        monkeypatch.setattr(supervisor, "llm", Reply())
        monkeypatch.setattr(supervisor, "_run", supervisor._run_session)
        async with run_store() as db:
            await append_message(db, sender, role="user", content="Send data")
            await append_message(
                db,
                sender,
                role="assistant",
                tool_calls=[
                    {
                        "id": "locked-call",
                        "type": "function",
                        "function": {
                            "name": "message_to_session",
                            "arguments": {"session_id": target, "message": "data"},
                        },
                    }
                ],
            )
            db.add(
                ToolCall(
                    id="locked-call",
                    session_id=sender,
                    name="message_to_session",
                    arguments={"session_id": target, "message": "data"},
                    risk_level="network_access",
                    status="pending",
                )
            )
            await db.commit()

        lock_acquired = asyncio.Event()
        release_lock = asyncio.Event()

        async def hold_lock() -> None:
            async with run_store() as db:
                await db.execute(select(Session.id).where(Session.id == target).with_for_update())
                lock_acquired.set()
                await release_lock.wait()
                await db.rollback()

        lock_task = asyncio.create_task(hold_lock())
        try:
            await lock_acquired.wait()
            # Keep the target locked through retries and the next inference request.
            await supervisor.run_job(sender, 1)
            release_lock.set()
            await lock_task
        finally:
            release_lock.set()
            if not lock_task.done():
                lock_task.cancel()
            await asyncio.gather(lock_task, return_exceptions=True)

        async with run_store() as db:
            call = await db.get(ToolCall, "locked-call")
            assert call.status == "error" and call.result["success"] is False
            assert "Inbox temporarily busy" in call.result["error"]
            assert (await db.get(SessionRunOutcome, (sender, 1))).status == "completed"
            assert (await db.get(Session, sender)).status == "completed"
            assert await db.scalar(select(func.count()).select_from(SessionInbox)) == 0
            closing = list(
                await db.scalars(select(Message).where(Message.tool_call_id == "locked-call"))
            )
            assert len(closing) == 1 and '"success": false' in closing[0].content
        assert len(seen) == 1 and "Inbox temporarily busy" in str(seen[0])

        # Retry under a fresh run fence; repeated enqueue must keep one receipt.
        async with run_store() as db:
            await new_generation(db, sender)
            await db.execute(update(Session).where(Session.id == sender).values(status="pending"))
            await db.commit()
            fence = await claim(db, sender, lease_seconds=60)
        assert fence is not None
        token = CURRENT_FENCE.set(fence)
        try:
            async with run_store() as db:
                await enqueue(db, sender, target, "data", "tool:locked-call")
                await db.commit()
                await enqueue(db, sender, target, "data", "tool:locked-call")
                await db.commit()
        finally:
            CURRENT_FENCE.reset(token)
            async with run_store() as db:
                await release(db, fence)
        async with run_store() as db:
            assert await db.scalar(select(func.count()).select_from(SessionInbox)) == 1


async def test_inbox_delivery_rejects_expired_and_reclaimed_fence(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime, timedelta

    from agent.models import SessionInbox
    from agent.session_inbox import deliver

    first, target, _ = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        db.add(
            SessionInbox(
                target_session_id=target,
                sender_session_id=first,
                kind="message",
                content="pending",
                hop=0,
                dedupe_key="pending",
            )
        )
        await db.commit()
        old = await claim(db, target, lease_seconds=60)
        await db.execute(
            update(Session)
            .where(Session.id == target)
            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
        await db.commit()
        fresh = await claim(db, target, lease_seconds=60)
    assert fresh is not None and old is not None and fresh.epoch > old.epoch
    token = CURRENT_FENCE.set(old)
    try:
        async with run_store() as db:
            with pytest.raises(FenceLostError):
                await deliver(db, target)
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    token = CURRENT_FENCE.set(fresh)
    try:
        async with run_store() as db:
            assert (await deliver(db, target))[0]
            await db.commit()
    finally:
        CURRENT_FENCE.reset(token)


async def test_postgres_stop_waits_for_delivery_commit(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from agent.models import Message, SessionInbox
    from agent.session_inbox import deliver

    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    first, target, _ = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        db.add(
            SessionInbox(
                target_session_id=target,
                sender_session_id=first,
                kind="message",
                content="pending",
                hop=0,
                dedupe_key="pending",
            )
        )
        await db.commit()
        fence = await claim(db, target, lease_seconds=60)
    staged, stop_entered, commit = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pids: dict[str, int] = {}

    async def delivery() -> None:
        token = CURRENT_FENCE.set(fence)
        try:
            async with run_store() as db:
                pids["delivery"] = await db.scalar(text("SELECT pg_backend_pid()"))
                assert (await deliver(db, target))[0]
                staged.set()
                await commit.wait()
                await db.commit()
        finally:
            CURRENT_FENCE.reset(token)

    async def stop() -> None:
        await staged.wait()
        async with run_store() as db:
            pids["stop"] = await db.scalar(text("SELECT pg_backend_pid()"))
            stop_entered.set()
            await finish_run(db, target, None, "stopped", source="stop")
            await db.commit()

    delivery_task = asyncio.create_task(delivery())
    stop_task = asyncio.create_task(stop())
    try:
        await asyncio.wait_for(stop_entered.wait(), 10)
        await wait_for_blocker(run_store, pids["stop"], pids["delivery"], stop_task)
        commit.set()
        await asyncio.wait_for(asyncio.gather(delivery_task, stop_task), 10)
    finally:
        commit.set()
        for task in (delivery_task, stop_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(delivery_task, stop_task, return_exceptions=True)
    async with run_store() as db:
        rows = list(await db.scalars(select(Message).where(Message.session_id == target)))
        assert len(rows) == 1
        assert (await db.get(Session, target)).status == "stopped"
        receipt = await db.scalar(select(SessionInbox).where(SessionInbox.dedupe_key == "pending"))
        assert receipt.delivered_at is not None


async def test_postgres_child_finish_never_locks_notice_parent(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    parent, child, _ = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        await db.execute(
            update(Session).where(Session.id == child).values(parent_id=parent, notify_generation=1)
        )
        await db.commit()
    async with run_store() as locked:
        await locked.execute(select(Session).where(Session.id == parent).with_for_update())

        async def finish_child() -> None:
            async with run_store() as db:
                await finish_run(db, child, 1, "completed")
                await db.commit()

        async with asyncio.timeout(10):
            await finish_child()
        await locked.rollback()
    async with run_store() as db:
        outcome = await db.get(SessionRunOutcome, (child, 1))
        assert outcome.notice_target_id == parent


async def test_postgres_completion_and_post_leave_a_durable_wake(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from sqlalchemy.exc import DBAPIError

    from agent.session_fence import release
    from agent.session_inbox import auto_claim, deliver, enqueue, retryable

    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    sender, target, _ = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        sender_fence = await claim(db, sender, lease_seconds=60)
        target_fence = await claim(db, target, lease_seconds=60)
    checked, collision, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def complete() -> None:
        token = CURRENT_FENCE.set(target_fence)
        try:
            async with run_store() as db:
                assert not (await deliver(db, target))[0]
                checked.set()
                await collision.wait()
                await finish_run(db, target, 1, "completed")
                await db.commit()
        finally:
            CURRENT_FENCE.reset(token)
        async with run_store() as db:
            await release(db, target_fence)
        completed.set()

    async def send() -> None:
        await checked.wait()
        token = CURRENT_FENCE.set(sender_fence)
        try:
            async with run_store() as db:
                with pytest.raises(DBAPIError) as caught:
                    await enqueue(db, sender, target, "late", "late")
                assert retryable(caught.value)
                await db.rollback()
                collision.set()
                await completed.wait()
                await enqueue(db, sender, target, "late", "late")
                await db.commit()
        finally:
            CURRENT_FENCE.reset(token)

    async with asyncio.timeout(10):
        await asyncio.gather(complete(), send())
    async with run_store() as db:
        assert await auto_claim(db, target, 1, get_settings()) == 2
        await db.commit()
        assert (await db.get(SessionRunOutcome, (target, 1))).status == "completed"


async def test_postgres_post_and_notice_share_the_capacity_lock(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from sqlalchemy import func

    from agent.models import SessionInbox
    from agent.session_inbox import enqueue, promote_notices

    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    sender, child, target = await inbox_sessions(run_store, tmp_path)
    async with run_store() as db:
        await db.execute(
            update(Session).where(Session.id == child).values(parent_id=target, notify_generation=1)
        )
        await finish_run(db, child, 1, "completed")
        for index in range(19):
            db.add(
                SessionInbox(
                    target_session_id=target,
                    sender_session_id=sender,
                    kind="message",
                    content="seed",
                    hop=0,
                    dedupe_key=f"seed:{index}",
                )
            )
        await db.commit()
        fence = await claim(db, sender, lease_seconds=60)
    entered = asyncio.Event()
    pids: dict[str, int] = {}

    class PromotionSession(AsyncSession):
        async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            if "FROM sessions" in str(statement) and "FOR UPDATE" in str(statement):
                pids["promotion"] = (
                    await super().execute(text("SELECT pg_backend_pid()"))
                ).scalar_one()
                entered.set()
            return await super().execute(statement, *args, **kwargs)

    promotion_factory = async_sessionmaker(
        run_store.kw["bind"], class_=PromotionSession, expire_on_commit=False
    )
    token = CURRENT_FENCE.set(fence)
    task: asyncio.Task[None] | None = None
    try:
        async with run_store() as db:
            pids["post"] = await db.scalar(text("SELECT pg_backend_pid()"))
            await enqueue(db, sender, target, "last slot", "last")
            empty_fence = CURRENT_FENCE.set(None)
            try:
                task = asyncio.create_task(promote_notices(promotion_factory))
            finally:
                CURRENT_FENCE.reset(empty_fence)
            await asyncio.wait_for(entered.wait(), 10)
            await wait_for_blocker(run_store, pids["promotion"], pids["post"], task)
            await db.commit()
            await asyncio.wait_for(task, 10)
    finally:
        CURRENT_FENCE.reset(token)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    async with run_store() as db:
        assert await db.scalar(select(func.count()).select_from(SessionInbox)) == 20
        assert (
            await db.scalar(
                select(SessionInbox.id).where(SessionInbox.dedupe_key == f"child:{child}:1")
            )
            is None
        )
        assert (await db.get(SessionRunOutcome, (child, 1))).notice_target_id == target


async def test_inbox_migration_roundtrip_uses_native_environment(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    from sqlalchemy import inspect

    session_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        await finish_run(db, session_id, 1, "completed")
        await db.commit()
    engine = run_store.kw["bind"]

    def migrate(connection: Connection, action: str, revision: str) -> None:
        config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
        config.attributes["connection"] = connection
        getattr(command, action)(config, revision)

    async with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            await connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            await connection.commit()
        await connection.run_sync(migrate, "downgrade", "0015")
        await connection.commit()
        checks = await connection.run_sync(
            lambda sync: inspect(sync).get_check_constraints("session_run_outcomes")
        )
        assert "ck_session_run_outcomes_status" in {row["name"] for row in checks}
        assert "ck_session_run_outcomes_notice" not in {row["name"] for row in checks}
        assert await connection.scalar(text("SELECT count(*) FROM sessions WHERE title='run'")) == 1
        assert (
            await connection.scalar(text("SELECT status FROM session_run_outcomes")) == "completed"
        )
        await connection.commit()
        await connection.run_sync(migrate, "upgrade", "0016")
        await connection.commit()
        # Restore the runtime schema before reading it through the current ORM.
        await connection.run_sync(migrate, "upgrade", "0018")
        await connection.commit()
        if engine.dialect.name == "sqlite":
            assert (await connection.execute(text("PRAGMA foreign_key_check"))).all() == []
            await connection.commit()
            await connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            await connection.commit()
    async with run_store() as db:
        row = await db.get(Session, session_id)
        assert row.inbox_hop == 0 and row.notify_generation is None
        assert (await db.get(SessionRunOutcome, (session_id, 1))).notice_target_id is None


@pytest.mark.parametrize("write", ["bulk", "flush", "autoflush", "claim", "approval"])
async def test_executor_locks_session_before_first_dml(
    run_store: async_sessionmaker[AsyncSession], tmp_path: Path, write: str
) -> None:
    session_id = await seed(run_store, tmp_path)
    call_id, approval_id = await seed_call(run_store, session_id, approval=write == "approval")
    async with run_store() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    mutations: list[str] = []

    def record_dml(connection: Any, cursor: Any, sql: str, *args: Any) -> None:
        normalized = sql.lower().lstrip()
        if normalized.startswith(("insert ", "update ", "delete ")):
            mutations.append(normalized)

    engine = run_store.kw["bind"].sync_engine
    event.listen(engine, "before_cursor_execute", record_dml)
    token = CURRENT_FENCE.set(fence)
    try:
        async with run_store() as db:
            call = await db.get(ToolCall, call_id)
            if write == "claim":
                assert await AgentSupervisor(get_settings())._claim_tool_call(db, call)
            elif write in {"bulk", "approval"}:
                statement = (
                    update(Approval).where(Approval.id == approval_id).values(status="approved")
                    if write == "approval"
                    else update(ToolCall).where(ToolCall.id == call_id).values(status="running")
                )
                await db.execute(statement)
            else:
                call.status = "running"
                if write == "autoflush":
                    await db.execute(select(ToolCall))
                else:
                    await db.flush()
            assert mutations and mutations[0].startswith("update sessions set")
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
        event.remove(engine, "before_cursor_execute", record_dml)


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_approval_locks_session_before_approval(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    session_id = await seed(run_store, tmp_path)
    _, approval_id = await seed_call(run_store, session_id, approval=True)
    assert approval_id is not None
    locks: list[str] = []
    supervisor = AgentSupervisor(get_settings())

    async def no_dispatch(*args: Any, **kwargs: Any) -> None:
        pass

    def record_locks(
        connection: Any, cursor: Any, sql: str, parameters: Any, context: Any, *args: Any
    ) -> None:
        if context.compiled is not None:
            statement = str(context.compiled.statement)
            if "FOR UPDATE" in statement:
                locks.append(statement)

    monkeypatch.setattr(supervisor, "start", no_dispatch)
    engine = run_store.kw["bind"].sync_engine
    event.listen(engine, "before_cursor_execute", record_locks)
    try:
        assert await supervisor.resolve_approval(approval_id, decision, None) == session_id
    finally:
        event.remove(engine, "before_cursor_execute", record_locks)
    assert locks and "FROM sessions" in locks[0]


async def wait_for_blocker(
    factory: async_sessionmaker[AsyncSession],
    waiter_pid: int,
    blocker_pid: int,
    task: asyncio.Task[Any],
) -> None:
    async with asyncio.timeout(10), factory() as db:
        while True:
            blockers = await db.scalar(text("SELECT pg_blocking_pids(:pid)"), {"pid": waiter_pid})
            if blocker_pid in blockers:
                return
            if task.done():
                await task
                pytest.fail("The transaction completed instead of waiting for the row lock")


@pytest.mark.parametrize("noop", ["repeated", "terminal", "new_generation"])
async def test_noop_stop_locks_session_before_conditional_invalidation(
    run_store: async_sessionmaker[AsyncSession], tmp_path: Path, noop: str
) -> None:
    session_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        if noop == "repeated":
            await finish_run(db, session_id, None, "stopped", source="stop")
        elif noop == "terminal":
            await finish_run(db, session_id, 1, "completed")
        else:
            await new_generation(db, session_id)
        await db.commit()
        session = await db.get(Session, session_id)
        before = (session.run_generation, session.status, session.stop_requested)
        outcomes_before = len(list(await db.scalars(select(SessionRunOutcome))))
        events_before = len(list(await db.scalars(select(Event))))
    statements: list[str] = []

    def record_statement(connection: Any, cursor: Any, sql: str, *args: Any) -> None:
        statements.append(sql.lower().strip())

    engine = run_store.kw["bind"].sync_engine
    event.listen(engine, "before_cursor_execute", record_statement)
    try:
        async with run_store() as db:
            assert not await finish_run(
                db,
                session_id,
                None if noop == "repeated" else 1,
                "stopped",
                source="stop" if noop == "repeated" else "scoped_stop",
            )
            await db.commit()
    finally:
        event.remove(engine, "before_cursor_execute", record_statement)
    first = statements[0]
    assert "for update" in first or first.startswith("update sessions")
    predicate = first.split("where", 1)[1]
    assert "stop_requested" not in predicate
    assert "run_generation" not in predicate
    async with run_store() as db:
        session = await db.get(Session, session_id)
        assert (session.run_generation, session.status, session.stop_requested) == before
        assert len(list(await db.scalars(select(SessionRunOutcome)))) == outcomes_before
        assert len(list(await db.scalars(select(Event)))) == events_before


@pytest.mark.parametrize("existing_child", [False, True])
async def test_postgres_repeated_stop_locks_root_until_cascade_commit(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_child: bool,
) -> None:
    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    parent_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        parent = await db.get(Session, parent_id)
        project_id = parent.project_id
        if existing_child:
            db.add(
                Session(
                    project_id=project_id,
                    parent_id=parent_id,
                    title="old branch",
                    configuration={},
                    run_generation=1,
                )
            )
        await db.commit()
    supervisor = AgentSupervisor(get_settings())
    await supervisor.stop(parent_id)
    cascade_paused = asyncio.Event()
    release_stop = asyncio.Event()
    start_entered = asyncio.Event()
    pids: dict[str, int] = {}
    child_id = uuid.uuid4().hex

    class PausedCascadeSession(AsyncSession):
        async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            if "stop" not in pids:
                pids["stop"] = (await super().execute(text("SELECT pg_backend_pid()"))).scalar_one()
            if "sessions.parent_id =" in str(statement) and not cascade_paused.is_set():
                cascade_paused.set()
                await release_stop.wait()
            return await super().execute(statement, *args, **kwargs)

    monkeypatch.setattr(
        runtime,
        "session_factory",
        async_sessionmaker(
            run_store.kw["bind"], class_=PausedCascadeSession, expire_on_commit=False
        ),
    )

    async def start_and_spawn() -> None:
        async with run_store() as db:
            pids["start"] = await db.scalar(text("SELECT pg_backend_pid()"))
            start_entered.set()
            await new_generation(db, parent_id)
            await db.execute(
                update(Session).where(Session.id == parent_id).values(status="pending")
            )
            db.add(
                Session(
                    id=child_id,
                    project_id=project_id,
                    parent_id=parent_id,
                    title="new branch",
                    configuration={},
                    run_generation=1,
                )
            )
            await db.commit()

    stop_task = asyncio.create_task(supervisor.stop(parent_id))
    start_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(cascade_paused.wait(), 10)
        start_task = asyncio.create_task(start_and_spawn())
        await asyncio.wait_for(start_entered.wait(), 10)
        await wait_for_blocker(run_store, pids["start"], pids["stop"], start_task)
        assert not stop_task.done()
        release_stop.set()
        await asyncio.wait_for(asyncio.gather(stop_task, start_task), 15)
    finally:
        release_stop.set()
        for task in (stop_task, start_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (stop_task, start_task) if task is not None),
            return_exceptions=True,
        )
    async with run_store() as db:
        parent = await db.get(Session, parent_id)
        assert (parent.status, parent.run_generation, parent.stop_requested) == (
            "pending",
            3,
            False,
        )
        child = await db.get(Session, child_id)
        assert (child.status, child.run_generation, child.stop_requested) == ("pending", 1, False)
        assert await db.get(SessionRunOutcome, (parent_id, 1)) is not None
        assert await db.get(SessionRunOutcome, (parent_id, 2)) is None
        assert await db.get(SessionRunOutcome, (parent_id, 3)) is None
        assert await db.get(SessionRunOutcome, (child_id, 1)) is None
        assert not list(await db.scalars(select(Event).where(Event.session_id == child_id)))


@pytest.mark.parametrize("operation", ["claim", "approve", "reject"])
async def test_postgres_stop_executor_lock_order(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    session_id = await seed(run_store, tmp_path)
    call_id, approval_id = await seed_call(run_store, session_id, approval=operation != "claim")
    async with run_store() as db:
        fence = await claim(db, session_id, lease_seconds=60)
    assert fence is not None
    executor_locked = asyncio.Event()
    release_executor = asyncio.Event()
    stop_entered = asyncio.Event()
    pids: dict[str, int] = {}
    original_finish = runtime.finish_run

    class PausedExecutorSession(AsyncSession):
        async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            result = await super().execute(statement, *args, **kwargs)
            sql = str(statement)
            pause = (
                sql.startswith("UPDATE tool_calls")
                if operation == "claim"
                else "FROM approvals" in sql and "FOR UPDATE" in sql
            )
            if pause and not executor_locked.is_set():
                executor_locked.set()
                await release_executor.wait()
            return result

    executor_factory = async_sessionmaker(
        run_store.kw["bind"], class_=PausedExecutorSession, expire_on_commit=False
    )
    supervisor = AgentSupervisor(get_settings())

    async def no_dispatch(*args: Any, **kwargs: Any) -> None:
        pass

    async def stop_finish(db: AsyncSession, *args: Any, **kwargs: Any) -> Any:
        pids["stop"] = await db.scalar(text("SELECT pg_backend_pid()"))
        stop_entered.set()
        return await original_finish(db, *args, **kwargs)

    async def executor() -> None:
        if operation == "claim":
            token = CURRENT_FENCE.set(fence)
            try:
                async with executor_factory() as db:
                    pids["executor"] = await db.scalar(text("SELECT pg_backend_pid()"))
                    call = await db.get(ToolCall, call_id)
                    assert await supervisor._claim_tool_call(db, call)
            finally:
                CURRENT_FENCE.reset(token)
        else:
            assert approval_id is not None
            assert await supervisor.resolve_approval(approval_id, operation, None) == session_id

    class ApprovalSession(PausedExecutorSession):
        async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            if "executor" not in pids:
                pids["executor"] = (
                    await super().execute(text("SELECT pg_backend_pid()"))
                ).scalar_one()
            return await super().execute(statement, *args, **kwargs)

    if operation != "claim":
        monkeypatch.setattr(
            runtime,
            "session_factory",
            async_sessionmaker(
                run_store.kw["bind"], class_=ApprovalSession, expire_on_commit=False
            ),
        )
    monkeypatch.setattr(supervisor, "start", no_dispatch)
    monkeypatch.setattr(runtime, "finish_run", stop_finish)
    executor_task = asyncio.create_task(executor())
    stop_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(executor_locked.wait(), 10)
        monkeypatch.setattr(runtime, "session_factory", run_store)
        stop_task = asyncio.create_task(supervisor.stop(session_id))
        await asyncio.wait_for(stop_entered.wait(), 10)
        await wait_for_blocker(run_store, pids["stop"], pids["executor"], stop_task)
        release_executor.set()
        await asyncio.wait_for(asyncio.gather(executor_task, stop_task), 15)
    finally:
        release_executor.set()
        for task in (executor_task, stop_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (executor_task, stop_task) if task is not None),
            return_exceptions=True,
        )
    async with run_store() as db:
        assert (await db.get(Session, session_id)).status == "stopped"
        assert (await db.get(SessionRunOutcome, (session_id, 1))).status == "stopped"


@pytest.mark.parametrize("terminal_child", [False, True])
async def test_postgres_cascade_locks_child_generation_until_commit(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_child: bool,
) -> None:
    if run_store.kw["bind"].dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock race")
    parent_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        parent = await db.get(Session, parent_id)
        child = Session(
            project_id=parent.project_id,
            parent_id=parent_id,
            title="child",
            configuration={},
            run_generation=1,
        )
        db.add(child)
        await db.commit()
        child_id, project_id = child.id, child.project_id
        if terminal_child:
            await finish_run(db, child_id, 1, "completed")
            await db.commit()
    children_read = asyncio.Event()
    release_stop = asyncio.Event()
    start_entered = asyncio.Event()
    pids: dict[str, int] = {}
    grandchild_id = uuid.uuid4().hex

    class PausedCascadeSession(AsyncSession):
        async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            if "stop" not in pids:
                pids["stop"] = (await super().execute(text("SELECT pg_backend_pid()"))).scalar_one()
            result = await super().execute(statement, *args, **kwargs)
            if "sessions.parent_id =" in str(statement) and not children_read.is_set():
                children_read.set()
                await release_stop.wait()
            return result

    monkeypatch.setattr(
        runtime,
        "session_factory",
        async_sessionmaker(
            run_store.kw["bind"], class_=PausedCascadeSession, expire_on_commit=False
        ),
    )

    async def start_and_spawn() -> None:
        async with run_store() as db:
            pids["start"] = await db.scalar(text("SELECT pg_backend_pid()"))
            start_entered.set()
            await new_generation(db, child_id)
            await db.execute(update(Session).where(Session.id == child_id).values(status="pending"))
            db.add(
                Session(
                    id=grandchild_id,
                    project_id=project_id,
                    parent_id=child_id,
                    title="new branch",
                    configuration={},
                    run_generation=1,
                )
            )
            await db.commit()

    stop_task = asyncio.create_task(AgentSupervisor(get_settings()).stop(parent_id))
    start_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(children_read.wait(), 10)
        start_task = asyncio.create_task(start_and_spawn())
        await asyncio.wait_for(start_entered.wait(), 10)
        await wait_for_blocker(run_store, pids["start"], pids["stop"], start_task)
        release_stop.set()
        await asyncio.wait_for(asyncio.gather(stop_task, start_task), 15)
    finally:
        release_stop.set()
        for task in (stop_task, start_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (stop_task, start_task) if task is not None),
            return_exceptions=True,
        )
    async with run_store() as db:
        assert (await db.get(Session, parent_id)).status == "stopped"
        assert (await db.get(Session, child_id)).status == "pending"
        assert (await db.get(Session, grandchild_id)).status == "pending"
        assert await db.get(SessionRunOutcome, (grandchild_id, 1)) is None


@pytest.mark.parametrize("stop_first", [False, True])
async def test_finish_stop_serialization(
    run_store: async_sessionmaker[AsyncSession], tmp_path: Path, stop_first: bool
) -> None:
    session_id = await seed(run_store, tmp_path)
    acquired = asyncio.Event()
    second_entered = asyncio.Event()
    release = asyncio.Event()
    pids: dict[str, int] = {}
    postgres = run_store.kw["bind"].dialect.name == "postgresql"

    async def first() -> None:
        async with run_store() as db:
            if postgres:
                pids["first"] = await db.scalar(text("SELECT pg_backend_pid()"))
            await finish_run(
                db,
                session_id,
                1,
                "stopped" if stop_first else "completed",
                source="scoped_stop" if stop_first else "run",
            )
            acquired.set()
            await release.wait()
            await db.commit()

    async def second() -> None:
        await acquired.wait()
        async with run_store() as db:
            if postgres:
                pids["second"] = await db.scalar(text("SELECT pg_backend_pid()"))
            second_entered.set()
            await finish_run(
                db,
                session_id,
                1 if stop_first else None,
                "completed" if stop_first else "stopped",
                source="run" if stop_first else "stop",
            )
            await db.commit()

    first_task = asyncio.create_task(first())
    await acquired.wait()
    second_task = asyncio.create_task(second())
    try:
        await asyncio.wait_for(second_entered.wait(), 10)
        if postgres:
            await wait_for_blocker(run_store, pids["second"], pids["first"], second_task)
        assert not second_task.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), 15)
    async with run_store() as db:
        outcome = (await db.scalars(select(SessionRunOutcome))).one()
        assert outcome.generation == 1
        assert outcome.status == ("stopped" if stop_first else "completed")
        session = await db.get(Session, session_id)
        assert session.status == "stopped" and session.run_generation == 2
        rows = list(await db.scalars(select(Event)))
        assert len(rows) == (1 if stop_first else 2)


async def test_stop_start_rejects_old_finish(
    run_store: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    session_id = await seed(run_store, tmp_path)
    supervisor = AgentSupervisor(get_settings())
    await supervisor.stop(session_id)
    async with run_store() as db:
        await new_generation(db, session_id)
        await db.execute(update(Session).where(Session.id == session_id).values(status="pending"))
        await db.commit()
    async with run_store() as db:
        assert await finish_run(db, session_id, 1, "error", error="old run") == []
        assert await finish_run(db, session_id, 3, "completed")
        await db.commit()
        outcomes = list(
            await db.scalars(select(SessionRunOutcome).order_by(SessionRunOutcome.generation))
        )
        assert [(row.generation, row.status) for row in outcomes] == [
            (1, "stopped"),
            (3, "completed"),
        ]


async def test_reclaim_fences_pending_owner_and_finish_transaction(
    run_store: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    from datetime import UTC, datetime, timedelta

    session_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        old = await claim(
            db, session_id, lease_seconds=60, now=datetime.now(UTC) - timedelta(minutes=2)
        )
    async with run_store() as db:
        current = await claim(db, session_id, lease_seconds=60)
    assert old is not None and current is not None and old.epoch < current.epoch
    token = CURRENT_FENCE.set(old)
    try:
        async with run_store() as db:
            with pytest.raises(FenceLostError):
                await finish_run(db, session_id, 1, "error", error="stale owner")
            await db.rollback()
    finally:
        CURRENT_FENCE.reset(token)
    token = CURRENT_FENCE.set(current)
    try:
        async with run_store() as db:
            await finish_run(db, session_id, 1, "completed")
            await db.commit()
    finally:
        CURRENT_FENCE.reset(token)
    async with run_store() as db:
        assert [row.status for row in await db.scalars(select(SessionRunOutcome))] == ["completed"]
        assert len(list(await db.scalars(select(Event)))) == 1


@pytest.mark.parametrize("child_first", [False, True])
async def test_cascade_serializes_with_child_creation(
    run_store: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_first: bool,
) -> None:
    parent_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        fence = await claim(db, parent_id, lease_seconds=60)
    assert fence is not None
    supervisor = AgentSupervisor(get_settings())
    entered = asyncio.Event()
    stop_entered = asyncio.Event()
    release = asyncio.Event()
    pids: dict[str, int] = {}
    postgres = run_store.kw["bind"].dialect.name == "postgresql"
    original_finish = runtime.finish_run

    async def stop_finish(db: AsyncSession, *args: Any, **kwargs: Any) -> Any:
        if postgres:
            pids["stop"] = await db.scalar(text("SELECT pg_backend_pid()"))
        stop_entered.set()
        return await original_finish(db, *args, **kwargs)

    monkeypatch.setattr(runtime, "finish_run", stop_finish)

    async def create_child() -> None:
        token = CURRENT_FENCE.set(fence)
        try:
            async with run_store() as db:
                if postgres:
                    pids["child"] = await db.scalar(text("SELECT pg_backend_pid()"))
                await fence_check(db, fence)
                parent = await db.get(Session, parent_id)
                child = Session(
                    project_id=parent.project_id,
                    parent_id=parent_id,
                    title="child",
                    configuration={},
                    run_generation=1,
                )
                db.add(child)
                await db.flush()
                entered.set()
                if child_first:
                    await release.wait()
                await db.commit()
        finally:
            CURRENT_FENCE.reset(token)

    if child_first:
        child_task = asyncio.create_task(create_child())
        await entered.wait()
        stop_task = asyncio.create_task(supervisor.stop(parent_id))
        try:
            await asyncio.wait_for(stop_entered.wait(), 10)
            if postgres:
                await wait_for_blocker(run_store, pids["stop"], pids["child"], stop_task)
            assert not stop_task.done()
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(child_task, stop_task), 15)
    else:
        await supervisor.stop(parent_id)
        with pytest.raises(FenceLostError):
            await create_child()
    async with run_store() as db:
        children = list(await db.scalars(select(Session).where(Session.parent_id == parent_id)))
        assert len(children) == (1 if child_first else 0)
        for child in children:
            assert child.status == "stopped"
            assert (await db.get(SessionRunOutcome, (child.id, 1))).status == "stopped"


async def test_migrated_outcome_constraints_and_cascade(
    run_store: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    session_id = await seed(run_store, tmp_path)
    async with run_store() as db:
        await finish_run(db, session_id, 1, "completed")
        await db.commit()
    for invalid in [
        SessionRunOutcome(session_id=session_id, generation=1, status="error"),
        SessionRunOutcome(session_id=session_id, generation=2, status="pending"),
        SessionRunOutcome(session_id="missing", generation=1, status="completed"),
    ]:
        async with run_store() as db:
            db.add(invalid)
            with pytest.raises(IntegrityError):
                await db.commit()
            await db.rollback()
    async with run_store() as db:
        session = await db.get(Session, session_id)
        await db.delete(session)
        await db.commit()
        assert list(await db.scalars(select(SessionRunOutcome))) == []
