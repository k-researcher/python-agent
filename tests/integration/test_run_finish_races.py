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
