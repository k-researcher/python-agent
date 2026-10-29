"""Cancellation must release database resources without waiting for garbage collection."""

from __future__ import annotations

import asyncio
import sqlite3
import traceback
from contextlib import closing
from pathlib import Path

import aiosqlite
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent.config import Settings
from agent.runtime import AgentSupervisor


@pytest.mark.parametrize("boundary", ["shutdown", "stop"])
async def test_cancelled_run_releases_sqlite_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    async with asyncio.timeout(5):
        path = tmp_path / "cancel.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        factory = async_sessionmaker(engine)
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE sample (value INTEGER)"))
            await connection.execute(text("INSERT INTO sample VALUES (0)"))
        entered = asyncio.Event()
        original = aiosqlite.Cursor.execute

        async def execute(self, sql, parameters=None):
            result = await original(self, sql, parameters)
            if "cancel_barrier" in sql:
                entered.set()
                await asyncio.Event().wait()
            return result

        monkeypatch.setattr(aiosqlite.Cursor, "execute", execute)

        async def work():
            async with factory() as db:
                await db.execute(text("UPDATE sample SET value = 1"))
                await db.execute(text("SELECT value FROM sample /* cancel_barrier */"))

        supervisor = AgentSupervisor(Settings())
        task = asyncio.create_task(work())
        supervisor._tasks["sample"] = task
        try:
            await entered.wait()
            if boundary == "shutdown":
                await supervisor.shutdown()
            else:
                task.cancel()
                await supervisor._settle_stopped({task}, [])
            assert task.cancelled()
            assert engine.pool.checkedout() == 0
            with closing(sqlite3.connect(path, timeout=0)) as probe:
                probe.execute("BEGIN IMMEDIATE")
                assert probe.execute("SELECT value FROM sample").fetchone() == (0,)
                probe.rollback()
        finally:
            # Also release the traceback on the unfixed implementation.
            try:
                task.result()
            except asyncio.CancelledError as error:
                traceback.clear_frames(error.__traceback__)
            await engine.dispose()


def test_database_backup_reports_a_locked_target(tmp_path: Path) -> None:
    from tests.conftest import backup_database

    source, target = tmp_path / "source.db", tmp_path / "target.db"
    with (
        closing(sqlite3.connect(source)) as snapshot,
        closing(sqlite3.connect(target)) as locked,
    ):
        snapshot.execute("CREATE TABLE sample (value INTEGER)")
        locked.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="previous test.*connection or transaction"):
            backup_database(source, target, timeout=0)
        locked.rollback()
    backup_database(source, target)


async def test_shutdown_releases_cancelled_sql_before_releasing_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import update

    from agent import runtime
    from agent.database import engine, session_factory
    from agent.models import Project, Session

    async with asyncio.timeout(8):
        async with session_factory() as db:
            project = Project(name="cancel", root_path=str(tmp_path))
            db.add(project)
            await db.flush()
            session = Session(project_id=project.id, title="original", status="running")
            db.add(session)
            await db.commit()
            session_id = session.id

        entered = asyncio.Event()
        original = aiosqlite.Cursor.execute
        cancellations = []

        async def execute(self, sql, parameters=None):
            result = await original(self, sql, parameters)
            if "lease_cancel_barrier" in sql:
                entered.set()
                await asyncio.Event().wait()
            return result

        monkeypatch.setattr(aiosqlite.Cursor, "execute", execute)
        supervisor = AgentSupervisor(Settings())
        released = asyncio.Event()
        original_release = runtime.release

        async def release(db, fence):
            # Fail immediately on RED instead of entering the driver's busy wait
            # and a second cancellation during event-loop teardown.
            with closing(sqlite3.connect(str(engine.url.database), timeout=0)) as probe:
                probe.execute("BEGIN IMMEDIATE")
                probe.rollback()
            await original_release(db, fence)
            released.set()

        monkeypatch.setattr(runtime, "release", release)

        async def work(_session_id):
            try:
                async with session_factory() as db:
                    await db.execute(
                        update(Session).where(Session.id == session_id).values(title="uncommitted")
                    )
                    await db.execute(text("SELECT id FROM sessions /* lease_cancel_barrier */"))
            except asyncio.CancelledError as error:
                cancellations.append(error)
                raise

        monkeypatch.setattr(supervisor, "_run", work)
        assert await supervisor.start(session_id, new_run=False)
        task = supervisor._tasks[session_id]
        try:
            await entered.wait()
            await supervisor.shutdown()
            assert task.done(), "Lease cleanup is blocked by the cancelled run's cursor"
            assert released.is_set(), "The cancelled SQL cursor still holds the write lock"
            async with session_factory() as db:
                fresh = await db.get(Session, session_id)
                assert fresh.title == "original"
                assert fresh.owner_token is None
                assert fresh.lease_until is None
        finally:
            # Make RED safe too: the original run can now finish lease cleanup.
            for error in cancellations:
                traceback.clear_frames(error.__traceback__)
            await asyncio.wait({task}, timeout=1)
            assert task.done()

        async def later_run(_session_id):
            async with session_factory() as db:
                await db.execute(
                    update(Session).where(Session.id == session_id).values(title="next run")
                )
                await db.commit()

        monkeypatch.setattr(supervisor, "_run", later_run)
        assert await supervisor.start(session_id, new_run=False)
        await supervisor._tasks[session_id]
        async with session_factory() as db:
            fresh = await db.get(Session, session_id)
            assert fresh.title == "next run"
            assert fresh.owner_token is None
