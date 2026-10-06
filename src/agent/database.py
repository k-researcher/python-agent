from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent.config import PROJECT_ROOT, get_settings
from agent.models import Session, SessionStatus

settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
session_factory = async_sessionmaker(engine, expire_on_commit=False)

SQLITE_BUSY_TIMEOUT_MS = 5000


def configure_sqlite_connection(dbapi_connection: Any, *, foreign_keys: bool = True) -> None:
    """Apply per-connection SQLite settings.

    WAL lets readers proceed during writes and busy_timeout waits instead of failing with
    "database is locked". foreign_keys makes ON DELETE CASCADE/SET NULL work; migrations
    keep it off because SQLite batch table rebuilds conflict with enforced foreign keys.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    cursor.execute(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}")
    cursor.close()


if engine.dialect.name == "sqlite":

    @event.listens_for(engine.sync_engine, "connect")
    def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
        configure_sqlite_connection(dbapi_connection)


async def init_database(
    *, run_migrations: bool | None = None, recover_running: bool | None = None
) -> None:
    should_migrate = (
        settings.execution_mode == "embedded"
        if run_migrations is None
        else run_migrations
    )
    if should_migrate:
        def migrate() -> None:
            migration_config = Config(str(PROJECT_ROOT / "alembic.ini"))
            command.upgrade(migration_config, "head")

        await asyncio.to_thread(migrate)

    should_recover = (
        settings.execution_mode == "embedded"
        if recover_running is None
        else recover_running
    )
    async with session_factory() as db:
        await db.execute(select(Session.id).limit(1))
        if not should_recover:
            return
        await db.execute(
            update(Session)
            .where(Session.status == SessionStatus.running.value)
            .values(status=SessionStatus.interrupted.value, stop_requested=False)
        )
        await db.commit()


async def get_db() -> AsyncIterator[AsyncSession]:
    async with session_factory() as db:
        yield db
