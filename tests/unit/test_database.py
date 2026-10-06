from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete, func, select, text

from agent.config import PROJECT_ROOT, get_settings
from agent.database import engine, init_database, session_factory
from agent.models import Message, Project, Session


def test_tests_do_not_touch_the_developer_database() -> None:
    settings = get_settings()
    assert str(PROJECT_ROOT / "data") not in settings.database_url
    assert settings.api_token == ""


async def test_sqlite_connections_enable_wal_busy_timeout_and_foreign_keys() -> None:
    await init_database()
    async with engine.connect() as connection:
        journal_mode = (await connection.execute(text("PRAGMA journal_mode"))).scalar_one()
        busy_timeout = (await connection.execute(text("PRAGMA busy_timeout"))).scalar_one()
        foreign_keys = (await connection.execute(text("PRAGMA foreign_keys"))).scalar_one()
    assert journal_mode == "wal"
    assert busy_timeout >= 5000
    assert foreign_keys == 1


async def test_bulk_session_delete_cascades_to_messages(tmp_path: Path) -> None:
    await init_database()
    async with session_factory() as db:
        project = Project(name="Cascade", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="Cascade", configuration={})
        db.add(session)
        await db.flush()
        db.add(Message(session_id=session.id, role="user", content="hello"))
        await db.commit()

        await db.execute(delete(Session).where(Session.id == session.id))
        await db.commit()

        remaining = await db.scalar(
            select(func.count()).select_from(Message).where(Message.session_id == session.id)
        )
    assert remaining == 0
