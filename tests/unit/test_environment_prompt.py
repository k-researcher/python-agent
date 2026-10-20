from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import Message, Project
from agent.session_service import create_session_record


async def test_system_prompt_contains_creation_time_and_os(tmp_path: Path) -> None:
    await init_database(recover_running=False)
    (tmp_path / "dev.md").write_text("System prompt", encoding="utf-8")
    config = Settings(prompts_dir=tmp_path)
    created_at = datetime(2024, 5, 1, 12, 30, 45, tzinfo=UTC)

    async with session_factory() as db:
        project = Project(name="env", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = await create_session_record(
            db, config, project, "Task", "dev", created_at=created_at
        )
        rows = list(
            await db.scalars(
                select(Message).where(Message.session_id == session.id).order_by(Message.id)
            )
        )

    content = next(row.content or "" for row in rows if row.role == "system")
    assert "Корневая директория проекта" in content
    # ISO 8601 with the server timezone offset, seconds precision.
    assert "2024-05-01T12:30:45+00:00" in content
    assert f"Семейство ОС сервера: {platform.system()}" in content


async def test_system_prompt_uses_server_timezone_by_default(tmp_path: Path) -> None:
    """Without an override the timestamp carries the local timezone offset."""
    await init_database(recover_running=False)
    (tmp_path / "dev.md").write_text("System prompt", encoding="utf-8")
    config = Settings(prompts_dir=tmp_path)

    async with session_factory() as db:
        project = Project(name="env-now", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = await create_session_record(db, config, project, "Task", "dev")
        rows = list(
            await db.scalars(
                select(Message).where(Message.session_id == session.id).order_by(Message.id)
            )
        )

    content = next(row.content or "" for row in rows if row.role == "system")
    assert "Корневая директория проекта" in content
    assert "Дата и время создания сессии:" in content
