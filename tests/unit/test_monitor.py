"""Tests for the session monitor module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent.database import configure_sqlite_connection
from agent.models import (
    Base,
    Message,
    Project,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from agent.monitor import collect, display_width, format_duration, render_table, short_path

FIXED_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

_ENGINES: list[AsyncEngine] = []
_root_counter = 0


async def _make_factory(tmp_path: Path) -> async_sessionmaker[AsyncSession]:
    """Create an isolated SQLite engine with test tables."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'monitor.db'}")
    _ENGINES.append(engine)

    @event.listens_for(engine.sync_engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        configure_sqlite_connection(dbapi_connection)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


async def _add_session(
    factory: async_sessionmaker[AsyncSession],
    *,
    title: str = "Session",
    mode: str = "dev",
    llm_profile: str = "main",
    status: str = SessionStatus.running.value,
    parent_id: str | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> str:
    """Create a project with one session and return its id."""
    global _root_counter
    _root_counter += 1
    moment = created_at or FIXED_NOW - timedelta(minutes=10)
    activity = updated_at or moment
    async with factory() as db:
        project = Project(name="project", root_path=f"/tmp/project_{_root_counter}")
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            title=title,
            mode=mode,
            llm_profile=llm_profile,
            status=status,
            parent_id=parent_id,
            created_at=moment,
            updated_at=activity,
        )
        db.add(session)
        await db.commit()
        return session.id


async def _add_message(
    factory: async_sessionmaker[AsyncSession],
    session_id: str,
    *,
    role: str,
    content: str,
    created_at: datetime | None = None,
) -> int:
    """Insert a normal message into a session and return its id."""
    async with factory() as db:
        message = Message(
            session_id=session_id,
            role=role,
            kind="normal",
            content=content,
            created_at=created_at or FIXED_NOW,
        )
        db.add(message)
        await db.commit()
        return message.id


async def _add_tool_call(
    factory: async_sessionmaker[AsyncSession],
    session_id: str,
    *,
    name: str,
    status: str = ToolCallStatus.completed.value,
    updated_at: datetime | None = None,
) -> str:
    """Insert a tool call into a session and return its id."""
    tool_id = f"call_{name}_{session_id[:6]}"
    async with factory() as db:
        call = ToolCall(
            id=tool_id,
            session_id=session_id,
            name=name,
            risk_level="low",
            status=status,
            created_at=updated_at or FIXED_NOW,
            updated_at=updated_at or FIXED_NOW,
        )
        db.add(call)
        await db.commit()
        return tool_id


@pytest.fixture(autouse=True)
async def _dispose_engines() -> None:
    """Dispose every engine created for a test run."""
    yield
    for engine in _ENGINES:
        await engine.dispose()
    _ENGINES.clear()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0с"),
        (45, "45с"),
        (59, "59с"),
        (60, "1м"),
        (719, "11м"),
        (720, "12м"),
        (3900, "1ч05"),
        (3661, "1ч01"),
    ],
)
def test_format_duration(value: int, expected: str) -> None:
    """Format durations in the compact monitor notation."""
    assert format_duration(value) == expected


async def test_empty_list_renders_notice(tmp_path: Path) -> None:
    """An empty monitor renders a notice that fits the width."""
    factory = await _make_factory(tmp_path)
    async with factory() as db:
        rows = await collect(db, now=FIXED_NOW)
    lines = render_table(rows, 60)
    assert lines == ["нет активных сессий"]
    assert all(display_width(line) <= 60 for line in lines)


async def test_collect_sorts_by_group_and_activity(tmp_path: Path) -> None:
    """Sessions sort by status group, then by newest activity."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    await _add_session(
        factory,
        title="err-old",
        status=SessionStatus.error.value,
        created_at=now - timedelta(minutes=9),
    )
    await _add_session(
        factory,
        title="run-new",
        status=SessionStatus.running.value,
        created_at=now - timedelta(minutes=3),
    )
    await _add_session(
        factory,
        title="await",
        status=SessionStatus.awaiting_confirmation.value,
        created_at=now - timedelta(minutes=5),
    )
    await _add_session(
        factory,
        title="done-old",
        status=SessionStatus.completed.value,
        created_at=now - timedelta(minutes=20),
    )
    await _add_session(
        factory,
        title="done-new",
        status=SessionStatus.completed.value,
        created_at=now - timedelta(minutes=2),
    )
    await _add_session(
        factory,
        title="run-old",
        status=SessionStatus.running.value,
        created_at=now - timedelta(minutes=8),
    )
    async with factory() as db:
        rows = await collect(db, now=now)
    assert [row.name for row in rows] == [
        "run-new",
        "run-old",
        "await",
        "err-old",
        "done-new",
        "done-old",
    ]


async def test_last_action_from_tool_call(tmp_path: Path) -> None:
    """A newer tool call wins over an older message as the last action."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    session_id = await _add_session(
        factory,
        title="tools",
        status=SessionStatus.pending.value,
        created_at=now - timedelta(minutes=10),
    )
    await _add_message(
        factory,
        session_id,
        role="user",
        content="do the work",
        created_at=now - timedelta(minutes=5),
    )
    await _add_tool_call(
        factory,
        session_id,
        name="read_file",
        updated_at=now - timedelta(minutes=2),
    )
    async with factory() as db:
        rows = await collect(db, now=now)
    assert len(rows) == 1
    assert rows[0].last_action == "tool:read_file"
    assert rows[0].idle_seconds == 2 * 60


async def test_last_action_from_message(tmp_path: Path) -> None:
    """A newer replica message wins over an older tool call as the last action."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    session_id = await _add_session(
        factory,
        title="talk",
        status=SessionStatus.completed.value,
        created_at=now - timedelta(minutes=10),
    )
    await _add_tool_call(
        factory,
        session_id,
        name="write_file",
        updated_at=now - timedelta(minutes=5),
    )
    await _add_message(
        factory,
        session_id,
        role="assistant",
        content="all done",
        created_at=now - timedelta(minutes=2),
    )
    async with factory() as db:
        rows = await collect(db, now=now, history_minutes=30)
    assert len(rows) == 1
    assert rows[0].last_action == "assistant: all done"
    assert rows[0].idle_seconds == 2 * 60


async def test_collect_excludes_stale_completed(tmp_path: Path) -> None:
    """Completed sessions older than the history window are not listed."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    await _add_session(
        factory,
        title="stale",
        status=SessionStatus.completed.value,
        created_at=now - timedelta(minutes=60),
    )
    await _add_session(
        factory,
        title="fresh",
        status=SessionStatus.running.value,
        created_at=now - timedelta(minutes=1),
    )
    async with factory() as db:
        rows = await collect(db, now=now, history_minutes=30)
    assert [row.name for row in rows] == ["fresh"]


async def test_collect_applies_limit(tmp_path: Path) -> None:
    """The limit trims the newest working sessions."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    for minutes in (1, 2, 3):
        await _add_session(
            factory,
            title=f"run-{minutes}",
            status=SessionStatus.running.value,
            created_at=now - timedelta(minutes=minutes),
        )
    async with factory() as db:
        rows = await collect(db, now=now, limit=2)
    assert [row.name for row in rows] == ["run-1", "run-2"]


@pytest.mark.parametrize("width", [60, 100, 160])
async def test_render_table_fits_width(tmp_path: Path, width: int) -> None:
    """Every rendered line stays within the requested width."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    session_id = await _add_session(
        factory,
        title="Тестовая сессия с длинным названием для проверки обрезки текста в колонке",
        status=SessionStatus.running.value,
        created_at=now - timedelta(minutes=5),
    )
    await _add_message(
        factory,
        session_id,
        role="assistant",
        content="Привет мир привет мир привет мир привет мир привет мир",
        created_at=now - timedelta(minutes=2),
    )
    child_id = await _add_session(
        factory,
        title="Дочерняя сессия中国テスト",
        status=SessionStatus.awaiting_confirmation.value,
        parent_id=session_id,
        created_at=now - timedelta(minutes=3),
    )
    await _add_tool_call(
        factory,
        child_id,
        name="read_file",
        updated_at=now - timedelta(minutes=1),
    )
    async with factory() as db:
        rows = await collect(db, now=now)
    lines = render_table(rows, width)
    assert len(lines) >= 3  # header plus two rows
    assert all(display_width(line) <= width for line in lines)


async def test_render_table_columns_match_header(tmp_path: Path) -> None:
    """Every data row has the same number of separators as the header."""
    factory = await _make_factory(tmp_path)
    now = FIXED_NOW
    session_id = await _add_session(
        factory,
        title="Main",
        status=SessionStatus.running.value,
        created_at=now - timedelta(minutes=4),
    )
    await _add_message(
        factory,
        session_id,
        role="assistant",
        content="hello",
        created_at=now - timedelta(minutes=1),
    )
    async with factory() as db:
        rows = await collect(db, now=now)
    lines = render_table(rows, 100)
    separator_count = lines[0].count("│")
    assert separator_count == 9  # ten columns, including project and path
    for line in lines[1:]:
        assert line.count("│") == separator_count


async def test_rows_show_project_name_and_short_path(tmp_path: Path) -> None:
    factory = await _make_factory(tmp_path)
    await _add_session(
        factory,
        title="Main",
        status=SessionStatus.running.value,
        created_at=FIXED_NOW - timedelta(minutes=1),
    )
    async with factory() as db:
        rows = await collect(db, now=FIXED_NOW)
    assert rows and rows[0].project
    assert rows[0].path
    lines = render_table(rows, 160)
    assert "ПРОЕКТ" in lines[0] and "ПУТЬ" in lines[0]
    assert rows[0].project[:6] in lines[1]


def test_short_path_replaces_the_home_directory() -> None:
    assert short_path("/home/user/work/app", home="/home/user") == "~/work/app"
    assert short_path("/home/user", home="/home/user") == "~"
    assert short_path("/home/username/app", home="/home/user") == "/home/username/app"
    assert short_path("/srv/app", home="/home/user") == "/srv/app"
