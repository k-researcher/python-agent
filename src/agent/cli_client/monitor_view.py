"""``pyagent monitor``: a live table of agent sessions from the local database.

The view reads the same database as the server (``AGENT_DATABASE_URL`` or the
default in the data directory). It only reads: no migrations, no writes.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from agent.monitor import SessionRow, collect, render_table

REFRESH_S = 2.0
MIN_REFRESH_S = 0.5
CLEAR_SCREEN = "\x1b[H\x1b[2J"
TITLE = "Сессии агента · {time}"
LIVE_HINT = " · Ctrl+C — выход"
DB_ERROR = "Не удалось прочитать базу агента: {error}"
DB_MISSING = "База агента не найдена: {path}. Сначала запустите агента: pyagent"
EXIT_OK = 0
EXIT_NO_DATABASE = 2

RowSource = Callable[[], Awaitable[list[SessionRow]]]


async def collect_from_database() -> list[SessionRow]:
    """Read the monitor rows from the configured agent database."""
    from agent.database import session_factory

    async with session_factory() as db:
        return await collect(db)


def missing_sqlite_file(database_url: str) -> Path | None:
    """Return the path of a SQLite database file that does not exist, else None.

    A connection to a missing file would create an empty database, so the
    monitor checks the file first.
    """
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or url.database in (None, "", ":memory:"):
        return None
    path = Path(url.database).expanduser()
    return None if path.exists() else path


async def _dispose_database() -> None:
    from agent.database import engine

    await engine.dispose()


def _terminal_width() -> int:
    return shutil.get_terminal_size().columns


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


async def run_monitor(
    out: TextIO,
    *,
    once: bool = False,
    interval: float = REFRESH_S,
    live: bool | None = None,
    source: RowSource | None = None,
    width: Callable[[], int] = _terminal_width,
    clock: Callable[[], str] = _clock,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> int:
    """Show the session table; refresh it until Ctrl+C when the output is a terminal.

    Without a terminal (for example in a pipe) or with ``once`` the table is shown
    one time. Returns 0, or 2 when the database cannot be read.
    """
    # Refresh in place only on a terminal and without --once.
    live = (out.isatty() if live is None else live) and not once
    if source is None:
        from agent.config import get_settings

        missing = missing_sqlite_file(get_settings().database_url)
        if missing is not None:
            out.write(DB_MISSING.format(path=missing) + "\n")
            return EXIT_NO_DATABASE
    rows_source = source or collect_from_database
    try:
        while True:
            try:
                rows = await rows_source()
            except SQLAlchemyError as exc:
                # The original driver message is short ("no such table: ...");
                # the SQLAlchemy text adds the SQL statement and a link.
                out.write(DB_ERROR.format(error=getattr(exc, "orig", None) or exc) + "\n")
                return EXIT_NO_DATABASE
            title = TITLE.format(time=clock()) + (LIVE_HINT if live else "")
            lines = [title, *render_table(rows, width())]
            out.write((CLEAR_SCREEN if live else "") + "\n".join(lines) + "\n")
            out.flush()
            if not live:
                return EXIT_OK
            await sleep(max(interval, MIN_REFRESH_S))
    finally:
        if source is None:
            await _dispose_database()
