"""Tests for the ``pyagent monitor`` view."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from agent.cli_client.monitor_view import (
    CLEAR_SCREEN,
    DB_ERROR,
    EXIT_NO_DATABASE,
    EXIT_OK,
    LIVE_HINT,
    MIN_REFRESH_S,
    missing_sqlite_file,
    run_monitor,
)
from agent.monitor import SessionRow


def _row(name: str = "Задача") -> SessionRow:
    return SessionRow(
        session_id="s1",
        role="main",
        name=name,
        model="main",
        mode="dev",
        status="running",
        duration_seconds=65,
        idle_seconds=3,
        last_action="shell: pytest",
        project="demo",
        path="~/demo",
    )


class _Stop(Exception):
    """Ends the refresh loop in a test."""


def _monitor(out: io.StringIO, **kwargs: object) -> int:
    async def source() -> list[SessionRow]:
        return [_row()]

    params: dict[str, object] = {
        "source": source,
        "width": lambda: 120,
        "clock": lambda: "10:00:00",
    }
    params.update(kwargs)
    return asyncio.run(run_monitor(out, **params))  # type: ignore[arg-type]


def test_once_prints_one_table_without_screen_control() -> None:
    out = io.StringIO()
    assert _monitor(out, once=True, live=True) == EXIT_OK
    text = out.getvalue()
    assert CLEAR_SCREEN not in text
    assert LIVE_HINT not in text
    assert text.splitlines()[0] == "Сессии агента · 10:00:00"
    assert "Задача" in text and "demo" in text


def test_pipe_output_prints_once_even_without_flag() -> None:
    out = io.StringIO()
    assert _monitor(out, live=False) == EXIT_OK
    assert out.getvalue().count("Сессии агента") == 1


def test_live_mode_refreshes_with_clear_screen_and_hint() -> None:
    out = io.StringIO()
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 2:
            raise _Stop

    with pytest.raises(_Stop):
        _monitor(out, live=True, interval=0.1, sleep=sleep)
    text = out.getvalue()
    assert text.count(CLEAR_SCREEN) == 2
    assert LIVE_HINT in text
    # The refresh period never goes below the minimum.
    assert delays == [MIN_REFRESH_S, MIN_REFRESH_S]


def test_every_line_fits_the_terminal_width() -> None:
    out = io.StringIO()
    _monitor(out, once=True, live=False, width=lambda: 60)
    assert all(len(line) <= 60 for line in out.getvalue().splitlines()[1:])


def test_database_error_is_short_and_returns_code_two() -> None:
    async def broken() -> list[SessionRow]:
        raise OperationalError("SELECT * FROM sessions", {}, Exception("no such table: sessions"))

    out = io.StringIO()
    assert _monitor(out, once=True, source=broken) == EXIT_NO_DATABASE
    assert out.getvalue() == DB_ERROR.format(error="no such table: sessions") + "\n"


def test_missing_sqlite_file_is_detected_without_creating_it(tmp_path: Path) -> None:
    missing = tmp_path / "none.db"
    assert missing_sqlite_file(f"sqlite+aiosqlite:///{missing}") == missing
    assert not missing.exists()
    present = tmp_path / "a.db"
    present.touch()
    assert missing_sqlite_file(f"sqlite+aiosqlite:///{present}") is None
    assert missing_sqlite_file("sqlite+aiosqlite:///:memory:") is None
    assert missing_sqlite_file("postgresql+asyncpg://user@localhost/agent") is None


def test_monitor_help_is_russian(capsys: pytest.CaptureFixture[str]) -> None:
    from agent.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["monitor", "--help"])
    assert stopped.value.code == 0
    text = capsys.readouterr().out
    assert "показать один раз и выйти" in text
    assert "pyagent monitor" in text
