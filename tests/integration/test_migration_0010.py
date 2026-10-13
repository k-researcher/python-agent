"""Миграция 0010: защита сессии от устаревшего исполнителя."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _alembic(database: Path, *args: str) -> None:
    env = {
        **os.environ,
        "AGENT_ENV_FILE": "",
        "AGENT_MODELS_FILE": "",
        "AGENT_DATABASE_URL": f"sqlite+aiosqlite:///{database}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def _insert_project_and_session(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO projects (id, name, root_path, workspace_id, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            "p1",
            "Old project",
            "/tmp/old",
            "00000000-0000-0000-0000-000000000002",
            "2026-10-06 00:00:00",
        ),
    )
    db.execute(
        "INSERT INTO sessions (id, project_id, mode, title, status, archived,"
        " stop_requested, configuration, created_at, updated_at, created_by_user_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "s1",
            "p1",
            "dev",
            "Old session",
            "completed",
            0,
            0,
            "{}",
            "2026-10-06 00:00:00",
            "2026-10-06 00:00:00",
            "00000000-0000-0000-0000-000000000001",
        ),
    )


def test_migration_0010_keeps_rows_and_adds_defaults(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        db.execute(
            "INSERT INTO events (session_id, type, payload, created_at) VALUES (?, ?, ?, ?)",
            ("s1", "session.status", "{}", "2026-10-06 00:00:00"),
        )

    _alembic(database, "upgrade", "0010")
    with sqlite3.connect(database) as db:
        columns = db.execute("PRAGMA table_info(sessions)").fetchall()
        column_map = {col[1]: col for col in columns}
        assert column_map["run_generation"][2] == "INTEGER"
        assert column_map["run_generation"][3] == 1
        assert column_map["run_generation"][4].strip("'\"") == "0"
        assert column_map["owner_epoch"][2] == "INTEGER"
        assert column_map["owner_epoch"][3] == 1
        assert column_map["owner_epoch"][4].strip("'\"") == "0"
        assert column_map["owner_token"][2] == "VARCHAR(36)"
        assert column_map["owner_token"][3] == 0
        assert column_map["owner_token"][4] is None
        assert column_map["lease_until"][2] == "DATETIME"
        assert column_map["lease_until"][3] == 0
        assert column_map["lease_until"][4] is None

        row = db.execute(
            "SELECT run_generation, owner_epoch, owner_token, lease_until"
            " FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert row == (0, 0, None, None)

        event_rows = db.execute(
            "SELECT session_id, type, payload FROM events ORDER BY rowid"
        ).fetchall()
        assert event_rows == [("s1", "session.status", "{}")]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

        db.execute(
            "INSERT INTO sessions (id, project_id, mode, title, status, archived,"
            " stop_requested, configuration, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s2",
                "p1",
                "dev",
                "New session",
                "pending",
                0,
                0,
                "{}",
                "2026-10-06 00:00:00",
                "2026-10-06 00:00:00",
            ),
        )
        new_row = db.execute(
            "SELECT run_generation, owner_epoch, owner_token, lease_until"
            " FROM sessions WHERE id = 's2'"
        ).fetchone()
        assert new_row == (0, 0, None, None)

    _alembic(database, "downgrade", "0009")
    with sqlite3.connect(database) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
        assert "run_generation" not in columns
        assert "owner_epoch" not in columns
        assert "owner_token" not in columns
        assert "lease_until" not in columns

        session_rows = db.execute(
            "SELECT id, project_id, title FROM sessions ORDER BY id"
        ).fetchall()
        assert session_rows == [("s1", "p1", "Old session"), ("s2", "p1", "New session")]

        project_rows = db.execute("SELECT id, name FROM projects ORDER BY id").fetchall()
        assert project_rows == [("p1", "Old project")]

        event_rows = db.execute(
            "SELECT session_id, type, payload FROM events ORDER BY rowid"
        ).fetchall()
        assert event_rows == [("s1", "session.status", "{}")]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

    _alembic(database, "upgrade", "0010")
    with sqlite3.connect(database) as db:
        row = db.execute(
            "SELECT run_generation, owner_epoch, owner_token, lease_until"
            " FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert row == (0, 0, None, None)
        event_rows = db.execute(
            "SELECT session_id, type, payload FROM events ORDER BY rowid"
        ).fetchall()
        assert event_rows == [("s1", "session.status", "{}")]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_0010_upgrade_head_from_empty_database(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "head")
    with sqlite3.connect(database) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
        assert "run_generation" in columns
        assert "owner_epoch" in columns
        assert "owner_token" in columns
        assert "lease_until" in columns
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
