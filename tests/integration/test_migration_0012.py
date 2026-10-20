"""Миграция 0012: nullable источник ревизии вызова инструмента."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.models import Base
from tests.integration.test_migration_0010 import _alembic, _insert_project_and_session

SEED_TS = "2026-10-13 00:00:00"
REVISION_64 = "a" * 64


def _insert_tool_call(
    db: sqlite3.Connection,
    tool_call_id: str,
    *,
    revision: str | None = None,
) -> None:
    columns = "id, session_id, name, arguments, result, risk_level, status, created_at, updated_at"
    values = (
        tool_call_id,
        "s1",
        "server:tool",
        "{}",
        None,
        "network_access",
        "pending",
        SEED_TS,
        SEED_TS,
    )
    if revision is None:
        db.execute(f"INSERT INTO tool_calls ({columns}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
    else:
        db.execute(
            f"INSERT INTO tool_calls ({columns}, source_revision)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*values, revision),
        )


def _column(db: sqlite3.Connection, name: str) -> tuple[object, ...]:
    return next(row for row in db.execute("PRAGMA table_info(tool_calls)") if row[1] == name)


def test_migration_0012_legacy_calls_keep_null_and_schema_matches(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0011")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        _insert_tool_call(db, "tc-legacy")
    _alembic(database, "upgrade", "0012")
    with sqlite3.connect(database) as db:
        column = _column(db, "source_revision")
        assert column[2:5] == ("VARCHAR(64)", 0, None)
        model_column = Base.metadata.tables["tool_calls"].c["source_revision"]
        assert column[2] == str(model_column.type)
        assert column[3] == (0 if model_column.nullable else 1)
        assert db.execute(
            "SELECT source_revision FROM tool_calls WHERE id = 'tc-legacy'"
        ).fetchone() == (None,)
        _insert_tool_call(db, "tc-new", revision=REVISION_64)
        assert db.execute(
            "SELECT source_revision FROM tool_calls WHERE id = 'tc-new'"
        ).fetchone() == (REVISION_64,)


def test_migration_0012_downgrade_preserves_calls(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0011")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        _insert_tool_call(db, "tc-legacy")
    _alembic(database, "upgrade", "0012")
    with sqlite3.connect(database) as db:
        _insert_tool_call(db, "tc-rev", revision=REVISION_64)
        assert db.execute(
            "SELECT source_revision FROM tool_calls WHERE id = 'tc-rev'"
        ).fetchone() == (REVISION_64,)
    _alembic(database, "downgrade", "0011")
    with sqlite3.connect(database) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(tool_calls)")}
        assert "source_revision" not in columns
        rows = db.execute("SELECT id, name FROM tool_calls ORDER BY id").fetchall()
        assert rows == [("tc-legacy", "server:tool"), ("tc-rev", "server:tool")]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_0012_head_from_empty_database(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "head")
    with sqlite3.connect(database) as db:
        column = _column(db, "source_revision")
        assert column[2:5] == ("VARCHAR(64)", 0, None)
        assert db.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (0,)
