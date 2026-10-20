"""Миграция 0014: nullable колонка состояния бюджета сессии."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.models import Base
from tests.integration.test_migration_0010 import _alembic, _insert_project_and_session

SEED_TS = "2026-10-19 00:00:00"


def test_migration_0014_preserves_rows_and_downgrades(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    _alembic(database, "upgrade", "0013")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        assert db.execute("SELECT id, configuration FROM sessions WHERE id='s1'").fetchone() == (
            "s1",
            "{}",
        )
    _alembic(database, "upgrade", "0014")
    with sqlite3.connect(database) as db:
        columns = {row[1]: row for row in db.execute("PRAGMA table_info(sessions)")}
        assert "budget_state" in columns
        model_column = Base.metadata.tables["sessions"].c["budget_state"]
        assert columns["budget_state"][2] == str(model_column.type)
        assert columns["budget_state"][3] == (0 if model_column.nullable else 1)
        assert db.execute("SELECT budget_state FROM sessions WHERE id='s1'").fetchone() == (None,)
        db.execute(
            "UPDATE sessions SET budget_state = "
            '\'{"steps": 3, "started_at": "2026-10-19 00:00:00", '
            '"limits": {"max_steps": 10}}\' WHERE id=\'s1\''
        )
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    _alembic(database, "downgrade", "0013")
    with sqlite3.connect(database) as db:
        columns_after = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
        assert "budget_state" not in columns_after
        assert db.execute("SELECT id, configuration FROM sessions WHERE id='s1'").fetchone() == (
            "s1",
            "{}",
        )
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_0014_head_from_empty_database(tmp_path: Path) -> None:
    database = tmp_path / "empty.db"
    _alembic(database, "upgrade", "head")
    with sqlite3.connect(database) as db:
        column = next(
            row for row in db.execute("PRAGMA table_info(sessions)") if row[1] == "budget_state"
        )
        assert column[2:5] == ("JSON", 0, None)
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
