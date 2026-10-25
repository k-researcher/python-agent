from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.integration.test_migration_0010 import _alembic, _insert_project_and_session


def test_run_outcomes_schema_constraints_and_downgrade(tmp_path: Path) -> None:
    database = tmp_path / "outcomes.db"
    _alembic(database, "upgrade", "0014")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
    _alembic(database, "upgrade", "0015")
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        assert db.execute("SELECT COUNT(*) FROM session_run_outcomes").fetchone() == (0,)
        insert = (
            "INSERT INTO session_run_outcomes (session_id,generation,status,created_at) "
            "VALUES (?, ?, ?, '2026-10-24 00:00:00')"
        )
        db.execute(insert, ("s1", 0, "completed"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, ("s1", 0, "error"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, ("s1", 1, "pending"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, ("missing", 1, "completed"))
        db.execute("DELETE FROM sessions WHERE id='s1'")
        assert db.execute("SELECT COUNT(*) FROM session_run_outcomes").fetchone() == (0,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    _alembic(database, "downgrade", "0014")
    with sqlite3.connect(database) as db:
        assert (
            db.execute(
                "SELECT name FROM sqlite_master WHERE name='session_run_outcomes'"
            ).fetchall()
            == []
        )
