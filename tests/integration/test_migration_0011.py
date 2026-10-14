"""Схема дерева и перенос прежних сообщений без изменения живых баз."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.test_migration_0010 import ROOT, _alembic, _insert_project_and_session

PARENTS = [
    (1, None),
    (2, 1),
    (3, None),
    (4, 2),
    (5, 3),
    (6, None),
    (7, None),
    (8, None),
    (9, 4),
    (10, 5),
    (11, 9),
    (12, None),
    (13, None),
]


def _seed_legacy(database: Path) -> list[tuple[object, ...]]:
    _alembic(database, "upgrade", "0010")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        db.execute(
            "INSERT INTO sessions (id, project_id, mode, title, status, archived,"
            " stop_requested, configuration, created_at, updated_at)"
            " VALUES ('s2', 'p1', 'dev', 'Second', 'completed', 0, 0, '{}', ?, ?)",
            ("2026-10-13 00:00:00", "2026-10-13 00:00:00"),
        )
        rows = [
            (1, "s1", "system", "normal", 0, None),
            (2, "s1", "user", "normal", 0, None),
            (3, "s2", "system", "normal", 0, None),
            (4, "s1", "assistant", "normal", 0, None),
            (5, "s2", "user", "normal", 0, None),
            (6, "s1", "user", "summary", 0, 4),
            (7, "s1", "user", "normal", 1, None),
            (8, "s1", "assistant", "interrupted", 0, None),
            (9, "s1", "tool", "normal", 0, None),
            (10, "s2", "assistant", "normal", 0, None),
            (11, "s1", "user", "normal", 0, None),
            (12, "s2", "user", "summary", 0, 10),
            (13, "s1", "user", "summary", 1, 9),
        ]
        for message_id, session_id, role, kind, skipped, covered in rows:
            db.execute(
                "INSERT INTO messages (id, session_id, role, content, skipped, kind,"
                " covers_until, source_hash, summary_version, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    role,
                    f"content-{message_id}",
                    skipped,
                    kind,
                    covered,
                    "a" * 64 if covered else None,
                    1 if covered else None,
                    "2026-10-13 00:00:00",
                ),
            )
        db.execute("UPDATE sessions SET run_generation=3, owner_epoch=7 WHERE id='s1'")
        db.execute(
            "INSERT INTO events (session_id, type, payload, created_at)"
            " VALUES ('s1', 'session.status', '{}', '2026-10-13 00:00:00')"
        )
        return db.execute("SELECT * FROM messages ORDER BY id").fetchall()


def _legacy_rows(db: sqlite3.Connection) -> list[tuple[object, ...]]:
    columns = [row[1] for row in db.execute("PRAGMA table_info(messages)") if row[1] != "parent_id"]
    return db.execute(f"SELECT {','.join(columns)} FROM messages ORDER BY id").fetchall()


def test_message_tree_backfill_and_roundtrip_preserve_history(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    original = _seed_legacy(database)
    for repetition in range(2):
        _alembic(database, "upgrade", "0011")
        with sqlite3.connect(database) as db:
            parent_links = db.execute("SELECT id, parent_id FROM messages ORDER BY id").fetchall()
            assert parent_links == PARENTS
            assert _legacy_rows(db) == original
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
            assert db.execute(
                "SELECT run_generation, owner_epoch FROM sessions WHERE id='s1'"
            ).fetchone() == (3, 7)
            assert db.execute("SELECT COUNT(*) FROM events").fetchone() == (1,)
        if repetition == 0:
            _alembic(database, "downgrade", "0010")
            with sqlite3.connect(database) as db:
                assert "parent_id" not in {
                    row[1] for row in db.execute("PRAGMA table_info(messages)")
                }
                assert _legacy_rows(db) == original
                assert db.execute("PRAGMA foreign_key_check").fetchall() == []
                assert "ix_messages_session_parent_id" not in {
                    row[1] for row in db.execute("PRAGMA index_list(messages)")
                }


def test_message_tree_empty_database_and_nullable_column(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "head")
    with sqlite3.connect(database) as db:
        column = next(
            row for row in db.execute("PRAGMA table_info(messages)") if row[1] == "parent_id"
        )
        assert column[2:5] == ("INTEGER", 0, None)
        assert db.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
        indexes = {row[1] for row in db.execute("PRAGMA index_list(messages)")}
        assert "ix_messages_session_parent_id" in indexes
        assert [
            row[2] for row in db.execute("PRAGMA index_info(ix_messages_session_parent_id)")
        ] == ["session_id", "parent_id"]


def test_message_tree_foreign_key_rejects_unknown_parent_and_sets_null(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _seed_legacy(database)
    _alembic(database, "upgrade", "0011")
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        parent_fk = next(
            row for row in db.execute("PRAGMA foreign_key_list(messages)") if row[3] == "parent_id"
        )
        assert (parent_fk[2], parent_fk[4], parent_fk[6]) == ("messages", "id", "SET NULL")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE messages SET parent_id=999999 WHERE id=4")
        db.rollback()
        assert db.execute("SELECT parent_id FROM messages WHERE id=4").fetchone() == (2,)
        db.execute("DELETE FROM messages WHERE id=2")
        assert db.execute("SELECT parent_id FROM messages WHERE id=4").fetchone() == (None,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_message_tree_requires_explicit_parent_for_new_messages(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _seed_legacy(database)
    _alembic(database, "upgrade", "0011")
    with sqlite3.connect(database) as db:
        db.execute(
            "INSERT INTO messages (session_id, role, content, skipped, kind, created_at)"
            " VALUES ('s1', 'user', 'New message', 0, 'normal', '2026-10-13 00:00:00')"
        )
        assert db.execute("SELECT parent_id FROM messages WHERE id=14").fetchone() == (None,)


@pytest.mark.parametrize("arguments", [("upgrade", "0010:0011"), ("downgrade", "0011:0010")])
def test_message_tree_postgresql_sql_without_live_database(arguments: tuple[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *arguments, "--sql"],
        cwd=ROOT,
        env={
            **os.environ,
            "AGENT_ENV_FILE": "",
            "AGENT_MODELS_FILE": "",
            "AGENT_DATABASE_URL": "postgresql+asyncpg://unused:unused@localhost/unused",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    if arguments[0] == "upgrade":
        assert "ADD COLUMN parent_id INTEGER" in result.stdout
        assert "FOREIGN KEY(parent_id) REFERENCES messages (id) ON DELETE SET NULL" in result.stdout
        assert "CREATE INDEX ix_messages_session_parent_id" in result.stdout
        assert "UPDATE messages SET parent_id=" in result.stdout
    else:
        assert "DROP CONSTRAINT fk_messages_parent_id_messages" in result.stdout
        assert "DROP COLUMN parent_id" in result.stdout
