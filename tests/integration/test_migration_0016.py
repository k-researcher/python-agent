from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.integration.test_migration_0010 import _alembic, _insert_project_and_session


def test_inbox_upgrade_downgrade_preserves_existing_constraints(tmp_path: Path) -> None:
    database = tmp_path / "inbox.db"
    _alembic(database, "upgrade", "0015")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        db.execute(
            "INSERT INTO session_run_outcomes (session_id,generation,status,created_at) "
            "VALUES ('s1',0,'completed','2026-10-27')"
        )
        db.execute(
            "INSERT INTO messages (session_id,role,content,kind,skipped,created_at) "
            "VALUES ('s1','assistant','Old answer','normal',0,'2026-10-27')"
        )
    _alembic(database, "upgrade", "0016")
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        assert db.execute("SELECT inbox_hop,notify_generation FROM sessions").fetchone() == (
            0,
            None,
        )
        assert db.execute(
            "SELECT notice_target_id,notice_hop,notice_content,notice_message_id,notice_skipped_at "
            "FROM session_run_outcomes"
        ).fetchone() == (None, None, None, None, None)
        assert "run_generation" in {row[1] for row in db.execute("PRAGMA table_info(messages)")}
        assert db.execute("SELECT content,run_generation FROM messages").fetchone() == (
            "Old answer",
            None,
        )
        for statement in (
            "UPDATE sessions SET inbox_hop=-1",
            "UPDATE sessions SET notify_generation=1",
            "UPDATE session_run_outcomes SET notice_hop=0",
            "UPDATE session_run_outcomes SET notice_message_id=1",
            "UPDATE session_run_outcomes SET notice_skipped_at='2026-10-28'",
            "UPDATE session_run_outcomes SET notice_target_id='s1',notice_content='x'",
            "UPDATE session_run_outcomes SET notice_target_id='s1',"
            "notice_hop=-1,notice_content='x'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(statement)
        insert = (
            "INSERT INTO session_inbox "
            "(id,target_session_id,sender_session_id,kind,content,hop,dedupe_key,created_at) "
            "VALUES (?, 's1', 's1', ?, ?, ?, ?, '2026-10-27')"
        )
        for kind, content, hop in (
            ("other", "x", 0),
            ("message", "", 0),
            ("message", "я" * 8001, 0),
            ("message", "x", -1),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(insert, ("bad", kind, content, hop, "bad"))
        db.execute(insert, ("one", "message", "я" * 8000, 0, "key"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, ("two", "message", "x", 0, "key"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "UPDATE session_inbox SET delivered_at='2026-10-27',rejected_at='2026-10-27'"
            )
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    _alembic(database, "downgrade", "0015")
    with sqlite3.connect(database) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        for name, constraints in (
            ("session_run_outcomes", ["ck_session_run_outcomes_status"]),
            ("messages", ["ck_messages_summary_fields", "uq_messages_summary_checkpoint"]),
        ):
            sql = db.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
            assert all(constraint in sql for constraint in constraints)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE session_run_outcomes SET status='pending'")
        assert db.execute("SELECT generation,status FROM session_run_outcomes").fetchone() == (
            0,
            "completed",
        )
        assert db.execute("SELECT title FROM sessions WHERE id='s1'").fetchone() == ("Old session",)
        assert db.execute("SELECT content FROM messages").fetchone() == ("Old answer",)
        for table, removed in (
            ("messages", {"run_generation"}),
            ("session_run_outcomes", {"notice_message_id", "notice_skipped_at"}),
        ):
            assert not removed & {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    _alembic(database, "upgrade", "0016")
