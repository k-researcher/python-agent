"""Миграция 0009: пользователи и командные пространства."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from agent.model_overrides import LLMProviderOverride
from agent.models import Base

ROOT = Path(__file__).resolve().parents[2]

BOOTSTRAP_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_WORKSPACE_ID = "00000000-0000-0000-0000-000000000002"
SEED_TS = "2026-10-06 00:00:00"


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
        raise AssertionError(
            f"alembic {' '.join(args)} failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _insert_project(db: sqlite3.Connection, project_id: str, name: str, root_path: str) -> None:
    db.execute(
        "INSERT INTO projects (id, name, root_path, created_at) VALUES (?, ?, ?, ?)",
        (project_id, name, root_path, SEED_TS),
    )


def _insert_session(db: sqlite3.Connection, session_id: str, project_id: str, title: str) -> None:
    db.execute(
        "INSERT INTO sessions (id, project_id, mode, title, status, archived, stop_requested, "
        "configuration, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, project_id, "dev", title, "completed", 0, 0, "{}", SEED_TS, SEED_TS),
    )


def _insert_auth_session(db: sqlite3.Connection, id_hash: str) -> None:
    db.execute(
        "INSERT INTO auth_sessions (id_hash, csrf_token, user_agent, created_at, expires_at, "
        "last_seen) VALUES (?, ?, ?, ?, ?, ?)",
        (id_hash, "csrf", "", SEED_TS, "2026-11-06 00:00:00", SEED_TS),
    )


def _insert_auth_code(db: sqlite3.Connection, code_hash: str) -> None:
    db.execute(
        "INSERT INTO auth_bootstrap_codes (code_hash, created_at, expires_at) VALUES (?, ?, ?)",
        (code_hash, SEED_TS, "2026-11-06 00:00:00"),
    )


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def test_migration_0009_upgrade_creates_schema_and_seeds(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        assert db.execute(
            "SELECT id, display_name, enabled, is_operator FROM users"
        ).fetchone() == (BOOTSTRAP_USER_ID, "Owner", 1, 1)
        assert db.execute("SELECT id, name FROM workspaces").fetchone() == (
            DEFAULT_WORKSPACE_ID,
            "Основное пространство",
        )
        assert db.execute(
            "SELECT workspace_id, user_id, role, revoked_at FROM workspace_memberships"
        ).fetchone() == (DEFAULT_WORKSPACE_ID, BOOTSTRAP_USER_ID, "owner", None)
        for table, column in [
            ("projects", "workspace_id"),
            ("sessions", "created_by_user_id"),
            ("auth_sessions", "user_id"),
            ("auth_bootstrap_codes", "user_id"),
        ]:
            assert column in _columns(db, table)


def test_migration_0009_backfills_existing_records(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0008")
    with sqlite3.connect(database) as db:
        _insert_project(db, "p1", "project-one", "/root/p1")
        _insert_project(db, "p2", "project-two", "/root/p2")
        _insert_session(db, "s1", "p1", "session-one")
        _insert_auth_session(db, "abc123")
        _insert_auth_code(db, "def456")

    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT id, workspace_id FROM projects ORDER BY id").fetchall() == [
            ("p1", DEFAULT_WORKSPACE_ID),
            ("p2", DEFAULT_WORKSPACE_ID),
        ]
        assert db.execute("SELECT id, created_by_user_id FROM sessions").fetchall() == [
            ("s1", BOOTSTRAP_USER_ID)
        ]
        assert db.execute("SELECT id_hash, user_id FROM auth_sessions").fetchall() == [
            ("abc123", BOOTSTRAP_USER_ID)
        ]
        assert db.execute("SELECT code_hash, user_id FROM auth_bootstrap_codes").fetchall() == [
            ("def456", BOOTSTRAP_USER_ID)
        ]
        assert db.execute(
            "SELECT name, root_path, created_at FROM projects WHERE id = 'p1'"
        ).fetchone() == ("project-one", "/root/p1", SEED_TS)
        assert db.execute(
            "SELECT title, status, mode, created_at, updated_at FROM sessions WHERE id = 's1'"
        ).fetchone() == ("session-one", "completed", "dev", SEED_TS, SEED_TS)


def test_migration_0009_downgrade_preserves_original_data(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0008")
    with sqlite3.connect(database) as db:
        _insert_project(db, "p1", "project-one", "/root/p1")
        _insert_session(db, "s1", "p1", "session-one")
        _insert_auth_session(db, "abc123")
        _insert_auth_code(db, "def456")

    _alembic(database, "upgrade", "0009")
    _alembic(database, "downgrade", "0008")

    with sqlite3.connect(database) as db:
        assert db.execute("SELECT id, name, root_path FROM projects").fetchall() == [
            ("p1", "project-one", "/root/p1")
        ]
        assert db.execute("SELECT id, title, status FROM sessions").fetchall() == [
            ("s1", "session-one", "completed")
        ]
        assert db.execute("SELECT id_hash FROM auth_sessions").fetchall() == [("abc123",)]
        assert db.execute("SELECT code_hash FROM auth_bootstrap_codes").fetchall() == [("def456",)]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "users" not in tables
        assert "workspaces" not in tables
        assert "workspace_memberships" not in tables
        for table, column in [
            ("projects", "workspace_id"),
            ("sessions", "created_by_user_id"),
            ("auth_sessions", "user_id"),
            ("auth_bootstrap_codes", "user_id"),
        ]:
            assert column not in _columns(db, table)


def test_migration_0009_rejects_unknown_role(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        db.execute(
            "INSERT INTO users (id, display_name, enabled, is_operator, created_at) "
            "VALUES ('u1', 'User One', 1, 0, ?)",
            (SEED_TS,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO workspace_memberships (workspace_id, user_id, role, created_at) "
                "VALUES (?, ?, ?, ?)",
                (DEFAULT_WORKSPACE_ID, "u1", "superuser", SEED_TS),
            )


def test_migration_0009_partial_unique_index_enforces_single_owner(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        db.execute(
            "INSERT INTO users (id, display_name, enabled, is_operator, created_at) "
            "VALUES ('u1', 'User One', 1, 0, ?)",
            (SEED_TS,),
        )
        db.execute(
            "INSERT INTO users (id, display_name, enabled, is_operator, created_at) "
            "VALUES ('u2', 'User Two', 1, 0, ?)",
            (SEED_TS,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO workspace_memberships (workspace_id, user_id, role, created_at) "
                "VALUES (?, ?, ?, ?)",
                (DEFAULT_WORKSPACE_ID, "u1", "owner", SEED_TS),
            )
        db.execute(
            "INSERT INTO workspace_memberships (workspace_id, user_id, role, revoked_at, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (DEFAULT_WORKSPACE_ID, "u1", "owner", "2026-10-07 00:00:00", SEED_TS),
        )
        db.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES ('w2', 'Other', ?)",
            (SEED_TS,),
        )
        db.execute(
            "INSERT INTO workspace_memberships (workspace_id, user_id, role, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("w2", "u2", "owner", SEED_TS),
        )


def test_migration_0009_rejects_missing_fk_references(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO workspace_memberships (workspace_id, user_id, role, created_at) "
                "VALUES (?, ?, ?, ?)",
                (DEFAULT_WORKSPACE_ID, "no-such-user", "member", SEED_TS),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO workspace_memberships (workspace_id, user_id, role, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("no-such-workspace", BOOTSTRAP_USER_ID, "member", SEED_TS),
            )


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("projects", "workspace_id"),
        ("sessions", "created_by_user_id"),
        ("auth_sessions", "user_id"),
        ("auth_bootstrap_codes", "user_id"),
    ],
)
def test_migration_0009_enforces_all_added_foreign_keys(
    tmp_path: Path, table: str, column: str
) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0008")
    with sqlite3.connect(database) as db:
        _insert_project(db, "p1", "project-one", "/root/p1")
        _insert_session(db, "s1", "p1", "session-one")
        _insert_auth_session(db, "abc123")
        _insert_auth_code(db, "def456")
    _alembic(database, "upgrade", "0009")
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(f"UPDATE {table} SET {column} = ?", ("missing-reference",))


def test_migration_0009_roundtrip_preserves_messages_events_and_tokens(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0008")
    with sqlite3.connect(database) as db:
        _insert_project(db, "p1", "project-one", "/root/p1")
        _insert_session(db, "s1", "p1", "session-one")
        db.execute(
            "INSERT INTO messages (session_id, role, content, token_count, skipped, kind, "
            "reasoning_content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", "assistant", "Сохранённый ответ", 123, 0, "normal", "reasoning", SEED_TS),
        )
        db.execute(
            "INSERT INTO events (session_id, type, payload, created_at) VALUES (?, ?, ?, ?)",
            ("s1", "session.completed", '{"result":"saved"}', SEED_TS),
        )
        before = {
            table: db.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            for table in ("messages", "events")
        }
    for arguments in (("upgrade", "0009"), ("downgrade", "0008"), ("upgrade", "0009")):
        _alembic(database, *arguments)
        with sqlite3.connect(database) as db:
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
            for table, records in before.items():
                assert db.execute(f"SELECT * FROM {table} ORDER BY id").fetchall() == records


def test_migration_head_matches_orm_metadata(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "head")
    engine = create_engine(f"sqlite:///{database}")
    try:
        with engine.connect() as connection:
            assert "users" in Base.metadata.tables
            assert LLMProviderOverride.__tablename__ in Base.metadata.tables
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()
