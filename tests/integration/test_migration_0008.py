"""Migration 0008: summary checkpoints, token calibration and attempt audit fields."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _alembic(database: Path, *args: str) -> None:
    env = {
        **os.environ,
        "AGENT_ENV_FILE": "",
        "AGENT_MODELS_FILE": "",
        "AGENT_DATABASE_URL": f"sqlite+aiosqlite:///{database}",
    }
    subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )


def _insert(db: sqlite3.Connection, message_id: int, kind: str = "normal", **extra: object) -> None:
    columns = {
        "id": message_id,
        "session_id": "s1",
        "role": "user",
        "content": f"message {message_id}",
        "skipped": 0,
        "created_at": "2026-10-06 00:00:00",
        "kind": kind,
        **extra,
    }
    names = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    db.execute(f"INSERT INTO messages ({names}) VALUES ({marks})", tuple(columns.values()))


def test_migration_0008_keeps_sources_and_removes_only_checkpoints(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0007")
    with sqlite3.connect(database) as db:
        _insert(db, 1)
        _insert(db, 2, role="assistant")
        _insert(db, 3, kind="interrupted", role="assistant")

    _alembic(database, "upgrade", "0008")
    with sqlite3.connect(database) as db:
        rows = db.execute("SELECT id, kind, covers_until FROM messages ORDER BY id").fetchall()
        assert rows == [(1, "normal", None), (2, "normal", None), (3, "interrupted", None)]
        _insert(db, 4, kind="summary", covers_until=2, source_hash="a" * 64, summary_version=1)
        db.execute(
            "INSERT INTO token_calibrations (provider, model, wire_version, updated_at)"
            " VALUES ('p', 'm', 'chat-json-v1:x', '2026-10-06 00:00:00')"
        )
        assert db.execute("SELECT factor, samples FROM token_calibrations").fetchone() == (1.0, 0)

    _alembic(database, "downgrade", "0007")
    with sqlite3.connect(database) as db:
        assert [row[0] for row in db.execute("SELECT id FROM messages ORDER BY id")] == [1, 2, 3]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
        assert "token_calibrations" not in tables


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("summary", {"covers_until": 2, "source_hash": None, "summary_version": 1}),
        ("summary", {"covers_until": 0, "source_hash": "a" * 64, "summary_version": 1}),
        (
            "summary",
            {"role": "system", "covers_until": 2, "source_hash": "a", "summary_version": 1},
        ),
        ("summary", {"content": "", "covers_until": 2, "source_hash": "a", "summary_version": 1}),
        ("normal", {"covers_until": 2}),
        ("normal", {"summary_model": "light"}),
    ],
)
def test_migration_0008_rejects_invalid_summary_fields(
    tmp_path: Path, kind: str, extra: dict[str, object]
) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0008")
    with sqlite3.connect(database) as db, pytest.raises(sqlite3.IntegrityError):
        _insert(db, 10, kind=kind, **extra)


def test_migration_0008_rejects_duplicate_checkpoint(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    _alembic(database, "upgrade", "0008")
    checkpoint = {"covers_until": 2, "source_hash": "a" * 64, "summary_version": 1}
    with sqlite3.connect(database) as db:
        _insert(db, 10, kind="summary", **checkpoint)
        with pytest.raises(sqlite3.IntegrityError):
            _insert(db, 11, kind="summary", **checkpoint)
