"""Verify nullable historical costs and reversible attempt-ledger migration."""

import sqlite3
from pathlib import Path

from agent.models import Base
from tests.integration.test_migration_0010 import _alembic, _insert_project_and_session


def test_attempt_cost_migration_preserves_legacy_rows_and_downgrades(tmp_path: Path) -> None:
    database = tmp_path / "ledger.db"
    _alembic(database, "upgrade", "0012")
    with sqlite3.connect(database) as db:
        _insert_project_and_session(db)
        db.execute(
            "INSERT INTO outbound_audit "
            "(session_id,category,destination,operation,"
            "payload_bytes,payload_sha256,status,created_at) "
            "VALUES ('s1','llm','provider','chat',1,'abc','completed','2026-10-18 00:00:00')"
        )
    _alembic(database, "upgrade", "head")
    with sqlite3.connect(database) as db:
        columns = {row[1]: row for row in db.execute("PRAGMA table_info(outbound_audit)")}
        for name in ("pricing_snapshot", "cost_amount"):
            assert columns[name][2] == str(Base.metadata.tables["outbound_audit"].c[name].type)
            assert columns[name][3] == 0
        assert db.execute("SELECT pricing_snapshot,cost_amount FROM outbound_audit").fetchall() == [
            (None, None)
        ]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    _alembic(database, "downgrade", "0012")
    with sqlite3.connect(database) as db:
        columns_after = {row[1] for row in db.execute("PRAGMA table_info(outbound_audit)")}
        assert not {"pricing_snapshot", "cost_amount"} & columns_after
        assert db.execute("SELECT session_id,status FROM outbound_audit").fetchall() == [
            ("s1", "completed")
        ]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
