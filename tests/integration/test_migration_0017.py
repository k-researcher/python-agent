"""Backend XOR is enforced even for direct SQL; old sessions survive both directions."""

import asyncio
import os
import sqlite3
import uuid
from pathlib import Path

import pytest

from tests.integration.test_migration_0010 import _alembic, _insert_project_and_session


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
async def test_migrated_backend_xor_on_each_database(tmp_path: Path, dialect: str):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import create_async_engine

    schema = f"engine_test_{uuid.uuid4().hex}"
    if dialect == "postgresql":
        url = os.environ.get("RUN_FINISH_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("RUN_FINISH_TEST_POSTGRES_URL is not set")
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'backends.db'}")

    def migrate(connection, operation, revision):
        config = Config("alembic.ini")
        config.attributes["connection"] = connection
        getattr(command, operation)(config, revision)

    async with asyncio.timeout(8):
        try:
            async with engine.begin() as db:
                if dialect == "postgresql":
                    await db.execute(text(f'CREATE SCHEMA "{schema}"'))
                await db.run_sync(migrate, "upgrade", "0016")
                await db.execute(
                    text(
                        "INSERT INTO projects (id,name,root_path,created_at) "
                        "VALUES ('p','old','/tmp/old',CURRENT_TIMESTAMP)"
                    )
                )
                await db.execute(
                    text(
                        "INSERT INTO sessions (id,project_id,mode,title,status,archived,"
                        "stop_requested,configuration,created_at,updated_at) VALUES "
                        "('s','p','dev','old','completed',false,false,'{}',"
                        "CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
                    )
                )
                await db.run_sync(migrate, "upgrade", "0017")
                assert (
                    await db.execute(
                        text("SELECT llm_profile,engine_profile,engine_state FROM sessions")
                    )
                ).one() == ("default", None, None)
                for model, profile_id in [
                    (None, None),
                    ("default", "personal"),
                    ("", None),
                    (None, ""),
                    (" ", None),
                    (None, " "),
                ]:
                    with pytest.raises(IntegrityError):
                        async with db.begin_nested():
                            await db.execute(
                                text(
                                    "UPDATE sessions SET llm_profile=:model,engine_profile=:engine"
                                ),
                                {"model": model, "engine": profile_id},
                            )
                await db.execute(
                    text(
                        "UPDATE sessions SET llm_profile=NULL,"
                        "engine_profile='personal',engine_state='{}'"
                    )
                )
                assert (
                    await db.execute(text("SELECT llm_profile,engine_profile FROM sessions"))
                ).one() == (None, "personal")
                await db.execute(
                    text(
                        "UPDATE sessions SET llm_profile='default',"
                        "engine_profile=NULL,engine_state=NULL"
                    )
                )
                await db.run_sync(migrate, "downgrade", "0016")
                with pytest.raises(IntegrityError):
                    async with db.begin_nested():
                        await db.execute(text("UPDATE sessions SET llm_profile=NULL"))
                assert (await db.execute(text("SELECT llm_profile,title FROM sessions"))).one() == (
                    "default",
                    "old",
                )
                await db.run_sync(migrate, "upgrade", "0017")
        finally:
            if dialect == "postgresql":
                async with engine.begin() as db:
                    await db.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await engine.dispose()


def test_backend_migration_up_down(tmp_path: Path):
    path = tmp_path / "engines.db"
    _alembic(path, "upgrade", "0016")
    with sqlite3.connect(path) as db:
        _insert_project_and_session(db)
    _alembic(path, "upgrade", "0017")
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT llm_profile,engine_profile,engine_state FROM sessions"
        ).fetchone() == ("default", None, None)
        for model, engine in [(None, None), ("default", "personal"), ("", None), (None, "")]:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute("UPDATE sessions SET llm_profile=?,engine_profile=?", (model, engine))
        db.execute(
            "UPDATE sessions SET llm_profile=NULL,engine_profile='personal',engine_state='{}'"
        )
    # Downgrade cannot silently turn a personal engine session into an API session.
    with pytest.raises(RuntimeError, match="engine sessions"):
        _alembic(path, "downgrade", "0016")
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE sessions SET llm_profile='default',engine_profile=NULL,engine_state=NULL"
        )
    _alembic(path, "downgrade", "0016")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT llm_profile,title FROM sessions").fetchone() == (
            "default",
            "Old session",
        )
        assert not {"engine_profile", "engine_state"} & {
            r[1] for r in db.execute("PRAGMA table_info(sessions)")
        }
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE sessions SET llm_profile=NULL")
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    _alembic(path, "upgrade", "0017")
