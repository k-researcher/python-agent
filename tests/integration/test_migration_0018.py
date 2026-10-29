"""Journal uniqueness, state checks and reversible schema on both databases."""

import asyncio
import os
import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
async def test_engine_journal_migration(tmp_path, dialect):
    async with asyncio.timeout(8):
        schema = "journal_test_" + uuid.uuid4().hex
        if dialect == "postgresql":
            url = os.environ.get("RUN_FINISH_TEST_POSTGRES_URL")
            if not url:
                pytest.skip("RUN_FINISH_TEST_POSTGRES_URL is not set")
            engine = create_async_engine(
                url, connect_args={"server_settings": {"search_path": schema}}
            )
        else:
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")

        def migrate(connection, direction, revision):
            config = Config("alembic.ini")
            config.attributes["connection"] = connection
            getattr(command, direction)(config, revision)

        try:
            async with engine.begin() as db:
                if dialect == "postgresql":
                    await db.execute(text(f'CREATE SCHEMA "{schema}"'))
                await db.run_sync(migrate, "upgrade", "0017")
                await db.execute(
                    text(
                        "INSERT INTO projects (id,name,root_path,created_at) VALUES ('p','p','/"
                        "tmp',CURRENT_TIMESTAMP)"
                    )
                )
                await db.execute(
                    text(
                        "INSERT INTO sessions (id,project_id,mode,title,status,archived,stop_re"
                        "quested,configuration,created_at,updated_at) VALUES ('s','p','dev','s'"
                        ",'pending',false,false,'{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
                    )
                )
                await db.execute(
                    text(
                        "INSERT INTO messages (id,session_id,role,content,skipped,kind,created_"
                        "at) VALUES (1,'s','user','work',false,'normal',CURRENT_TIMESTAMP)"
                    )
                )
                await db.run_sync(migrate, "upgrade", "0018")
                insert = text(
                    "INSERT INTO engine_turns (id,session_id,generation,input_message_id,su"
                    "bmission_id,state,last_item_ids) VALUES (:id,'s',1,1,:submission,:stat"
                    "e,'{}')"
                )
                await db.execute(insert, {"id": "a", "submission": "first", "state": "prepared"})
                for state in ("prepared", "invalid"):
                    with pytest.raises(IntegrityError):
                        async with db.begin_nested():
                            await db.execute(
                                insert, {"id": "b", "submission": "second", "state": state}
                            )
                await db.run_sync(migrate, "downgrade", "0017")
                assert "engine_turns" not in await db.run_sync(
                    lambda c: inspect(c).get_table_names()
                )
                assert (
                    await db.execute(text("SELECT content FROM messages WHERE id=1"))
                ).scalar_one() == "work"
                await db.run_sync(migrate, "upgrade", "0018")
        finally:
            if dialect == "postgresql":
                async with engine.begin() as db:
                    await db.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await engine.dispose()
