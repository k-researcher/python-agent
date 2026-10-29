"""Native approval columns round-trip without losing existing model approvals."""

import asyncio

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine


async def test_approval_migration_preserves_model_records(tmp_path):
    async with asyncio.timeout(8):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'approval.db'}")

        def migrate(connection, direction, revision):
            config = Config("alembic.ini")
            config.attributes["connection"] = connection
            getattr(command, direction)(config, revision)

        try:
            async with engine.begin() as db:
                await db.run_sync(migrate, "upgrade", "0018")
                await db.execute(
                    text(
                        "INSERT INTO approvals (id,tool_call_id,status,created_at) "
                        "VALUES ('old','call','pending',CURRENT_TIMESTAMP)"
                    )
                )
                await db.run_sync(migrate, "upgrade", "0019")
                row = (
                    await db.execute(
                        text("SELECT executor,scope,generation FROM approvals WHERE id='old'")
                    )
                ).one()
                assert tuple(row) == ("python", "request", None)
                await db.run_sync(migrate, "downgrade", "0018")
                assert (
                    await db.execute(text("SELECT status FROM approvals WHERE id='old'"))
                ).scalar_one() == "pending"
                assert "generation" not in await db.run_sync(
                    lambda c: [column["name"] for column in inspect(c).get_columns("approvals")]
                )
                await db.run_sync(migrate, "upgrade", "0019")
        finally:
            await engine.dispose()
