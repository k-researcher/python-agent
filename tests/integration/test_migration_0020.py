"""The speech profile table is added and removed cleanly."""

import asyncio

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine


async def test_speech_profile_migration_round_trip(tmp_path):
    async with asyncio.timeout(8):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'speech.db'}")

        def migrate(connection, direction, revision):
            config = Config("alembic.ini")
            config.attributes["connection"] = connection
            getattr(command, direction)(config, revision)

        def tables(connection):
            return set(inspect(connection).get_table_names())

        try:
            async with engine.begin() as db:
                await db.run_sync(migrate, "upgrade", "0020")
                assert "speech_profiles" in await db.run_sync(tables)
                await db.run_sync(migrate, "downgrade", "0019")
                assert "speech_profiles" not in await db.run_sync(tables)
        finally:
            await engine.dispose()
