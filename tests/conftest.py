"""Isolate the test run from the developer's .env and runtime database.

Runs before any test module imports ``agent``: settings and the engine are
module-level singletons, so the environment must be prepared first.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

import pytest

_DATA_DIR = Path(tempfile.mkdtemp(prefix="python-agent-tests-"))

for _name in [name for name in os.environ if name.upper().startswith("AGENT_")]:
    del os.environ[_name]

os.environ["AGENT_ENV_FILE"] = ""
os.environ["AGENT_HOST"] = "127.0.0.1"
os.environ["AGENT_DATA_DIR"] = str(_DATA_DIR)
os.environ["AGENT_DATABASE_URL"] = f"sqlite+aiosqlite:///{_DATA_DIR / 'agent.db'}"
os.environ["AGENT_EXECUTION_MODE"] = "embedded"
os.environ["AGENT_API_TOKEN"] = ""
# Empty value: explicit legacy AGENT_LLM_* mode, independent of a local config/models.yaml.
os.environ["AGENT_MODELS_FILE"] = ""
os.environ["AGENT_AUTH_OPEN_BROWSER"] = "false"
os.environ["AGENT_ALLOWED_HOSTS"] = '["testserver", "localhost", "127.0.0.1"]'
os.environ["AGENT_EXTRA_ORIGINS"] = '["http://testserver"]'


def _worker_postgres_database() -> None:
    """Give each parallel test worker its own empty PostgreSQL database.

    The PostgreSQL race and migration tests build their schema in the database of
    RUN_FINISH_TEST_POSTGRES_URL. Parallel workers must not share it.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    url = os.environ.get("RUN_FINISH_TEST_POSTGRES_URL")
    if not worker or not url:
        return
    from sqlalchemy.engine import make_url

    base = make_url(url)
    name = f"{base.database}_{worker}"

    async def recreate() -> None:
        import asyncpg

        connection = await asyncpg.connect(
            host=base.host,
            port=base.port,
            user=base.username,
            password=base.password,
            database=base.database,
        )
        try:
            await connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await connection.execute(f'CREATE DATABASE "{name}"')
        finally:
            await connection.close()

    asyncio.run(recreate())
    os.environ["RUN_FINISH_TEST_POSTGRES_URL"] = base.set(database=name).render_as_string(
        hide_password=False
    )


_worker_postgres_database()


def backup_database(source_path: Path, target_path: Path, *, timeout: float = 5) -> None:
    """Bound SQLite backup retries, which ignore the connection busy timeout."""
    deadline = time.monotonic() + timeout

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Test database backup timed out: {source_path} -> {target_path}. "
                "A previous test may have left a connection or transaction open."
            )

    with (
        closing(sqlite3.connect(source_path, timeout=0)) as source,
        closing(sqlite3.connect(target_path, timeout=0)) as target,
    ):
        source.backup(target, pages=128, progress=progress, sleep=0.01)


@pytest.fixture(scope="session")
def empty_runtime_database() -> Path:
    from agent.database import engine, init_database

    asyncio.run(init_database(recover_running=False))
    asyncio.run(engine.dispose())
    snapshot = _DATA_DIR / "empty.db"
    backup_database(_DATA_DIR / "agent.db", snapshot)
    return snapshot


@pytest.fixture(autouse=True)
def isolate_runtime_database(empty_runtime_database: Path) -> None:
    """Do not let background recovery execute another test's durable intents."""
    from agent.database import engine

    asyncio.run(engine.dispose())
    backup_database(empty_runtime_database, _DATA_DIR / "agent.db")
