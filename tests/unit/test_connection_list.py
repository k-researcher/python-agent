from __future__ import annotations

from pathlib import Path

from agent.config import Settings
from agent.tools.advanced import ListDbConnectionsTool, ListSshConnectionsTool
from agent.tools.base import ToolContext


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=tmp_path)


async def test_db_connection_list_excludes_credentials(tmp_path: Path) -> None:
    settings = Settings(
        allow_network_tools=True,
        database_connections={
            "reporting": "postgresql+asyncpg://user:secret@db.example.com:5432/mydb"
        },
    )
    tool = ListDbConnectionsTool(settings)

    result = await tool.execute(context(tmp_path), {})

    assert result["connections"] == [
        {
            "id": "reporting",
            "driver": "postgresql",
            "host": "db.example.com",
            "port": 5432,
            "database": "mydb",
        }
    ]
    serialized = str(result)
    assert "secret" not in serialized
    assert "user" not in serialized
    assert ":secret@" not in serialized


async def test_ssh_connection_list_excludes_secrets(tmp_path: Path) -> None:
    settings = Settings(
        allow_network_tools=True,
        ssh_connections={
            "prod": {
                "host": "server.example.com",
                "port": 2222,
                "username": "admin",
                "password": "pw",
                "client_keys": ["/home/x/.ssh/id_rsa"],
                "known_hosts": ["host-key"],
                "description": "Production server",
            }
        },
    )
    tool = ListSshConnectionsTool(settings)

    result = await tool.execute(context(tmp_path), {})

    assert result["connections"] == [
        {
            "id": "prod",
            "host": "server.example.com",
            "port": 2222,
            "description": "Production server",
        }
    ]
    serialized = str(result)
    assert "admin" not in serialized
    assert "pw" not in serialized
    assert "id_rsa" not in serialized
    assert "host-key" not in serialized


async def test_empty_connection_config_returns_empty_lists(tmp_path: Path) -> None:
    settings = Settings(allow_network_tools=True)
    ctx = context(tmp_path)

    db_result = await ListDbConnectionsTool(settings).execute(ctx, {})
    ssh_result = await ListSshConnectionsTool(settings).execute(ctx, {})

    assert db_result["connections"] == []
    assert ssh_result["connections"] == []
