"""MCP servers over stdio from the UI: list, add, change, check and delete.

Secret environment values are write-only: the API stores them encrypted and returns only
their names. A server always runs in the required OS sandbox.
"""

from __future__ import annotations

import asyncio
import shutil
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import sandbox
from agent.config import Settings, get_settings
from agent.database import get_db
from agent.events import emit_event
from agent.mcp_servers import (
    MAX_COMMAND_PARTS,
    MAX_ENV_KEYS,
    MAX_READ_PATHS,
    McpServerError,
    check_command,
    check_env_key,
    check_read_paths,
    check_secret,
    check_server,
    check_server_id,
    suggested_read_paths,
)
from agent.model_overrides import SecretKeyError, encrypt_secret
from agent.models import McpServer, Project
from agent.tool_sources import ToolSourceError, load_tool_sources

router = APIRouter(prefix="/api/mcp/servers", tags=["mcp"])
Database = Annotated[AsyncSession, Depends(get_db)]
Configuration = Annotated[Settings, Depends(get_settings)]
Mode = Literal["dev", "ask"]
_MAX_PROJECTS = 200
# One check of a server at a time: two checks would share its sandbox directory.
_checking: set[str] = set()


def _dev_only() -> list[Mode]:
    return ["dev"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ServerCreate(_Body):
    id: str = Field(min_length=1, max_length=32)
    command: list[str] = Field(min_length=1, max_length=MAX_COMMAND_PARTS)
    # Variable name -> secret value. The values never come back.
    env: dict[str, str] = Field(default_factory=dict)
    # Empty: every project of the install.
    project_ids: list[str] = Field(default_factory=list, max_length=_MAX_PROJECTS)
    allowed_modes: list[Mode] = Field(default_factory=_dev_only, min_length=1, max_length=2)
    allow_network: bool = False
    # None: the install directory of the program, when the sandbox may open it.
    read_paths: list[str] | None = Field(default=None, max_length=MAX_READ_PATHS)
    enabled: bool = True


class ServerUpdate(_Body):
    command: list[str] | None = Field(default=None, min_length=1, max_length=MAX_COMMAND_PARTS)
    # A value replaces the secret, None removes the variable, a missing name keeps it.
    env: dict[str, str | None] | None = None
    project_ids: list[str] | None = Field(default=None, max_length=_MAX_PROJECTS)
    allowed_modes: list[Mode] | None = Field(default=None, min_length=1, max_length=2)
    allow_network: bool | None = None
    read_paths: list[str] | None = Field(default=None, max_length=MAX_READ_PATHS)
    enabled: bool | None = None


def _refuse(error: McpServerError | SecretKeyError) -> HTTPException:
    if isinstance(error, SecretKeyError):
        return HTTPException(422, "Не удалось зашифровать секреты: проверьте ключ шифрования")
    return HTTPException(422, str(error))


def _file_ids(settings: Settings) -> list[str]:
    try:
        return sorted(source.source_id for source in load_tool_sources(settings.tool_sources_file))
    except (ToolSourceError, OSError):
        return []


def _view(server: McpServer) -> dict[str, Any]:
    return {
        "id": server.id,
        "command": list(server.command),
        "env_keys": sorted(server.env_secrets),
        "project_ids": list(server.project_ids),
        "allowed_modes": list(server.allowed_modes),
        "allow_network": server.allow_network,
        "read_paths": list(server.read_paths),
        "enabled": server.enabled,
        "last_check": server.last_check,
        "updated_at": server.updated_at.isoformat() if server.updated_at else None,
    }


async def _list(db: AsyncSession, settings: Settings) -> dict[str, Any]:
    servers = (await db.scalars(select(McpServer).order_by(McpServer.id))).all()
    return {
        # The check runs a command once; later calls take the cached answer.
        "sandbox": await asyncio.to_thread(sandbox.available),
        "file_sources": _file_ids(settings),
        "servers": [_view(server) for server in servers],
    }


async def _changed(
    db: AsyncSession, settings: Settings, action: str, server_id: str
) -> dict[str, Any]:
    # The event commits the change of the same request.
    await emit_event(db, "mcp.changed", {"action": action, "id": server_id})
    return await _list(db, settings)


async def _server(db: AsyncSession, server_id: str) -> McpServer:
    server = await db.get(McpServer, server_id)
    if server is None:
        raise HTTPException(404, "MCP-сервер не найден")
    return server


async def _project_ids(db: AsyncSession, values: list[str]) -> list[str]:
    """Return the projects without a workspace; a UI server serves only such projects."""
    wanted = list(dict.fromkeys(values))
    if not wanted:
        return []
    found = (await db.scalars(select(Project).where(Project.id.in_(wanted)))).all()
    known = {project.id for project in found if project.workspace_id is None}
    missing = [value for value in wanted if value not in known]
    if missing:
        raise HTTPException(422, "Проект не найден или принадлежит команде")
    return wanted


def _secrets(values: dict[str, str], settings: Settings) -> dict[str, str]:
    if len(values) > MAX_ENV_KEYS:
        raise McpServerError(f"Не больше {MAX_ENV_KEYS} переменных")
    return {
        check_env_key(key): encrypt_secret(check_secret(value), settings)
        for key, value in values.items()
    }


@router.get("")
async def list_servers(db: Database, settings: Configuration) -> dict[str, Any]:
    return await _list(db, settings)


@router.post("", status_code=201)
async def create_server(
    payload: ServerCreate, db: Database, settings: Configuration
) -> dict[str, Any]:
    try:
        server_id = check_server_id(payload.id)
        if server_id in _file_ids(settings):
            raise McpServerError("Такой ID уже занят сервером из файла настроек")
        command = check_command(payload.command)
        read_paths = (
            suggested_read_paths(command, settings)
            if payload.read_paths is None
            else check_read_paths(payload.read_paths, settings)
        )
        secrets = _secrets(payload.env, settings)
    except (McpServerError, SecretKeyError) as error:
        raise _refuse(error) from None
    if await db.get(McpServer, server_id) is not None:
        raise HTTPException(409, "MCP-сервер с таким ID уже есть")
    db.add(
        McpServer(
            id=server_id,
            command=command,
            env_secrets=secrets,
            project_ids=await _project_ids(db, payload.project_ids),
            allowed_modes=list(dict.fromkeys(payload.allowed_modes)),
            allow_network=payload.allow_network,
            read_paths=read_paths,
            enabled=payload.enabled,
        )
    )
    return await _changed(db, settings, "created", server_id)


@router.put("/{server_id}")
async def update_server(
    server_id: str, payload: ServerUpdate, db: Database, settings: Configuration
) -> dict[str, Any]:
    server = await _server(db, server_id)
    launch_changed = False
    try:
        if payload.command is not None:
            server.command = check_command(payload.command)
            launch_changed = True
        if payload.env is not None:
            secrets = dict(server.env_secrets)
            added = {key: value for key, value in payload.env.items() if value is not None}
            for key, value in payload.env.items():
                if value is None:
                    secrets.pop(key, None)
            secrets.update(_secrets(added, settings))
            if len(secrets) > MAX_ENV_KEYS:
                raise McpServerError(f"Не больше {MAX_ENV_KEYS} переменных")
            server.env_secrets = secrets
            launch_changed = True
        if payload.read_paths is not None:
            server.read_paths = check_read_paths(payload.read_paths, settings)
            launch_changed = True
    except (McpServerError, SecretKeyError) as error:
        raise _refuse(error) from None
    if payload.project_ids is not None:
        server.project_ids = await _project_ids(db, payload.project_ids)
    if payload.allowed_modes is not None:
        server.allowed_modes = list(dict.fromkeys(payload.allowed_modes))
    if payload.allow_network is not None:
        server.allow_network = payload.allow_network
        launch_changed = True
    if payload.enabled is not None:
        server.enabled = payload.enabled
    if launch_changed:
        # The old result describes another launch.
        server.last_check = None
    return await _changed(db, settings, "updated", server_id)


@router.delete("/{server_id}")
async def delete_server(server_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    server = await _server(db, server_id)
    await db.delete(server)
    result = await _changed(db, settings, "deleted", server_id)
    workdir = settings.data_dir / "mcp-check" / server_id
    if server_id not in _checking:
        await asyncio.to_thread(shutil.rmtree, workdir, ignore_errors=True)
    return result


def _launch(server: McpServer) -> tuple[Any, ...]:
    return (
        tuple(server.command),
        tuple(sorted(server.env_secrets.items())),
        tuple(server.read_paths),
        server.allow_network,
    )


@router.post("/{server_id}/check")
async def check(server_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    """Start the server in its sandbox, list its tools and keep the result."""
    server = await _server(db, server_id)
    if server_id in _checking:
        raise HTTPException(409, "Проверка этого сервера уже идёт")
    _checking.add(server_id)
    try:
        before = _launch(server)
        # Do not hold a database transaction while the server starts.
        await db.commit()
        result = await check_server(settings, server)
    finally:
        _checking.discard(server_id)
    fresh = await db.get(McpServer, server_id, populate_existing=True)
    if fresh is None:
        raise HTTPException(404, "MCP-сервер удалён во время проверки")
    if _launch(fresh) != before:
        raise HTTPException(409, "Настройки сервера изменились во время проверки, проверьте снова")
    fresh.last_check = result
    return await _changed(db, settings, "checked", server_id)
