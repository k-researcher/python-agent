"""MCP servers from the UI: list, add, change, check and delete.

A server is a command over stdio or an address over HTTP. Secret environment values and
secret header values are write-only: the API stores them encrypted and returns only their
names. A command always runs in the required OS sandbox.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
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
    check_header_key,
    check_header_value,
    check_read_paths,
    check_secret,
    check_server,
    check_server_id,
    check_url,
    suggested_read_paths,
)
from agent.model_overrides import SecretKeyError, encrypt_secret
from agent.models import McpServer, Project
from agent.tool_sources import ToolSourceError, load_tool_sources

router = APIRouter(prefix="/api/mcp/servers", tags=["mcp"])
Database = Annotated[AsyncSession, Depends(get_db)]
Configuration = Annotated[Settings, Depends(get_settings)]
Mode = Literal["dev", "ask"]
Protocol = Literal["auto", "modern", "legacy"]
_MAX_PROJECTS = 200
# One check of a server at a time: two checks would share its sandbox directory.
_checking: set[str] = set()


def _dev_only() -> list[Mode]:
    return ["dev"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ServerCreate(_Body):
    id: str = Field(min_length=1, max_length=32)
    # A command over stdio or an address over HTTP: exactly one of them.
    command: list[str] | None = Field(default=None, min_length=1, max_length=MAX_COMMAND_PARTS)
    url: str | None = Field(default=None, max_length=2048)
    # Variable name -> secret value. The values never come back.
    env: dict[str, str] = Field(default_factory=dict)
    # Header name -> secret value of a server over HTTP. The values never come back.
    headers: dict[str, str] = Field(default_factory=dict)
    http_protocol: Protocol = "auto"
    # Empty: every project of the install.
    project_ids: list[str] = Field(default_factory=list, max_length=_MAX_PROJECTS)
    allowed_modes: list[Mode] = Field(default_factory=_dev_only, min_length=1, max_length=2)
    allow_network: bool = False
    # None: the install directory of the program, when the sandbox may open it.
    read_paths: list[str] | None = Field(default=None, max_length=MAX_READ_PATHS)
    enabled: bool = True


class ServerUpdate(_Body):
    command: list[str] | None = Field(default=None, min_length=1, max_length=MAX_COMMAND_PARTS)
    url: str | None = Field(default=None, max_length=2048)
    # A value replaces the secret, None removes the variable, a missing name keeps it.
    env: dict[str, str | None] | None = None
    # The same rules for the headers of a server over HTTP.
    headers: dict[str, str | None] | None = None
    http_protocol: Protocol | None = None
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
        "url": server.url,
        "header_keys": sorted(server.header_secrets),
        "http_protocol": server.http_protocol,
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


def _header_secrets(values: dict[str, str], settings: Settings) -> dict[str, str]:
    if len(values) > MAX_ENV_KEYS:
        raise McpServerError(f"Не больше {MAX_ENV_KEYS} заголовков")
    names = [name.lower() for name in values]
    if len(set(names)) != len(names):
        raise McpServerError("Имена заголовков повторяются")
    return {
        check_header_key(key): encrypt_secret(check_header_value(value), settings)
        for key, value in values.items()
    }


def _merge(
    stored: dict[str, str],
    changes: dict[str, str | None],
    encrypt: Callable[[dict[str, str]], dict[str, str]],
    limit_text: str,
) -> dict[str, str]:
    """Apply the changes: a value replaces the secret, None removes it, a missing name stays."""
    secrets = {key: value for key, value in stored.items() if changes.get(key, "") is not None}
    secrets.update(encrypt({key: value for key, value in changes.items() if value is not None}))
    if len(secrets) > MAX_ENV_KEYS:
        raise McpServerError(limit_text)
    return secrets


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
        if (payload.command is None) == (payload.url is None):
            raise McpServerError("Укажите команду запуска или адрес сервера")
        if payload.url is not None:
            if payload.env or payload.read_paths or payload.allow_network:
                # The agent itself connects to the address; nothing runs in the sandbox.
                raise McpServerError("У сервера по адресу нет переменных, папок и песочницы")
            launch: dict[str, Any] = {
                "command": [],
                "url": check_url(payload.url),
                "header_secrets": _header_secrets(payload.headers, settings),
                "http_protocol": payload.http_protocol,
                "read_paths": [],
            }
        else:
            if payload.headers or payload.http_protocol != "auto":
                raise McpServerError("Заголовки и версия протокола есть только у сервера по адресу")
            command = check_command(payload.command or [])
            launch = {
                "command": command,
                "env_secrets": _secrets(payload.env, settings),
                "allow_network": payload.allow_network,
                "read_paths": (
                    suggested_read_paths(command, settings)
                    if payload.read_paths is None
                    else check_read_paths(payload.read_paths, settings)
                ),
            }
    except (McpServerError, SecretKeyError) as error:
        raise _refuse(error) from None
    if await db.get(McpServer, server_id) is not None:
        raise HTTPException(409, "MCP-сервер с таким ID уже есть")
    db.add(
        McpServer(
            id=server_id,
            project_ids=await _project_ids(db, payload.project_ids),
            allowed_modes=list(dict.fromkeys(payload.allowed_modes)),
            enabled=payload.enabled,
            **launch,
        )
    )
    return await _changed(db, settings, "created", server_id)


@router.put("/{server_id}")
async def update_server(
    server_id: str, payload: ServerUpdate, db: Database, settings: Configuration
) -> dict[str, Any]:
    server = await _server(db, server_id)
    before = _launch(server)
    try:
        stdio_fields = (payload.command, payload.env, payload.read_paths, payload.allow_network)
        http_fields = (payload.url, payload.headers, payload.http_protocol)
        if server.url is None and any(value is not None for value in http_fields):
            raise McpServerError("Сервер запускается командой: у него нет адреса и заголовков")
        if server.url is not None and any(value is not None for value in stdio_fields):
            raise McpServerError("Сервер подключается по адресу: у него нет команды и песочницы")
        if payload.command is not None:
            server.command = check_command(payload.command)
        if payload.env is not None:
            server.env_secrets = _merge(
                server.env_secrets,
                payload.env,
                lambda values: _secrets(values, settings),
                f"Не больше {MAX_ENV_KEYS} переменных",
            )
        if payload.read_paths is not None:
            server.read_paths = check_read_paths(payload.read_paths, settings)
        if payload.url is not None:
            server.url = check_url(payload.url)
        if payload.headers is not None:
            merged = _merge(
                server.header_secrets,
                payload.headers,
                lambda values: _header_secrets(values, settings),
                f"Не больше {MAX_ENV_KEYS} заголовков",
            )
            if len({key.lower() for key in merged}) != len(merged):
                raise McpServerError("Имена заголовков повторяются")
            server.header_secrets = merged
        if payload.http_protocol is not None:
            server.http_protocol = payload.http_protocol
    except (McpServerError, SecretKeyError) as error:
        raise _refuse(error) from None
    if payload.project_ids is not None:
        server.project_ids = await _project_ids(db, payload.project_ids)
    if payload.allowed_modes is not None:
        server.allowed_modes = list(dict.fromkeys(payload.allowed_modes))
    if payload.allow_network is not None:
        server.allow_network = payload.allow_network
    launch_changed = _launch(server) != before
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
        server.url,
        tuple(sorted(server.header_secrets.items())),
        server.http_protocol,
    )


@router.post("/{server_id}/check")
async def check(server_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    """Connect to the server, list its tools and keep the result."""
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
