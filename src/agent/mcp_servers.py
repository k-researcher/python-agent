"""MCP servers that the owner adds in the UI: a command over stdio or an address over HTTP.

Such a server serves the projects without a workspace, that is, the single-user install.
Its definition becomes a ToolSource without a workspace, so discovery, the approval
revision, the source checks and the audit are the same as for a server from the operator
file. A command always runs in the required OS sandbox; the agent itself connects to an
address. Secret environment values and secret header values are kept encrypted and never
go back to the browser.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.mcp_client import McpClient, McpError, StdioMcpClient
from agent.mcp_http import RESERVED_HEADERS, HttpMcpClient, url_refusal
from agent.model_overrides import SecretKeyError, decrypt_secret
from agent.models import McpServer
from agent.sandbox import SandboxPolicy, SandboxUnavailableError, policy_from_settings
from agent.tool_sources import ToolSource
from agent.tools.base import RiskLevel

SERVER_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HEADER_NAME = re.compile(r"[A-Za-z0-9-]{1,64}")
MAX_COMMAND_PARTS = 64
MAX_ARGUMENT_CHARS = 4096
MAX_ENV_KEYS = 32
MAX_SECRET_CHARS = 8192
MAX_READ_PATHS = 16
CHECK_SECONDS = 60.0
# The sandbox sets these variables itself; the loader variables could change the program.
_RESERVED_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_ATTR_NOSYSTEM",
        "DEVELOPER_DIR",
    }
)
_RESERVED_PREFIXES = ("DYLD_", "LD_")
_LISTED_TOOLS = 100
_DESCRIPTION_CHARS = 300
_STDERR_BYTES = 4000
# The process gets these variables of the agent; everything else comes from the server form.
_PASSED_KEYS = ("PATH", "LANG")
logger = logging.getLogger(__name__)


class McpServerError(ValueError):
    """The server definition or its check is not valid."""


def check_server_id(value: str) -> str:
    if not SERVER_ID.fullmatch(value):
        raise McpServerError(
            "ID: строчные латинские буквы, цифры, дефис и подчёркивание, до 32 знаков"
        )
    return value


def check_command(command: Sequence[str]) -> list[str]:
    """Return the command as a list, or refuse an empty or too long one."""
    if not command or not command[0].strip():
        raise McpServerError("Укажите команду запуска")
    if len(command) > MAX_COMMAND_PARTS:
        raise McpServerError(f"Не больше {MAX_COMMAND_PARTS} частей команды")
    for part in command:
        if not part or "\x00" in part or len(part) > MAX_ARGUMENT_CHARS:
            raise McpServerError("Пустая или слишком длинная часть команды")
    return list(command)


def check_env_key(key: str) -> str:
    if not ENV_KEY.fullmatch(key) or len(key) > 128:
        raise McpServerError(f"Недопустимое имя переменной: {key[:40]}")
    if key in _RESERVED_KEYS or key.startswith(_RESERVED_PREFIXES):
        raise McpServerError(f"Переменную {key} задаёт песочница, её нельзя менять")
    return key


def check_secret(value: str) -> str:
    if "\x00" in value or len(value) > MAX_SECRET_CHARS:
        raise McpServerError("Значение переменной слишком длинное или содержит нулевой символ")
    return value


def check_url(value: str) -> str:
    """Return the address of a server over HTTP, or refuse it with the reason."""
    refusal = url_refusal(value)
    if refusal is not None:
        raise McpServerError(refusal)
    return value


def check_header_key(name: str) -> str:
    if not HEADER_NAME.fullmatch(name):
        raise McpServerError(f"Недопустимое имя заголовка: {name[:40]}")
    if name.lower() in RESERVED_HEADERS:
        raise McpServerError(f"Заголовок {name} задаёт агент, его нельзя менять")
    return name


def check_header_value(value: str) -> str:
    if any(char in value for char in "\r\n\x00") or len(value) > MAX_SECRET_CHARS:
        raise McpServerError("Значение заголовка слишком длинное или содержит перевод строки")
    return value


def _overlaps(path: Path, other: Path) -> bool:
    return path == other or path in other.parents or other in path.parents


def check_read_path(value: str, settings: Settings) -> str:
    """Return a normal absolute directory that the sandbox may open for reading.

    The file system root, the home directory and its parents are too wide. The data
    directory of the agent holds the database and the key of the secrets, so a path inside
    it or above it is refused.
    """
    if "\x00" in value or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise McpServerError("Путь для чтения должен быть абсолютным")
    path = Path(os.path.normpath(value))
    home = Path.home()
    if path == Path(path.anchor) or path == home or path in home.parents:
        raise McpServerError(f"Путь {path} слишком широкий для песочницы")
    data = Path(os.path.abspath(settings.data_dir))
    if _overlaps(path, data) or _overlaps(path.resolve(), data.resolve()):
        raise McpServerError(f"Путь {path} открывает папку данных агента")
    return str(path)


def check_read_paths(values: Iterable[str], settings: Settings) -> list[str]:
    paths = list(dict.fromkeys(check_read_path(value, settings) for value in values))
    if len(paths) > MAX_READ_PATHS:
        raise McpServerError(f"Не больше {MAX_READ_PATHS} путей для чтения")
    return paths


def _prefix(program: Path) -> str | None:
    """Return the directory above the bin directory of a program, unless it is too wide."""
    prefix = program.parent.parent
    home = Path.home()
    if prefix == Path(prefix.anchor) or prefix == home or prefix in home.parents:
        return None
    return str(prefix)


def _interpreter(script: Path) -> str | None:
    """Return the program named on the first line of a script, such as its Python."""
    try:
        with script.open("rb") as handle:
            line = handle.readline(512)
    except OSError:
        return None
    if not line.startswith(b"#!"):
        return None
    parts = line[2:].decode(errors="replace").split()
    if parts and Path(parts[0]).name == "env":
        parts = [part for part in parts[1:] if not part.startswith("-")]
        return shutil.which(parts[0]) if parts else None
    return parts[0] if parts and Path(parts[0]).is_absolute() else None


def runtime_read_paths(command: Sequence[str], settings: Settings) -> list[str]:
    """Return the install directories that a server program needs to start.

    A virtual environment links its Python to an interpreter elsewhere, such as a uv or
    pyenv install, and a script names its interpreter on the first line. The sandbox opens
    the real install directory of the program and of that interpreter for reading, so the
    settings do not need these folders and an interpreter update does not break the server.
    For a program in a virtual environment it opens the project folder that holds the
    environment, because a project installed in development mode imports its code from there.
    """
    found = command[0] if "/" in command[0] else shutil.which(command[0])
    if found is None or not Path(found).is_absolute():
        return []
    # Both the path as written and the real path count: npx of Homebrew is a link from
    # /opt/homebrew/bin into lib/node_modules, and it needs the whole install.
    programs = [Path(os.path.abspath(found)), Path(found).resolve()]
    interpreter = _interpreter(programs[1])
    if interpreter is not None:
        programs += [Path(os.path.abspath(interpreter)), Path(interpreter).resolve()]
    paths: list[str] = []
    for program in programs:
        prefix = _prefix(program)
        if prefix is not None and (Path(prefix) / "pyvenv.cfg").is_file():
            # A project installed into its own environment keeps its code next to it.
            # When that folder is too wide, the environment alone is opened.
            prefix = _prefix(Path(prefix) / "bin") or prefix
        if prefix is None or prefix in paths:
            continue
        try:
            paths.append(check_read_path(prefix, settings))
        except McpServerError:
            continue
    return paths


def to_source(server: McpServer) -> ToolSource:
    """Return the trusted source definition of a server from the UI."""
    if server.url is not None:
        launch: dict[str, Any] = {
            "url": server.url,
            "header_keys": tuple(sorted(server.header_secrets)),
            "http_protocol": server.http_protocol,
        }
    else:
        launch = {
            "command": tuple(server.command),
            "env_keys": tuple(sorted(server.env_secrets)),
            "allow_network": server.allow_network,
            "read_paths": tuple(server.read_paths),
        }
    return ToolSource(
        source_id=server.id,
        workspace_id=None,
        project_ids=frozenset(server.project_ids),
        enabled=server.enabled,
        risk_level=RiskLevel(server.risk_level),
        allowed_modes=frozenset(server.allowed_modes),  # type: ignore[arg-type]
        call_timeout_seconds=server.call_timeout,
        **launch,
    )


def server_environment(server: McpServer, settings: Settings) -> dict[str, str]:
    """Return the process environment: a few agent variables and the decrypted secrets."""
    environment = {key: os.environ[key] for key in _PASSED_KEYS if key in os.environ}
    for key, ciphertext in server.env_secrets.items():
        environment[key] = decrypt_secret(ciphertext, settings)
    return environment


def server_secrets(server: McpServer, settings: Settings) -> dict[str, str]:
    """Return the process environment of a command, or the headers of an address."""
    if server.url is None:
        return server_environment(server, settings)
    return {key: decrypt_secret(value, settings) for key, value in server.header_secrets.items()}


def open_client(
    settings: Settings,
    source: ToolSource,
    secrets: dict[str, str],
    *,
    cwd: Path,
    session_id: str,
    request_timeout: float = 30.0,
    stderr_tail_bytes: int = 0,
) -> McpClient:
    """Return a client for the source; ``secrets`` are its environment or its headers."""
    if source.url is not None:
        return HttpMcpClient(
            source.url,
            headers=secrets,
            request_timeout=request_timeout,
            call_timeout=source.call_timeout_seconds,
            protocol=source.http_protocol,
        )
    return StdioMcpClient(
        list(source.command),
        cwd=cwd,
        env=secrets,
        sandbox=sandbox_policy(settings, source),
        session_id=session_id,
        request_timeout=request_timeout,
        call_timeout=source.call_timeout_seconds,
        stderr_tail_bytes=stderr_tail_bytes,
    )


def file_environment(source: ToolSource) -> dict[str, str]:
    """Return the process environment of a source from the operator file."""
    return {key: os.environ[key] for key in (*_PASSED_KEYS, *source.env_keys) if key in os.environ}


def sandbox_policy(settings: Settings, source: ToolSource) -> SandboxPolicy:
    """Return the required sandbox of a server: its network choice and its read paths."""
    base = policy_from_settings(settings)
    found = runtime_read_paths(source.command, settings) if source.command else []
    return replace(
        base,
        mode="required",
        allow_network=base.allow_network or source.allow_network,
        extra_read_paths=tuple(dict.fromkeys((*base.extra_read_paths, *source.read_paths, *found))),
    )


async def ui_sources(
    db: AsyncSession, settings: Settings
) -> list[tuple[ToolSource, dict[str, str]]]:
    """Return the enabled servers from the UI with their secrets (see ``server_secrets``).

    A server whose secrets cannot be decrypted is left out, so the other servers still work.
    """
    found: list[tuple[ToolSource, dict[str, str]]] = []
    for server in (await db.scalars(select(McpServer).where(McpServer.enabled.is_(True)))).all():
        try:
            found.append((to_source(server), server_secrets(server, settings)))
        except (SecretKeyError, ValueError):
            logger.warning(
                "MCP server %s is left out: its definition or secrets are not valid", server.id
            )
    return found


def _check_error(error: McpError) -> str:
    """Explain a refusal of a server over HTTP in the words of the UI."""
    if error.status in {401, 403}:
        return "Сервер отказал в доступе: проверьте заголовок с ключом"
    if error.status is not None and 300 <= error.status < 400:
        return "Сервер перенаправляет запрос на другой адрес: укажите конечный адрес"
    if error.status == 404:
        return "По этому адресу нет MCP-сервера: проверьте путь, например /mcp"
    if error.message == "could not connect":
        return "Не удалось подключиться к серверу"
    if error.message.startswith("unsupported protocol version"):
        return "Сервер говорит на версии MCP, которую агент пока не поддерживает"
    return error.message


async def check_server(settings: Settings, server: McpServer) -> dict[str, Any]:
    """Connect to the server, list its tools and disconnect.

    A command runs in its sandbox in an empty directory of its own, not in a project.
    Return the result that the UI shows and that is kept as ``McpServer.last_check``.
    """
    checked_at = datetime.now(UTC).isoformat()
    source = to_source(server)
    workdir = settings.data_dir / "mcp-check" / server.id
    if source.url is None:
        workdir.mkdir(parents=True, exist_ok=True)
    try:
        secrets = server_secrets(server, settings)
    except SecretKeyError:
        return {"ok": False, "checked_at": checked_at, "error": "Не удалось расшифровать секреты"}
    client = open_client(
        settings,
        source,
        secrets,
        cwd=workdir,
        session_id=f"mcp-check-{server.id}",
        request_timeout=CHECK_SECONDS,
        stderr_tail_bytes=_STDERR_BYTES,
    )
    try:
        async with asyncio.timeout(CHECK_SECONDS):
            await client.start()
            tools = await client.list_tools()
            info = client.server_info
            protocol = client.protocol_version
    except SandboxUnavailableError:
        return {
            "ok": False,
            "checked_at": checked_at,
            "error": "Песочница ОС недоступна на этом сервере, MCP-серверы не запускаются",
        }
    except (McpError, OSError, TimeoutError) as exc:
        if isinstance(exc, TimeoutError) or (
            isinstance(exc, McpError) and exc.message == "timed out"
        ):
            message = "Сервер не ответил за минуту"
        elif isinstance(exc, McpError):
            message = _check_error(exc)
        else:
            message = str(exc)
        failed = {"ok": False, "checked_at": checked_at, "error": message}
        if isinstance(client, StdioMcpClient):
            failed["stderr"] = _last_lines(client.stderr_tail)
        return failed
    finally:
        await client.close()
    return {
        "ok": True,
        "checked_at": checked_at,
        "server": {"name": str(info.get("name", "")), "version": str(info.get("version", ""))},
        "tools": [
            {
                "name": tool.name,
                "description": " ".join(tool.description.split())[:_DESCRIPTION_CHARS],
            }
            for tool in tools[:_LISTED_TOOLS]
        ],
        "tool_count": len(tools),
        # The version that the server and the agent agreed on over HTTP.
        **({"protocol": protocol} if protocol else {}),
    }


def _last_lines(text: str, count: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-count:])
