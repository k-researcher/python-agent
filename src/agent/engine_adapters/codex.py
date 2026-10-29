"""Personal plan app-server transport, pinned to the recorded 0.164.0 protocol.

No credentials are read here. The unmodified CLI owns authentication and history.
Restricted permissions profiles require the experimental protocol fields; legacy
sandbox overrides would disable them and are deliberately never sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent.config import EngineProfileSettings, get_settings
from agent.engine import (
    ApprovalEvent,
    DiffEvent,
    DoneEvent,
    EngineEvent,
    EngineSpec,
    ErrorEvent,
    Health,
    Limits,
    LimitsEvent,
    LimitWindow,
    ReasoningEvent,
    TextEvent,
    ThreadRef,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UserInputEvent,
)
from agent.engine_process import MAX_LINE_BYTES, GuardedProcess, decode_line

_DEFAULT_RANGE = ">=0.164.0,<0.164.1"
_PROFILE = "agent_session"
_APPROVALS = {
    "item/commandExecution/requestApproval": "command",
    "item/fileChange/requestApproval": "file",
    "item/permissions/requestApproval": "permissions",
}
_DENIED_PARTS = {".git", ".agents", ".codex", ".aws", ".ssh", ".gnupg", ".azure"}
_DENIED_FILES = {"config/models.yaml", "config/sources.yaml"}
_DENIED_NAMES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    "auth.json",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
_DENIED_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".env")


class _ProtocolError(RuntimeError):
    """Messages are static: never include native auth packets or stderr."""


class _IsolationError(RuntimeError):
    pass


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _ProtocolError("Expected protocol object")
    return value


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise _ProtocolError("Expected protocol string")
    return value


def _identity(value: Any) -> str:
    text = _string(value)
    if not 1 <= len(text) <= 200:
        raise _ProtocolError("Invalid protocol identity")
    return text


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise _ProtocolError("Expected nonnegative protocol integer")
    return value


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise _ProtocolError("Expected protocol string list")
    return value


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + "=" + _toml(v) for k, v in value.items()) + "}"
    return json.dumps(value, ensure_ascii=False)


def _supported(version: str, setting: Any) -> bool:
    match = re.fullmatch(r">=(\d+\.\d+\.\d+),<(\d+\.\d+\.\d+)", str(setting))
    if not match or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return False

    def parts(v: str) -> tuple[int, ...]:
        return tuple(map(int, v.split(".")))

    return parts(match[1]) <= parts(version) < parts(match[2])


class Engine:
    def __init__(self, profile: EngineProfileSettings) -> None:
        self.profile = profile
        self._health: Health | None = None
        self._process: GuardedProcess | None = None
        self._reader: asyncio.Task[None] | None = None
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        self._responses: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._incoming: asyncio.Queue[dict[str, Any] | Exception] = asyncio.Queue(128)
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._spec: EngineSpec | None = None
        self._thread: str | None = None
        self._turn: str | None = None
        self._submission: str | None = None
        self._seq = 0
        self._pending: dict[str, tuple[str | int, str, dict[str, Any]]] = {}
        self._stopping = False
        self._terminal = False
        self._limits: Limits | None = None
        self._finals: set[str] = set()
        self._seen_requests: set[str] = set()
        self._auth_kind: str | None = None
        self._calls: set[str] = set()
        self._changes: dict[str, list[dict[str, Any]]] = {}
        self._config: dict[str, Any] = {}

    def _envelope(self, item_id: str | None = None) -> dict[str, Any]:
        self._seq += 1
        return dict(
            submission_id=self._submission,
            thread_id=self._thread,
            turn_id=self._turn,
            item_id=item_id,
            seq=self._seq,
        )

    def _degraded(self, diagnostic: str) -> None:
        previous = self._health or Health(status="degraded")
        self._health = Health(
            status="degraded",
            binary=previous.binary,
            version=previous.version,
            supported=previous.supported,
            auth_kind=previous.auth_kind,
            isolation_capable=False,
            diagnostic=diagnostic,
        )

    async def health(self) -> Health:
        if self._health is not None:
            return self._health
        binary = shutil.which(self.profile.binary)
        if binary is None:
            self._health = Health(status="unavailable", diagnostic="CLI binary unavailable")
            return self._health
        version = None
        supported = False
        auth = None
        status: Literal["ready", "degraded", "unavailable"] = "unavailable"
        diagnostic = None
        try:
            async with asyncio.timeout(self.profile.timeouts.get("open", 30)):
                self._temp = tempfile.TemporaryDirectory(prefix="agent-engine-")
                root = Path(self._temp.name).resolve()
                self._process = await GuardedProcess.start(
                    [binary, "--version"],
                    cwd=root,
                    lock_path=root / "version.lock",
                    temporary_dir=root,
                )
                raw = await self._process.stdout.readline()
                if len(raw) > MAX_LINE_BYTES:
                    raise _ProtocolError("Oversize version response")
                text = raw.decode("utf-8").strip()
                version = text.removeprefix("codex-cli ")
                if version == text:
                    raise _ProtocolError("Invalid CLI version response")
                await self._process.close()
                self._process = None
                if not _supported(
                    version, self.profile.adapter_settings.get("supported_versions", _DEFAULT_RANGE)
                ):
                    status = "degraded"
                    raise _ProtocolError("Unsupported CLI version")
                supported = True
                # No thread and no turn: this checks protocol/auth/config only.
                await self._connect(EngineSpec(root, "read", "", {}), binary)
                auth = "chatgpt"
                status = "ready"
        except _IsolationError:
            diagnostic = "Required restricted filesystem isolation unavailable"
        except PermissionError:
            diagnostic = "Plan login required; API or unknown authentication refused"
        except _ProtocolError as error:
            status = "degraded"
            diagnostic = str(error)
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            diagnostic = f"CLI handshake unavailable ({type(error).__name__})"
        finally:
            await self.close()
        self._health = Health(
            status=status,
            binary=binary,
            version=version,
            supported=supported,
            auth_kind=auth or self._auth_kind,
            isolation_capable=status == "ready",
            diagnostic=diagnostic,
        )
        return self._health

    def _restricted_config(self, spec: EngineSpec, temporary: Path) -> dict[str, Any]:
        filesystem: dict[str, Any] = {
            ":root": "deny",
            ":minimal": "read",
            # The guard replaces TMPDIR with this instance's private 0700 directory.
            ":tmpdir": "write",
            ":slash_tmp": "deny",
            str(spec.project_root): "write" if spec.mode == "dev" else "read",
            str(temporary): "write",
        }
        for name in _DENIED_PARTS:
            filesystem[str(Path.home() / name)] = "deny"
        filesystem[str(Path.home() / "Library/Keychains")] = "deny"
        filesystem[str(Path.home() / ".config")] = "deny"
        # Nested globs protect both native file tools and sandboxed commands.
        filesystem[":workspace_roots"] = {
            ".": "write" if spec.mode == "dev" else "read",
            **{f"**/{name}": "deny" for name in _DENIED_PARTS},
            "**/.env": "deny",
            "**/.env.*": "deny",
            **{f"**/*{suffix}": "deny" for suffix in _DENIED_SUFFIXES},
            **{f"**/{name}": "deny" for name in _DENIED_NAMES},
            **dict.fromkeys(_DENIED_FILES, "deny"),
        }
        return {
            "approval_policy": "on-request",
            "approvals_reviewer": "user",
            "forced_login_method": "chatgpt",
            "default_permissions": _PROFILE,
            "permissions": {
                _PROFILE: {
                    "extends": ":workspace" if spec.mode == "dev" else ":read-only",
                    "filesystem": filesystem,
                    "network": {"enabled": False},
                }
            },
            "mcp_servers": {},
            "web_search": "disabled",
            "features": {"multi_agent": False, "apps": False},
            "shell_environment_policy": {"inherit": "none", "set": {"TMPDIR": str(temporary)}},
        }

    async def _connect(self, spec: EngineSpec, binary: str) -> None:
        if self._temp is None:
            self._temp = tempfile.TemporaryDirectory(prefix="agent-engine-")
        temporary = Path(self._temp.name).resolve()
        self._config = self._restricted_config(spec, temporary)
        command = [binary, "app-server", "--listen", "stdio://", "--strict-config"]
        for key, value in self._config.items():
            command.extend(["-c", key + "=" + _toml(value)])
        session = spec.settings.get("session_id")
        if session:
            locks = get_settings().data_dir / "engine-locks"
            locks.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock = locks / (hashlib.sha256(str(session).encode()).hexdigest() + ".lock")
        else:
            lock = temporary / "probe.lock"
        self._process = await GuardedProcess.start(
            command, cwd=spec.project_root, lock_path=lock, temporary_dir=temporary
        )
        self._reader = asyncio.create_task(self._read())
        init = await self._request(
            "initialize",
            {
                "clientInfo": {"name": "python_agent", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        for field in ("userAgent", "platformFamily", "platformOs", "codexHome"):
            _string(init.get(field))
        await self._send({"method": "initialized", "params": {}})
        account = await self._request("account/read", {"refreshToken": False})
        value = account.get("account")
        kind = value.get("type") if isinstance(value, dict) else None
        self._auth_kind = kind if kind in ("chatgpt", "apiKey") else "unknown"
        if kind != "chatgpt":
            raise PermissionError("Plan login required")
        # Never retain the account packet, email, home or auth details.
        config = await self._request(
            "config/read",
            {
                "cwd": str(spec.project_root),
                "includeLayers": False,
            },
        )
        effective = _object(config.get("config"))
        if any(effective.get(key) != value for key, value in self._config.items()) or (
            effective.get("sandbox_mode") is not None
            or effective.get("sandbox_workspace_write") is not None
        ):
            raise _IsolationError()

    async def open(self, spec: EngineSpec, thread_id: str | None = None) -> ThreadRef:
        health = await self.health()
        if health.status != "ready" or not health.binary:
            raise RuntimeError(f"Engine unavailable: {health.diagnostic}")
        if self._process is not None:
            raise RuntimeError("Engine already open")
        root = spec.project_root.resolve(strict=True)
        if root != spec.project_root or not root.is_dir() or spec.mode not in ("read", "dev"):
            raise RuntimeError("Canonical project root and supported mode required")
        self._spec = spec
        try:
            async with asyncio.timeout(self.profile.timeouts.get("open", 30)):
                await self._connect(spec, health.binary)
                params: dict[str, Any] = {
                    "cwd": str(root),
                    "model": self.profile.model,
                    "approvalPolicy": "on-request",
                    "approvalsReviewer": "user",
                    "developerInstructions": spec.instructions,
                }
                if thread_id is not None:
                    params["threadId"] = _identity(thread_id)
                result = await self._request(
                    "thread/resume" if thread_id else "thread/start", params
                )
                expected = "workspaceWrite" if spec.mode == "dev" else "readOnly"
                sandbox = _object(result.get("sandbox"))
                if (
                    result.get("approvalPolicy") != "on-request"
                    or result.get("approvalsReviewer") != "user"
                    or result.get("cwd") != str(root)
                    or result.get("model") != self.profile.model
                    or sandbox.get("type") != expected
                    or sandbox.get("networkAccess", False) is not False
                    or _object(result.get("activePermissionProfile")).get("id") != _PROFILE
                ):
                    raise _IsolationError()
                native_id = _identity(_object(result.get("thread")).get("id"))
                if thread_id is not None and native_id != thread_id:
                    raise _ProtocolError("Resume changed thread identity")
                self._thread = native_id
                return ThreadRef(native_id)
        except BaseException as error:
            if isinstance(error, (PermissionError, _IsolationError)):
                self._health = Health(
                    status="unavailable",
                    binary=health.binary,
                    version=health.version,
                    supported=True,
                    auth_kind=self._auth_kind,
                    diagnostic="Authentication or required isolation unavailable",
                )
            elif isinstance(error, _ProtocolError):
                self._degraded(str(error))
            await self.close()
            raise

    async def _send(self, packet: dict[str, Any]) -> None:
        raw = json.dumps(packet, ensure_ascii=False).encode() + b"\n"
        if len(raw) > MAX_LINE_BYTES:
            raise _ProtocolError("Outgoing protocol line exceeds 1 MiB")
        async with self._write_lock:
            process = self._process
            if process is None or process.stdin is None or process.returncode is not None:
                raise _ProtocolError("CLI channel closed")
            process.stdin.write(raw)
            await process.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._responses[request_id] = future
        try:
            async with asyncio.timeout(self.profile.timeouts.get("request", 10)):
                await self._send({"id": request_id, "method": method, "params": params})
                return await future
        finally:
            self._responses.pop(request_id, None)

    async def _read(self) -> None:
        assert self._process is not None
        try:
            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    raise _ProtocolError("CLI EOF before reliable completion")
                packet = decode_line(raw)
                if "method" in packet:
                    _string(packet["method"])
                    _object(packet.get("params", {}))
                    if "id" in packet and self._submission is None:
                        await self._send(
                            {
                                "id": packet["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "Server requests require an active turn",
                                },
                            }
                        )
                        raise _ProtocolError("Unexpected server request during handshake")
                    await self._incoming.put(packet)
                else:
                    request_id = packet.get("id")
                    if type(request_id) is not int or request_id not in self._responses:
                        raise _ProtocolError("Unexpected RPC response identity")
                    future = self._responses[request_id]
                    if future.done():
                        raise _ProtocolError("Duplicate RPC response")
                    if "error" in packet:
                        future.set_exception(_ProtocolError("CLI refused protocol request"))
                    else:
                        future.set_result(_object(packet.get("result")))
        except (ValueError, OSError, _ProtocolError) as error:
            failure = _ProtocolError(
                str(error)
                if isinstance(error, _ProtocolError)
                else "Malformed or oversize CLI packet"
            )
            for future in self._responses.values():
                if not future.done():
                    future.set_exception(failure)
            await self._incoming.put(failure)

    def _check_context(self, params: dict[str, Any]) -> None:
        if params.get("threadId", self._thread) != self._thread:
            raise _ProtocolError("Foreign thread notification")
        if params.get("turnId", self._turn) != self._turn:
            raise _ProtocolError("Foreign turn notification")

    async def turn(
        self, thread: ThreadRef, input: str, submission_id: str
    ) -> AsyncIterator[EngineEvent]:
        if thread.native_id != self._thread or self._turn is not None or self._stopping:
            raise RuntimeError("Engine thread is not ready")
        self._submission = submission_id
        try:
            params: dict[str, Any] = {
                "threadId": self._thread,
                "input": [{"type": "text", "text": input, "text_elements": []}],
            }
            if self.profile.effort is not None:
                params["effort"] = self.profile.effort
            response = await self._request("turn/start", params)
            self._turn = _identity(_object(response.get("turn")).get("id"))
            while True:
                packet = await self._incoming.get()
                if isinstance(packet, Exception):
                    raise packet
                method = _string(packet.get("method"))
                payload = _object(packet.get("params", {}))
                if "id" in packet:
                    events = await self._server_request(packet)
                else:
                    events = self._notification(method, payload)
                for event in events:
                    if isinstance(event, DoneEvent):
                        self._terminal = True
                    yield event
                    if isinstance(event, DoneEvent):
                        return
        except (ValueError, OSError, TimeoutError, _ProtocolError) as error:
            self._degraded("Protocol changed or CLI channel failed")
            yield ErrorEvent(
                **self._envelope(),
                code="protocol_error",
                message=str(error) if isinstance(error, _ProtocolError) else "CLI protocol failed",
                retryable=False,
            )
            yield DoneEvent(**self._envelope(), status="error", reason="CLI protocol failed")
        finally:
            # Per-turn process lifetime, including consumer cancellation and early aclose.
            await self.close()

    def _notification(self, method: str, params: dict[str, Any]) -> list[EngineEvent]:
        self._check_context(params)
        if method.startswith(("item/", "thread/tokenUsage/", "turn/")) and method in {
            "item/agentMessage/delta",
            "item/reasoning/summaryTextDelta",
            "item/reasoning/textDelta",
            "item/started",
            "item/completed",
            "thread/tokenUsage/updated",
            "turn/diff/updated",
            "turn/started",
            "turn/completed",
        }:
            _identity(params.get("threadId"))
            if method not in ("turn/started", "turn/completed"):
                _identity(params.get("turnId"))
        if method == "item/agentMessage/delta":
            item = _identity(params.get("itemId"))
            if "text:" + item in self._finals:
                return []
            return [
                TextEvent(**self._envelope(item), text=_string(params.get("delta")), phase="delta")
            ]
        if method in ("item/reasoning/summaryTextDelta", "item/reasoning/textDelta"):
            item = _identity(params.get("itemId"))
            if "reasoning:" + item in self._finals:
                return []
            channel = "summary" if "summary" in method else "content"
            _integer(params.get("summaryIndex" if channel == "summary" else "contentIndex"))
            delta = _string(params.get("delta"))
            return [ReasoningEvent(**self._envelope(item), text=delta, phase="delta")]
        if method in ("item/started", "item/completed"):
            return self._item(_object(params.get("item")), method == "item/completed")
        if method == "turn/diff/updated":
            patch = _string(params.get("diff"))
            paths = re.findall(r"^diff --git a/.*? b/(.+)$", patch, flags=re.MULTILINE)
            return [DiffEvent(**self._envelope(), paths=paths, patch=patch, status="proposed")]
        if method == "thread/tokenUsage/updated":
            usage = _object(_object(params.get("tokenUsage")).get("total"))
            return [
                UsageEvent(
                    **self._envelope(),
                    input_tokens=_integer(usage.get("inputTokens")),
                    output_tokens=_integer(usage.get("outputTokens")),
                    cache_tokens=_integer(usage.get("cachedInputTokens")),
                    reported_cost=None,
                    estimated_cost=True,
                )
            ]
        if method == "account/rateLimits/updated":
            self._limits = self._parse_limits(_object(params.get("rateLimits")))
            return [LimitsEvent(**self._envelope(), limits=self._limits)]
        if method in ("turn/started", "turn/completed"):
            turn = _object(params.get("turn"))
            if _identity(turn.get("id")) != self._turn:
                raise _ProtocolError("Foreign completed turn")
            status = _string(turn.get("status"))
            if method == "turn/started":
                if status != "inProgress":
                    raise _ProtocolError("Invalid started turn")
                return []
            if status not in ("completed", "failed", "interrupted"):
                raise _ProtocolError("Unknown terminal turn status")
            outcome: Literal["completed", "error", "interrupted"] = (
                "completed"
                if status == "completed"
                else "error"
                if status == "failed"
                else "interrupted"
            )
            return [
                DoneEvent(
                    **self._envelope(),
                    status=outcome,
                    reason="CLI turn failed" if status == "failed" else None,
                )
            ]
        if method == "error":
            message = _string(_object(params.get("error")).get("message"))
            return [
                ErrorEvent(
                    **self._envelope(), code="engine_error", message=message, retryable=False
                )
            ]
        # Optional extensions may be ignored. Required known payloads are checked above.
        return []

    def _item(self, item: dict[str, Any], completed: bool) -> list[EngineEvent]:
        identity, kind = _identity(item.get("id")), _string(item.get("type"))
        key = kind + ":" + identity
        if key in self._finals:
            return []
        if completed:
            self._finals.add(key)
        if kind == "agentMessage" and completed:
            self._finals.add("text:" + identity)
            return [
                TextEvent(**self._envelope(identity), text=_string(item.get("text")), phase="final")
            ]
        if kind == "reasoning" and completed:
            summary = _strings(item.get("summary", []))
            content = _strings(item.get("content", []))
            return [
                ReasoningEvent(
                    **self._envelope(identity), text="".join(summary + content), phase="final"
                )
            ]
        if kind not in ("commandExecution", "fileChange"):
            return []
        status = _string(item.get("status"))
        if status not in ("inProgress", "completed", "failed", "declined"):
            raise _ProtocolError("Unknown tool item status")
        events: list[EngineEvent] = []
        file_diff: tuple[list[str], str] | None = None
        if kind == "commandExecution":
            arguments: dict[str, Any] = {
                "command": _string(item.get("command")),
                "cwd": _string(item.get("cwd")),
            }
            output = item.get("aggregatedOutput") or ""
            _string(output)
        else:
            changes = item.get("changes")
            if not isinstance(changes, list):
                raise _ProtocolError("Invalid file changes")
            self._changes[identity] = [_object(change) for change in changes]
            paths = [_string(change.get("path")) for change in changes]
            patch = "\n".join(_string(change.get("diff")) for change in changes)
            arguments = {"changes": changes}
            output = patch
            file_diff = (paths, patch)
        if identity not in self._calls:
            self._calls.add(identity)
            events.insert(
                0,
                ToolCallEvent(
                    **self._envelope(identity),
                    name="command" if kind == "commandExecution" else "file",
                    input=arguments,
                    status="running",
                ),
            )
        if file_diff is not None:
            paths, patch = file_diff
            events.append(
                DiffEvent(
                    **self._envelope(identity),
                    paths=paths,
                    patch=patch,
                    status="applied" if status == "completed" else "proposed",
                )
            )
        if completed:
            events.append(
                ToolResultEvent(
                    **self._envelope(identity),
                    related_item_id=identity,
                    output=output,
                    success=status == "completed" and item.get("exitCode", 0) in (0, None),
                )
            )
        return events

    def _allowed_path(self, raw: str, write: bool = False) -> bool:
        if self._spec is None or (write and self._spec.mode != "dev"):
            return False
        root = self._spec.project_root
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.resolve().relative_to(root)
        except (ValueError, OSError, RuntimeError):
            return False
        return not (
            any(
                part in _DENIED_PARTS
                or part in _DENIED_NAMES
                or part.startswith(".env")
                or part.endswith(_DENIED_SUFFIXES)
                for part in relative.parts
            )
            or relative.as_posix() in _DENIED_FILES
        )

    def _grant_path(self, raw: str, write: bool) -> bool:
        if not self._allowed_path(raw, write):
            return False
        assert self._spec is not None
        path = Path(raw)
        if not path.is_absolute():
            path = self._spec.project_root / path
        # A broad directory grant cannot express exclusions for secret descendants.
        # Keep it empty rather than reopening those paths for the rest of the turn.
        return not path.is_dir()

    def _intersection(self, requested: dict[str, Any]) -> dict[str, Any]:
        allowed: dict[str, Any] = {}
        fs = requested.get("fileSystem")
        if fs is None:
            return allowed
        fs = _object(fs)
        result: dict[str, Any] = {}
        for operation in ("read", "write"):
            if fs.get(operation) is not None:
                paths = _strings(fs[operation])
                safe = [p for p in paths if self._grant_path(p, operation == "write")]
                if safe:
                    result[operation] = safe
        if fs.get("entries") is not None:
            entries = fs["entries"]
            if not isinstance(entries, list):
                raise _ProtocolError("Invalid filesystem permissions")
            safe_entries = []
            for entry in entries:
                entry = _object(entry)
                path = _object(entry.get("path"))
                access = entry.get("access")
                if (
                    access in ("read", "write")
                    and path.get("type") == "path"
                    and self._grant_path(_string(path.get("path")), access == "write")
                ):
                    safe_entries.append(entry)
            if safe_entries:
                result["entries"] = safe_entries
        if result:
            allowed["fileSystem"] = result
        return allowed

    def _safe_file(self, item_id: str) -> bool:
        changes = self._changes.get(item_id)
        return bool(changes) and all(
            self._allowed_path(_string(change.get("path")), write=True) for change in changes or []
        )

    async def _server_request(self, packet: dict[str, Any]) -> list[EngineEvent]:
        native_id = packet["id"]
        if type(native_id) not in (str, int):
            raise _ProtocolError("Invalid server request identity")
        request_id = _identity(str(native_id))
        method = _string(packet.get("method"))
        params = _object(packet.get("params"))
        self._check_context(params)
        kind = _APPROVALS.get(method)
        if kind:
            _identity(params.get("threadId"))
            _identity(params.get("turnId"))
            _integer(params.get("startedAtMs"))
            item = _identity(params.get("itemId"))
            if request_id in self._seen_requests:
                raise _ProtocolError("Duplicate server request")
            self._seen_requests.add(request_id)
            if self._stopping:
                await self._send(
                    {
                        "id": native_id,
                        "result": (
                            {"permissions": {}} if kind == "permissions" else {"decision": "cancel"}
                        ),
                    }
                )
                return []
            if kind == "file" and (params.get("grantRoot") or not self._safe_file(item)):
                await self._send({"id": native_id, "result": {"decision": "decline"}})
                return []
            safe = dict(params)
            if kind == "permissions":
                safe["permissions"] = self._intersection(_object(params.get("permissions")))
            # With on-request approvals a sandboxed command runs without a request. A request
            # without additionalPermissions asks to run outside the sandbox: always decline it.
            elif kind == "command" and (
                (params.get("cwd") and not self._allowed_path(_string(params["cwd"])))
                or not isinstance(params.get("command"), str)
                or params.get("additionalPermissions") is None
                or params.get("networkApprovalContext")
                or (
                    params.get("additionalPermissions") is not None
                    and self._intersection(_object(params["additionalPermissions"]))
                    != params["additionalPermissions"]
                )
            ):
                await self._send({"id": native_id, "result": {"decision": "decline"}})
                return []
            self._pending[request_id] = (native_id, kind, safe)
            return [
                ApprovalEvent(
                    **self._envelope(item),
                    request_id=request_id,
                    action=kind,
                    parameters=safe,
                    scope="turn" if kind == "permissions" else "request",
                )
            ]
        if method == "item/tool/requestUserInput":
            await self._send({"id": native_id, "result": {"answers": {}}})
            await self.interrupt(self._turn or "")
            return [UserInputEvent(**self._envelope(params.get("itemId")), request_id=request_id)]
        if method == "mcpServer/elicitation/request":
            await self._send({"id": native_id, "result": {"action": "decline"}})
            return []
        await self._send(
            {
                "id": native_id,
                "error": {
                    "code": -32601,
                    "message": "Unsupported server request",
                },
            }
        )
        self._degraded("Unknown server request refused")
        await self.interrupt(self._turn or "")
        raise _ProtocolError("Unknown server request refused")

    async def answer(self, request_id: str, decision: Literal["allow", "deny"]) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        native_id, kind, params = pending
        if kind == "permissions":
            result = (
                {"permissions": self._intersection(params["permissions"]), "scope": "turn"}
                if decision == "allow" and not self._stopping
                else {"permissions": {}}
            )
        else:
            allowed = decision == "allow" and not self._stopping
            if allowed and kind == "file":
                allowed = self._safe_file(_identity(params.get("itemId")))
            result = {
                "decision": "cancel" if self._stopping else "accept" if allowed else "decline"
            }
        await self._send({"id": native_id, "result": result})

    async def cancel_approval(self, request_id: str) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            native_id, kind, _ = pending
            await self._send(
                {
                    "id": native_id,
                    "result": (
                        {"permissions": {}} if kind == "permissions" else {"decision": "cancel"}
                    ),
                }
            )

    async def answer_user_input(self, request_id: str) -> None:
        # Already answered by the decoder; repeated runner delivery is inert.
        return None

    async def interrupt(self, turn_id: str) -> None:
        self._stopping = True
        for request_id in list(self._pending):
            await self.cancel_approval(request_id)
        if self._process is not None and self._thread is not None:
            await self._request(
                "turn/interrupt",
                {
                    "threadId": self._thread,
                    "turnId": self._turn or turn_id,
                },
            )

    @staticmethod
    def _parse_limits(snapshot: dict[str, Any]) -> Limits:
        windows = []
        for name in ("primary", "secondary"):
            value = snapshot.get(name)
            if value is None:
                continue
            value = _object(value)
            percent = _integer(value.get("usedPercent"))
            if percent > 100:
                raise _ProtocolError("Invalid rate limit percentage")
            reset = value.get("resetsAt")
            duration = value.get("windowDurationMins")
            windows.append(
                LimitWindow(
                    name=name,
                    used_share=percent / 100,
                    reset_at=datetime.fromtimestamp(_integer(reset), UTC)
                    if reset is not None
                    else None,
                    duration_seconds=_integer(duration) * 60 if duration is not None else None,
                    status="exhausted" if percent >= 100 else "available",
                )
            )
        status = "unknown" if not windows else "available"
        if snapshot.get("rateLimitReachedType") or any(w.status == "exhausted" for w in windows):
            status = "exhausted"
        return Limits(windows=windows, status=status, observed_at=datetime.now(UTC))

    async def read_limits(self) -> Limits | None:
        health = await self.health()
        if health.status != "ready":
            return self._limits
        own_process = self._process is None
        try:
            if own_process:
                assert health.binary is not None
                self._temp = tempfile.TemporaryDirectory(prefix="agent-engine-")
                await self._connect(
                    EngineSpec(Path(self._temp.name).resolve(), "read", "", {}), health.binary
                )
            result = await self._request("account/rateLimits/read", {})
            self._limits = self._parse_limits(_object(result.get("rateLimits")))
            return self._limits
        except (ValueError, OSError, TimeoutError, _ProtocolError):
            self._degraded("Rate limits protocol unavailable")
            return self._limits
        finally:
            if own_process:
                await self.close()

    async def close(self) -> None:
        if self._process is not None and self._turn and not self._terminal and not self._stopping:
            try:
                async with asyncio.timeout(2):
                    await self.interrupt(self._turn)
            except (OSError, RuntimeError, TimeoutError):
                pass
        for future in self._responses.values():
            if not future.done():
                future.cancel()
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        if self._process is not None:
            process, self._process = self._process, None
            try:
                await process.close()
            finally:
                # Native stderr/auth errors never reach logs or port diagnostics.
                process.stderr_tail = b""
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._incoming = asyncio.Queue(128)
        self._pending.clear()
