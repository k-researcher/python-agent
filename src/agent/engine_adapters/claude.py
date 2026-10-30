"""Subscription CLI stream-json transport pinned to the 2.1.311 findings.

The recorded initialize response does not attest effective sandbox/file-tool
policy. Until a verified pre-dispatch attestation exists, health fails closed.
There is deliberately no config switch to bypass that gate. No credentials are
read here; the unmodified CLI owns authentication and native history.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from agent.config import EngineProfileSettings, get_settings
from agent.engine import (
    ApprovalEvent,
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
)
from agent.engine_process import MAX_LINE_BYTES, GuardedProcess, decode_line

_DEFAULT_RANGE = ">=2.1.311,<2.1.312"
_FILE_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"}
_TOOLS = _FILE_TOOLS | {"Bash"}
_WRITES = {"Write", "Edit", "NotebookEdit"}
_DENIED_PARTS = {
    ".git",
    ".agents",
    ".codex",
    ".claude",
    ".aws",
    ".ssh",
    ".gnupg",
    ".azure",
    ".config",
}
_DENIED_NAMES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    "auth.json",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    ".claude.json",
}
_DENIED_FILES = {"config/models.yaml", "config/sources.yaml"}
_DENIED_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_MAX_ITEMS = 4096


class _ProtocolError(RuntimeError):
    """Static diagnostics only; never expose auth packets or native stderr."""


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
    if not 1 <= len(text) <= 200 or any(ord(c) < 32 for c in text):
        raise _ProtocolError("Invalid protocol identity")
    return text


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise _ProtocolError("Expected nonnegative protocol integer")
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise _ProtocolError("Expected protocol list")
    return value


def _supported(version: str, setting: Any) -> bool:
    match = re.fullmatch(r">=(\d+\.\d+\.\d+),<(\d+\.\d+\.\d+)", str(setting))
    if not match or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return False
    return (
        tuple(map(int, match[1].split(".")))
        <= tuple(map(int, version.split(".")))
        < tuple(map(int, match[2].split(".")))
    )


class Engine:
    def __init__(self, profile: EngineProfileSettings) -> None:
        self.profile = profile
        self._health: Health | None = None
        self._process: GuardedProcess | None = None
        self._reader: asyncio.Task[None] | None = None
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        self._incoming: asyncio.Queue[dict[str, Any] | Exception] = asyncio.Queue(128)
        self._responses: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._spec: EngineSpec | None = None
        self._thread: str | None = None
        self._submission: str | None = None
        self._seq = 0
        self._stopping = False
        self._finishing = False
        self._terminal = False
        self._limits: Limits | None = None
        self._pending: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self._denied: set[str] = set()
        self._seen_requests: set[str] = set()
        self._finals: set[str] = set()
        self._calls: set[str] = set()
        self._results: set[str] = set()
        self._message: str | None = None
        self._blocks: dict[int, tuple[str, str]] = {}
        self._auth_kind: str | None = None

    def _envelope(self, item: str | None = None) -> dict[str, Any]:
        self._seq += 1
        return dict(
            submission_id=self._submission,
            thread_id=self._thread,
            turn_id=self._submission,
            item_id=item,
            seq=self._seq,
        )

    def _unhealthy(self, status: Literal["degraded", "unavailable"], diagnostic: str) -> None:
        previous = self._health or Health(status=status)
        self._health = Health(
            status=status,
            binary=previous.binary,
            version=previous.version,
            supported=previous.supported,
            auth_kind=self._auth_kind,
            isolation_capable=False,
            diagnostic=diagnostic,
        )

    def _temporary(self) -> Path:
        if self._temp is None:
            self._temp = tempfile.TemporaryDirectory(prefix="agent-engine-")
        return Path(self._temp.name).resolve()

    async def _probe(self, command: list[str]) -> bytes:
        root = self._temporary()
        self._process = await GuardedProcess.start(
            command, cwd=root, lock_path=root / "probe.lock", temporary_dir=root
        )
        raw = await self._process.stdout.readline()
        if len(raw) > MAX_LINE_BYTES:
            raise _ProtocolError("Oversize CLI probe response")
        # Parse the answer before cleanup: a slow reap must not erase a known
        # unsupported version or change a known auth refusal into a timeout.
        # health's finally always reaps, including cancellation during a probe.
        return raw

    async def _finish_probe(self) -> None:
        assert self._process is not None
        process = self._process
        try:
            await process.close()
        finally:
            process.stderr_tail = b""
        # Retain ownership if cleanup was cancelled, so health's finally retries
        # the bounded reap instead of losing the only process reference.
        self._process = None

    async def health(self) -> Health:
        if self._health is not None:
            return self._health
        binary = shutil.which(self.profile.binary)
        if binary is None:
            self._health = Health(status="unavailable", diagnostic="CLI binary unavailable")
            return self._health
        version, supported = None, False
        status: Literal["ready", "degraded", "unavailable"] = "unavailable"
        diagnostic = None
        try:
            async with asyncio.timeout(self.profile.timeouts.get("open", 30)):
                raw = await self._probe([binary, "--version"])
                match = re.fullmatch(r"([^\s]+) \(Claude Code\)", raw.decode("utf-8").strip())
                if match is None:
                    raise _ProtocolError("Invalid CLI version response")
                version = match[1]
                if not _supported(
                    version, self.profile.adapter_settings.get("supported_versions", _DEFAULT_RANGE)
                ):
                    raise _ProtocolError("Unsupported CLI version")
                supported = True
                await self._finish_probe()
                raw = await self._probe([binary, "auth", "status", "--json"])
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError) as error:
                    raise _ProtocolError("Invalid auth status JSON") from error
                auth = value if isinstance(value, dict) else {}
                kind = auth.get("authMethod")
                self._auth_kind = (
                    "subscription"
                    if kind == "claude.ai"
                    else "api_key"
                    if kind == "api_key"
                    else "token"
                    if kind == "oauth_token"
                    else "unknown"
                )
                if (
                    auth.get("loggedIn") is not True
                    or kind != "claude.ai"
                    or auth.get("apiProvider") != "firstParty"
                    or auth.get("subscriptionType") not in ("pro", "max")
                ):
                    raise PermissionError("Subscription login required")
                # Retain only the auth kind, never account details or the raw packet.
                del auth, raw
                await self._finish_probe()
                root = self._temporary()
                await self._connect(EngineSpec(root, "read", "", {}), binary)
                status = "ready"
        except _IsolationError:
            diagnostic = (
                "Required isolation unavailable: initialize does not attest "
                "effective sandbox and file-tool policy"
            )
        except PermissionError:
            diagnostic = "Subscription login required; API, token or unknown authentication refused"
        except (_ProtocolError, ValueError, UnicodeError):
            status, diagnostic = "degraded", "Unsupported version or malformed CLI protocol"
        except (OSError, RuntimeError, TimeoutError) as error:
            diagnostic = f"CLI handshake unavailable ({type(error).__name__})"
        finally:
            await self.close()
        self._health = Health(
            status=status,
            binary=binary,
            version=version,
            supported=supported,
            auth_kind=self._auth_kind,
            isolation_capable=status == "ready",
            diagnostic=diagnostic,
        )
        return self._health

    @staticmethod
    def _verify_isolation(response: dict[str, Any]) -> None:
        # The ordinary initialize response contains commands/models/output styles,
        # NOT effective sandbox, protected paths, hooks or file-tool confinement.
        # Neither a successful ack, requested settings nor tool count proves them.
        # Do not invent an effectiveSettings field, trust an operator boolean, or
        # spend a model turn probing. A verified protocol extension is required.
        raise _IsolationError()

    def _restricted_settings(self, spec: EngineSpec, temporary: Path) -> dict[str, Any]:
        root = str(spec.project_root)
        denied = [str(Path.home() / name) for name in sorted(_DENIED_PARTS | _DENIED_NAMES)]
        denied += [str(Path.home() / "Library/Keychains")]
        denied += [
            str(spec.project_root / "**" / name) for name in sorted(_DENIED_PARTS | _DENIED_NAMES)
        ]
        denied += [
            str(spec.project_root / "**" / ".env*"),
            *[str(spec.project_root / p) for p in sorted(_DENIED_FILES)],
        ]
        denied += [str(spec.project_root / "**" / ("*" + suffix)) for suffix in _DENIED_SUFFIXES]
        rules = [f"{tool}(/{path}/**)" for tool in sorted(_FILE_TOOLS) for path in denied]
        # Whole-file patterns matter too (a trailing /** alone misses files).
        rules += [f"{tool}(/{path})" for tool in sorted(_FILE_TOOLS) for path in denied]
        rules += [
            "Agent",
            "Task",
            "WebFetch",
            "WebSearch",
            "Skill",
            "NotebookEdit",
            "EnterWorktree",
            "ExitWorktree",
        ]
        if spec.mode == "read":
            rules += sorted(_WRITES)
        settings: dict[str, Any] = {
            "sandbox": {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "excludedCommands": [],
                "filesystem": {
                    "allowWrite": [root] if spec.mode == "dev" else [],
                    "denyWrite": denied
                    + ["/tmp", "/private/tmp"]
                    + ([root] if spec.mode == "read" else []),
                    "denyRead": ["/", *denied],
                    "allowRead": [
                        root,
                        str(temporary),
                        "/usr",
                        "/bin",
                        "/sbin",
                        "/lib",
                        "/lib64",
                        "/System/Library",
                        "/Library/Apple",
                        "/dev/null",
                        "/dev/urandom",
                    ],
                },
                "network": {
                    "allowedDomains": [],
                    "allowLocalBinding": False,
                    "allowAllUnixSockets": False,
                    "allowUnixSockets": [],
                },
            },
            # Every file-tool operation must also cross our canonical path gate.
            # Bash is confined by its native OS sandbox, never by parsing shell text.
            "permissions": {
                "defaultMode": "default",
                "allow": [],
                "ask": sorted(_FILE_TOOLS),
                "deny": rules,
                "additionalDirectories": [],
            },
            "hooks": {},
            "disableAllHooks": True,
            "enabledPlugins": {},
        }
        return settings

    async def _connect(self, spec: EngineSpec, binary: str, thread_id: str | None = None) -> None:
        temporary = self._temporary()
        settings = self._restricted_settings(spec, temporary)
        command = [
            binary,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.profile.model,
            "--permission-prompt-tool",
            "stdio",
            "--permission-mode",
            "default",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--settings",
            json.dumps(settings, ensure_ascii=False),
        ]
        if thread_id is not None:
            command.extend(["--resume", _identity(thread_id)])
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
        response = await self._request({"subtype": "initialize", "hooks": None})
        _list(response.get("commands"))
        _list(response.get("models"))
        self._verify_isolation(response)

    async def open(self, spec: EngineSpec, thread_id: str | None = None) -> ThreadRef:
        health = await self.health()
        if health.status != "ready" or not health.binary or not health.isolation_capable:
            raise RuntimeError(f"Engine unavailable: {health.diagnostic}")
        if self._process is not None or self._stopping:
            raise RuntimeError("Engine already open or stopping")
        root = spec.project_root.resolve(strict=True)
        if root != spec.project_root or not root.is_dir() or spec.mode not in ("read", "dev"):
            raise RuntimeError("Canonical project root and supported mode required")
        self._spec = spec
        try:
            async with asyncio.timeout(self.profile.timeouts.get("open", 30)):
                await self._connect(spec, health.binary, thread_id)
                packet = await self._incoming.get()
                if isinstance(packet, Exception):
                    raise packet
                self._validate_init(packet, spec, thread_id)
                assert self._thread is not None
                return ThreadRef(self._thread)
        except BaseException as error:
            if isinstance(error, (PermissionError, _IsolationError)):
                self._unhealthy("unavailable", "Authentication or required isolation unavailable")
            elif isinstance(error, (ValueError, _ProtocolError, TimeoutError)):
                self._unhealthy("degraded", "Pre-dispatch initialization unavailable or changed")
            await self.close()
            raise

    def _validate_init(
        self, packet: dict[str, Any], spec: EngineSpec, expected: str | None
    ) -> None:
        if packet.get("type") != "system" or packet.get("subtype") != "init":
            raise _ProtocolError("Expected pre-dispatch session initialization")
        if packet.get("apiKeySource") != "none":
            raise PermissionError("Non-subscription initialization refused")
        if (
            packet.get("cwd") != str(spec.project_root)
            or packet.get("permissionMode") != "default"
            or packet.get("model") != self.profile.model
            or packet.get("mcp_servers") != []
            or packet.get("plugins") != []
            or packet.get("skills", []) != []
            or packet.get("slash_commands", []) != []
        ):
            raise _IsolationError()
        tools = _list(packet.get("tools"))
        if not tools or any(not isinstance(t, str) or t not in _TOOLS for t in tools):
            raise _IsolationError()
        native = _identity(packet.get("session_id"))
        if expected is not None and native != expected:
            raise _ProtocolError("Resume changed native session identity")
        self._thread = native

    async def _send(self, packet: dict[str, Any]) -> None:
        async with self._write_lock:
            await self._write_packet(packet)

    async def _write_packet(self, packet: dict[str, Any]) -> None:
        """Called with the writer lock held; no yield before stdin.write."""
        raw = json.dumps(packet, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(raw) > MAX_LINE_BYTES:
            raise _ProtocolError("Outgoing protocol line exceeds 1 MiB")
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise _ProtocolError("CLI channel closed")
        process.stdin.write(raw)
        await process.stdin.drain()

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        identity = "host-" + str(self._next_id)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._responses[identity] = future
        try:
            async with asyncio.timeout(self.profile.timeouts.get("request", 10)):
                await self._send(
                    {"type": "control_request", "request_id": identity, "request": request}
                )
                return await future
        finally:
            self._responses.pop(identity, None)

    async def _reply(self, identity: str, value: dict[str, Any]) -> None:
        await self._send(
            {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": identity, "response": value},
            }
        )

    async def _deny(self, identity: str) -> None:
        await self._reply(identity, {"behavior": "deny", "message": "Permission denied"})

    async def _read(self) -> None:
        assert self._process is not None
        process = self._process
        try:
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    raise _ProtocolError("CLI EOF without reliable completion")
                packet = decode_line(raw)
                kind = _string(packet.get("type"))
                if kind == "control_response":
                    response = _object(packet.get("response"))
                    identity = _identity(response.get("request_id"))
                    future = self._responses.get(identity)
                    if future is None or future.done():
                        raise _ProtocolError("Unexpected or duplicate control response")
                    if response.get("subtype") != "success":
                        future.set_exception(_ProtocolError("CLI refused control request"))
                    else:
                        future.set_result(_object(response.get("response")))
                elif kind == "control_request" and (self._submission is None or self._stopping):
                    await self._deny(_identity(packet.get("request_id")))
                    if not self._stopping:
                        raise _ProtocolError("Unexpected control request before dispatch")
                else:
                    await self._incoming.put(packet)
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

    async def turn(
        self, thread: ThreadRef, input: str, submission_id: str
    ) -> AsyncIterator[EngineEvent]:
        if (
            thread.native_id != self._thread
            or self._submission is not None
            or self._stopping
            or self._process is None
        ):
            raise RuntimeError("Engine thread is not ready")
        self._submission = _identity(submission_id)
        assert self._spec is not None
        text = self._spec.instructions + "\n\n" + input if self._spec.instructions else input
        try:
            await self._send({"type": "user", "message": {"role": "user", "content": text}})
            while not self._terminal:
                packet = await self._incoming.get()
                if isinstance(packet, Exception):
                    raise packet
                for event in await self._map(packet):
                    if isinstance(event, DoneEvent):
                        self._terminal = True
                    yield event
        except (ValueError, OSError, OverflowError, _ProtocolError, TimeoutError):
            self._unhealthy("degraded", "Malformed, incompatible or closed stream-json protocol")
            with contextlib.suppress(OSError, RuntimeError, TimeoutError):
                async with asyncio.timeout(1):
                    await self.interrupt(submission_id)
            yield ErrorEvent(
                **self._envelope(),
                code="engine_protocol",
                message="CLI protocol unavailable; result may be unknown",
                retryable=False,
            )
            if not self._terminal:
                self._terminal = True
                yield DoneEvent(
                    **self._envelope(), status="error", reason="CLI protocol unavailable"
                )

    async def _map(self, packet: dict[str, Any]) -> list[EngineEvent]:
        if packet.get("session_id", self._thread) != self._thread:
            raise _ProtocolError("Foreign session packet")
        kind = _string(packet.get("type"))
        if packet.get("parent_tool_use_id") is not None:
            raise _ProtocolError("Unexpected delegated tool stream")
        if kind == "control_request":
            return await self._approval(packet)
        if kind == "assistant":
            return self._assistant(_object(packet.get("message")))
        if kind == "stream_event":
            return self._stream(_object(packet.get("event")))
        if kind == "user":
            return self._tool_results(_object(packet.get("message")))
        if kind == "rate_limit_event":
            self._limits = self._parse_limits(_object(packet.get("rate_limit_info")))
            return [LimitsEvent(**self._envelope(), limits=self._limits)]
        if kind == "result":
            events = self._result(packet)
            # Revoke decisions before yielding usage or done. A queued allow
            # must also see terminal state when it acquires the writer lock.
            self._finishing = True
            for identity in list(self._pending):
                await self.cancel_approval(identity)
            return events
        if kind == "system" and packet.get("subtype") == "init":
            raise _ProtocolError("Repeated session initialization")
        # Unknown optional notifications can be ignored, never permission requests.
        return []

    def _bounded(self) -> None:
        if (
            max(
                len(self._finals),
                len(self._calls),
                len(self._results),
                len(self._seen_requests),
                len(self._blocks),
            )
            >= _MAX_ITEMS
        ):
            raise _ProtocolError("Too many protocol items")

    def _assistant(self, message: dict[str, Any]) -> list[EngineEvent]:
        identity = _identity(message.get("id"))
        if message.get("role") != "assistant":
            raise _ProtocolError("Invalid assistant role")
        events: list[EngineEvent] = []
        for index, raw in enumerate(_list(message.get("content"))):
            self._bounded()
            block = _object(raw)
            kind = _string(block.get("type"))
            item = f"{identity}:{index}"
            if kind in ("text", "thinking"):
                key = kind + ":" + item
                if key not in self._finals:
                    self._finals.add(key)
                    value = _string(block.get("text" if kind == "text" else "thinking"))
                    event_type = TextEvent if kind == "text" else ReasoningEvent
                    events.append(event_type(**self._envelope(item), text=value, phase="final"))
            elif kind == "tool_use":
                events.extend(
                    self._tool_call(
                        _identity(block.get("id")),
                        _string(block.get("name")),
                        _object(block.get("input")),
                    )
                )
            elif kind != "redacted_thinking":
                raise _ProtocolError("Unknown required assistant block")
        return events

    def _stream(self, event: dict[str, Any]) -> list[EngineEvent]:
        kind = _string(event.get("type"))
        if kind == "message_start":
            self._message = _identity(_object(event.get("message")).get("id"))
            self._blocks = {}
        elif kind in ("content_block_start", "content_block_delta"):
            self._bounded()
            if self._message is None:
                raise _ProtocolError("Stream block before message start")
            index = _integer(event.get("index"))
            item = f"{self._message}:{index}"
            if kind == "content_block_start":
                if index in self._blocks:
                    raise _ProtocolError("Duplicate stream block")
                block = _object(event.get("content_block"))
                block_type = _string(block.get("type"))
                self._blocks[index] = (block_type, item)
                if block_type in ("text", "thinking"):
                    value = _string(block.get("text" if block_type == "text" else "thinking"))
                    if value and block_type + ":" + item not in self._finals:
                        event_type = TextEvent if block_type == "text" else ReasoningEvent
                        return [event_type(**self._envelope(item), text=value, phase="delta")]
                elif block_type not in ("tool_use", "redacted_thinking"):
                    raise _ProtocolError("Unknown stream block")
            else:
                block_info = self._blocks.get(index)
                if block_info is None:
                    raise _ProtocolError("Delta before stream block")
                block_type, item = block_info
                delta = _object(event.get("delta"))
                delta_type = _string(delta.get("type"))
                if delta_type in ("text_delta", "thinking_delta"):
                    expected = "text" if delta_type == "text_delta" else "thinking"
                    if block_type != expected:
                        raise _ProtocolError("Mismatched stream delta")
                    if expected + ":" + item not in self._finals:
                        event_type = TextEvent if expected == "text" else ReasoningEvent
                        value = _string(delta.get("text" if expected == "text" else "thinking"))
                        return [event_type(**self._envelope(item), text=value, phase="delta")]
                elif delta_type not in ("signature_delta", "input_json_delta"):
                    raise _ProtocolError("Unknown stream delta")
        elif kind not in ("content_block_stop", "message_delta", "message_stop", "ping"):
            raise _ProtocolError("Unknown required stream event")
        return []

    def _tool_call(self, item: str, name: str, arguments: dict[str, Any]) -> list[EngineEvent]:
        if item in self._calls:
            return []
        self._bounded()
        self._calls.add(item)
        return [
            ToolCallEvent(
                **self._envelope(item), name=name, input=copy.deepcopy(arguments), status="running"
            )
        ]

    def _tool_results(self, message: dict[str, Any]) -> list[EngineEvent]:
        if message.get("role") != "user":
            raise _ProtocolError("Invalid tool result role")
        events: list[EngineEvent] = []
        for raw in _list(message.get("content")):
            block = _object(raw)
            if block.get("type") != "tool_result":
                raise _ProtocolError("Unknown required tool result block")
            item = _identity(block.get("tool_use_id"))
            if item not in self._calls:
                raise _ProtocolError("Tool result without observed call")
            if item in self._results:
                continue
            self._bounded()
            self._results.add(item)
            content = block.get("content")
            if isinstance(content, list):
                values = [_object(p) for p in content]
                if any(p.get("type") != "text" for p in values):
                    raise _ProtocolError("Unsupported tool result content")
                content = "\n".join(_string(p.get("text")) for p in values)
            content = _string(content)
            failed = block.get("is_error", False)
            if type(failed) is not bool:
                raise _ProtocolError("Invalid tool result status")
            events.append(
                ToolResultEvent(
                    **self._envelope(item), related_item_id=item, output=content, success=not failed
                )
            )
        return events

    def _result(self, packet: dict[str, Any]) -> list[EngineEvent]:
        subtype = _string(packet.get("subtype"))
        if subtype not in (
            "success",
            "error_during_execution",
            "error_max_turns",
            "error_max_budget_usd",
            "error_max_structured_output_retries",
        ):
            raise _ProtocolError("Unknown result status")
        if type(packet.get("is_error")) is not bool:
            raise _ProtocolError("Invalid result error status")
        events: list[EngineEvent] = []
        usage = _object(packet.get("usage"))
        cost = packet.get("total_cost_usd")
        if cost is not None and (
            type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0
        ):
            raise _ProtocolError("Invalid reported cost")
        cache = sum(
            _integer(usage.get(k, 0))
            for k in ("cache_read_input_tokens", "cache_creation_input_tokens")
        )
        events.append(
            UsageEvent(
                **self._envelope(),
                input_tokens=_integer(usage.get("input_tokens")),
                output_tokens=_integer(usage.get("output_tokens")),
                cache_tokens=cache,
                reported_cost=Decimal(str(cost)) if cost is not None else None,
                estimated_cost=True,
            )
        )
        status: Literal["completed", "stopped", "error", "interrupted"] = "completed"
        if self._stopping:
            status = "stopped"
        elif packet.get("terminal_reason") == "aborted_streaming":
            status = "interrupted"
        elif packet["is_error"] or subtype != "success":
            status = "error"
        events.append(
            DoneEvent(
                **self._envelope(),
                status=status,
                reason=None if status == "completed" else "CLI turn " + status,
            )
        )
        return events

    def _allowed_path(self, raw: str, write: bool = False) -> bool:
        if self._spec is None or (write and self._spec.mode != "dev"):
            return False
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._spec.project_root / candidate
        try:
            relative = candidate.resolve().relative_to(self._spec.project_root)
        except (ValueError, OSError, RuntimeError):
            return False
        return not (
            any(
                p in _DENIED_PARTS
                or p in _DENIED_NAMES
                or p.startswith(".env")
                or p.endswith(_DENIED_SUFFIXES)
                for p in relative.parts
            )
            or relative.as_posix() in _DENIED_FILES
        )

    def _safe_tool(self, name: str, arguments: dict[str, Any], request: dict[str, Any]) -> bool:
        if (
            request.get("blocked_path")
            or arguments.get("dangerouslyDisableSandbox")
            or request.get("agent_id")
        ):
            return False
        if name == "Bash":
            # A request after a sandbox violation must not authorize an escape.
            # Shell contents are not a substitute for OS confinement.
            cwd = arguments.get("cwd")
            return (
                isinstance(arguments.get("command"), str)
                and not request.get("decision_reason")
                and (cwd is None or (isinstance(cwd, str) and self._allowed_path(cwd)))
            )
        if name not in _FILE_TOOLS:
            return False
        key = (
            "notebook_path"
            if name == "NotebookEdit"
            else "path"
            if name in ("Glob", "Grep")
            else "file_path"
        )
        raw = arguments.get(key, "." if name in ("Glob", "Grep") else None)
        if not isinstance(raw, str) or not self._allowed_path(raw, name in _WRITES):
            return False
        # Pattern-bearing search tools can traverse symlinks or name secrets;
        # deny their requests unless a verified native boundary protects them.
        return name not in ("Glob", "Grep", "NotebookEdit")

    async def _approval(self, packet: dict[str, Any]) -> list[EngineEvent]:
        identity = _identity(packet.get("request_id"))
        request = _object(packet.get("request"))
        if request.get("subtype") != "can_use_tool":
            await self._deny(identity)
            raise _ProtocolError("Unknown control request refused")
        if identity in self._seen_requests:
            await self._deny(identity)
            raise _ProtocolError("Duplicate approval request")
        self._bounded()
        self._seen_requests.add(identity)
        name = _string(request.get("tool_name"))
        arguments = _object(request.get("input"))
        item = _identity(request.get("tool_use_id"))
        events = self._tool_call(item, name, arguments)
        if self._stopping or not self._safe_tool(name, arguments, request):
            await self._deny(identity)
            return events
        self._pending[identity] = (item, name, copy.deepcopy(arguments))
        # Only input crosses the port; suggestions/updates never reach storage.
        return [
            *events,
            ApprovalEvent(
                **self._envelope(item),
                request_id=identity,
                action=name,
                parameters={"input": copy.deepcopy(arguments)},
                scope="request",
            ),
        ]

    async def answer(self, request_id: str, decision: Literal["allow", "deny"]) -> None:
        if decision == "deny" and request_id in self._pending:
            self._denied.add(request_id)
        async with self._write_lock:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                return
            _, name, original = pending
            allowed = (
                decision == "allow"
                and not self._stopping
                and not self._finishing
                and not self._terminal
                and request_id not in self._denied
                and self._safe_tool(name, original, {})
            )
            self._denied.discard(request_id)
            value = (
                {"behavior": "allow", "updatedInput": original}
                if allowed
                else {"behavior": "deny", "message": "Permission denied"}
            )
            # Recheck Stop, TTL and paths after backpressure, immediately before
            # the write. No queued allow may overtake a cancellation.
            await self._write_packet(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": request_id,
                        "response": value,
                    },
                }
            )

    async def cancel_approval(self, request_id: str) -> None:
        await self.answer(request_id, "deny")

    async def answer_user_input(self, request_id: str) -> None:
        await self.cancel_approval(request_id)

    async def interrupt(self, turn_id: str) -> None:
        self._stopping = True
        for identity in list(self._pending):
            await self.cancel_approval(identity)
        if self._process is not None:
            response = await self._request({"subtype": "interrupt"})
            _list(response.get("still_queued"))

    @staticmethod
    def _parse_limits(info: dict[str, Any]) -> Limits:
        status = _string(info.get("status"))
        states = {"allowed": "available", "allowed_warning": "available", "rejected": "exhausted"}
        if status not in states:
            raise _ProtocolError("Unknown rate limit status")
        windows: list[LimitWindow] = []
        values = _object(info.get("unifiedWindows", {}))
        if not values and info.get("rateLimitType") is not None:
            name = _identity(info["rateLimitType"])
            reset = info.get("resetsAt")
            windows.append(
                LimitWindow(
                    name=name,
                    used_share=None,
                    reset_at=datetime.fromtimestamp(_integer(reset), UTC)
                    if reset is not None
                    else None,
                    duration_seconds={"five_hour": 18000, "seven_day": 604800}.get(name),
                    status=states[status],
                )
            )
        for name, raw in values.items():
            value = _object(raw)
            utilization = value.get("utilization")
            if (
                not isinstance(utilization, (int, float))
                or isinstance(utilization, bool)
                or not math.isfinite(utilization)
                or not 0 <= utilization <= 1
            ):
                raise _ProtocolError("Invalid rate limit utilization")
            native_status = _string(value.get("status", status))
            if native_status not in states:
                raise _ProtocolError("Unknown window status")
            reset = value.get("resetsAt")
            windows.append(
                LimitWindow(
                    name=name,
                    used_share=float(utilization),
                    reset_at=datetime.fromtimestamp(_integer(reset), UTC)
                    if reset is not None
                    else None,
                    duration_seconds={"five_hour": 18000, "seven_day": 604800}.get(name),
                    status="exhausted" if utilization == 1 else states[native_status],
                )
            )
        normalized = (
            "exhausted"
            if states[status] == "exhausted" or any(w.status == "exhausted" for w in windows)
            else "available"
            if windows
            else "unknown"
        )
        return Limits(windows=windows, status=normalized, observed_at=datetime.now(UTC))

    async def read_limits(self) -> Limits | None:
        return self._limits

    async def close(self) -> None:
        if self._process is not None:
            if self._submission and not self._terminal and not self._stopping:
                with contextlib.suppress(OSError, RuntimeError, TimeoutError):
                    async with asyncio.timeout(1):
                        await self.interrupt(self._submission)
            else:
                for identity in list(self._pending):
                    with contextlib.suppress(OSError, RuntimeError):
                        await self.cancel_approval(identity)
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
                process.stderr_tail = b""
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._incoming = asyncio.Queue(128)
        self._pending.clear()
        self._denied.clear()
