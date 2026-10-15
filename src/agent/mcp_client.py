"""Stdio client for the Model Context Protocol.

Speak JSON-RPC 2.0 over a child process stdio.
Each message is one line of JSON.
This module is decoupled from the tool registry.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# JSON-RPC 2.0 reserved error codes.
_METHOD_NOT_FOUND = -32601
_MAX_LINE_BYTES = 4 * 1024 * 1024
_MAX_LIST_PAGES = 50


class McpError(Exception):
    """An error returned or raised by the MCP transport."""

    def __init__(self, code: int | None, message: str) -> None:
        """Store the JSON-RPC code and a human-readable message."""
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class McpTool:
    """A tool exposed by the remote MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]


class StdioMcpClient:
    """Talk to an MCP server over its standard streams."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        """Bind to a server command but do not start it.

        Raise ValueError when the command list is empty.
        The default environment keeps only PATH, HOME, and LANG.
        """
        if not command:
            raise ValueError("command must not be empty")
        self._command = list(command)
        self._cwd = cwd
        self._env = dict(env) if env is not None else self._default_env()
        self._request_timeout = request_timeout

        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 1
        self._reader_task: asyncio.Task[None] | None = None
        # Strong references keep the reply tasks alive until they finish.
        self._reply_tasks: set[asyncio.Task[None]] = set()
        # Set when the connection can no longer work; new requests fail at once.
        self._broken: McpError | None = None
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}

    @staticmethod
    def _default_env() -> dict[str, str]:
        """Build a minimal environment for the child process."""
        result: dict[str, str] = {}
        for key in ("PATH", "HOME", "LANG"):
            value = os.environ.get(key)
            if value is not None:
                result[key] = value
        return result

    async def __aenter__(self) -> StdioMcpClient:
        """Start the server and return this client."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close the server process."""
        await self.close()

    async def start(self) -> None:
        """Spawn the server and perform the initialize handshake."""
        if self._process is not None:
            return
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            cwd=str(self._cwd),
            env=self._env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        self._broken = None
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await self._initialize()
        except BaseException:
            # A failed handshake must not leave the server process running.
            await self.close()
            raise

    async def _initialize(self) -> None:
        init_params: dict[str, Any] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "python-agent", "version": "0.1"},
        }
        result = await self._request("initialize", init_params)
        self.server_info = dict(result.get("serverInfo", {}))
        self.capabilities = dict(result.get("capabilities", {}))

        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> list[McpTool]:
        """Fetch all tools, following pagination cursors."""
        tools: list[McpTool] = []
        cursor: str | None = None
        for _ in range(_MAX_LIST_PAGES):
            params: dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            result = await self._request("tools/list", params)
            items = result.get("tools", [])
            if not isinstance(items, list):
                raise McpError(None, "protocol error: tools is not a list")
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise McpError(None, "protocol error: a tool has no name")
                schema = item.get("inputSchema", {})
                tools.append(
                    McpTool(
                        name=item["name"],
                        description=str(item.get("description", "")),
                        input_schema=dict(schema) if isinstance(schema, dict) else {},
                    )
                )
            cursor = result.get("nextCursor")
            if not isinstance(cursor, str):
                break
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a remote tool and return its structured result."""
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        result = await self._request("tools/call", params)
        return dict(result)

    async def close(self) -> None:
        """Terminate the server and release its resources."""
        process = self._process
        if process is None:
            return

        # Signal the server that no more input follows.
        stdin = process.stdin
        if stdin is not None:
            with contextlib.suppress(BrokenPipeError, OSError):
                stdin.close()

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=2.0)

        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)

        # The group can outlive its leader (a launcher that started the real server).
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        # Fail any requests still waiting for a response.
        self._fail_pending(McpError(None, "server closed"))
        self._process = None

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a request and wait for its response."""
        call_id = self._next_id
        self._next_id += 1

        if self._process is None or self._process.stdin is None:
            raise McpError(None, "client is not started")
        if self._broken is not None:
            raise self._broken

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future

        line = json.dumps(
            {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params},
            separators=(",", ":"),
        )
        try:
            self._process.stdin.write((line + "\n").encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, OSError) as exc:
            self._pending.pop(call_id, None)
            if not future.done():
                future.set_exception(McpError(None, f"write failed: {exc}"))
        except asyncio.CancelledError:
            # A cancelled caller must not orphan the pending future.
            self._pending.pop(call_id, None)
            raise

        try:
            result = await asyncio.wait_for(future, timeout=self._request_timeout)
        except TimeoutError:
            self._pending.pop(call_id, None)
            raise McpError(None, "timed out") from None
        except asyncio.CancelledError:
            # Keep the client usable after an external cancellation.
            self._pending.pop(call_id, None)
            raise
        if not isinstance(result, dict):
            raise McpError(None, f"protocol error: result of {method} is not an object")
        return result

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification without waiting for a reply."""
        if self._process is None or self._process.stdin is None:
            raise McpError(None, "client is not started")
        line = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params},
            separators=(",", ":"),
        )
        self._process.stdin.write((line + "\n").encode())
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Read stdout lines and dispatch each JSON-RPC message."""
        buffer = bytearray()
        assert self._process is not None
        assert self._process.stdout is not None
        reader = self._process.stdout
        while True:
            try:
                chunk = await reader.read(65536)
            except asyncio.CancelledError:
                raise
            except OSError as exc:
                self._fail_pending(McpError(None, f"read failed: {exc}"))
                return
            if not chunk:
                break
            buffer.extend(chunk)
            if b"\n" not in buffer and len(buffer) > _MAX_LINE_BYTES:
                # A line without an end must not grow without a limit.
                self._fail_pending(McpError(None, "protocol error: line too long"))
                return
            while b"\n" in buffer:
                idx = buffer.index(b"\n")
                raw = bytes(buffer[:idx])
                del buffer[: idx + 1]
                if len(raw) > _MAX_LINE_BYTES:
                    self._fail_pending(McpError(None, "protocol error: line too long"))
                    return
                line = raw.decode("utf-8", errors="replace")
                self._handle_line(line)

        # The server closed its stdout or exited.
        self._fail_pending(McpError(None, "server closed"))

    def _handle_line(self, line: str) -> None:
        """Parse one JSON line and resolve or reject the matching request."""
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(message, dict):
            return

        message_id = message.get("id")
        method = message.get("method")
        error_data = message.get("error")

        # A server-side request with an id expects a reply.
        if message_id is not None and isinstance(method, str):
            task = asyncio.create_task(self._reply_error(message_id, _METHOD_NOT_FOUND))
            self._reply_tasks.add(task)
            task.add_done_callback(self._reply_tasks.discard)
            return

        if message_id is None:
            # Server notification: ignore.
            return

        if error_data is not None:
            code = error_data.get("code") if isinstance(error_data, dict) else None
            text = error_data.get("message", "error") if isinstance(error_data, dict) else "error"
            self._resolve(message_id, McpError(code, str(text)))
            return

        result = message.get("result", {})
        self._resolve(message_id, result)

    def _resolve(self, call_id: Any, value: Any) -> None:
        """Set the result for the request with the given id."""
        if not isinstance(call_id, int):
            return
        future = self._pending.pop(call_id, None)
        if future is None or future.done():
            return
        if isinstance(value, McpError):
            future.set_exception(value)
        else:
            future.set_result(dict(value) if isinstance(value, dict) else value)

    async def _reply_error(self, call_id: Any, code: int) -> None:
        """Answer a server request with a method-not-found error."""
        if self._process is None or self._process.stdin is None:
            return
        line = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": call_id,
                "error": {"code": code, "message": "Method not found"},
            },
            separators=(",", ":"),
        )
        with contextlib.suppress(BrokenPipeError, OSError):
            self._process.stdin.write((line + "\n").encode())
            await self._process.stdin.drain()

    def _fail_pending(self, error: McpError) -> None:
        """Fail every unresolved request and mark the connection as broken."""
        self._broken = error
        futures = list(self._pending.values())
        self._pending.clear()
        for future in futures:
            if not future.done():
                future.set_exception(error)
