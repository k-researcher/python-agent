"""Unit tests for the stdio MCP client.

Run a fake MCP server as a child process.
The server is a plain Python script driven by a JSON config file.
It logs every inbound message to a JSON lines file.
The tests assert on server info, tool listings, calls, and errors.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agent.mcp_client import McpError, McpTool, StdioMcpClient

_SERVER_SCRIPT = """import json
import os
import sys
import time


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


def main():
    with open(sys.argv[1]) as handle:
        config = json.load(handle)
    events_file = config.get("events_file")
    stay_alive = config.get("stay_alive", False)
    pages = config.get("tools_pages", [])
    delays = config.get("delay_methods", {})
    errors = config.get("error_on", {})
    hang = set(config.get("hang_on", []))
    die = set(config.get("die_on", []))
    ping = config.get("server_request", False)
    infinite_pages = config.get("infinite_pages", False)

    def log(obj):
        if events_file:
            with open(events_file, "a") as handle:
                handle.write(json.dumps(obj) + "\\n")

    if config.get("log_env"):
        present = [key for key in ["PATH", "HOME", "LANG", "EXTRA"] if key in os.environ]
        log({"type": "env", "present": present})

    while True:
        line = sys.stdin.readline()
        if line == "":
            if stay_alive:
                time.sleep(5)
                continue
            break
        text = line.strip()
        if not text:
            continue
        msg = json.loads(text)
        log(msg)
        if "method" not in msg:
            continue
        method = msg["method"]
        mid = msg.get("id")
        if mid is None:
            continue
        params = msg.get("params", {}) or {}
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": config.get("capabilities", {}),
                        "serverInfo": config.get(
                            "serverInfo", {"name": "fake-server", "version": "1.0"}
                        ),
                    },
                }
            )
            if ping:
                send({"jsonrpc": "2.0", "id": 777, "method": "custom/ping", "params": {}})
        elif method == "tools/list":
            if infinite_pages:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {"tools": [], "nextCursor": "page-2"},
                    }
                )
            elif len(pages) > 1 and not params.get("cursor"):
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "result": {"tools": pages[0], "nextCursor": "page-2"},
                    }
                )
            else:
                page = pages[-1] if params.get("cursor") else (pages[0] if pages else [])
                result = {"tools": page}
                if params.get("cursor") and len(pages) > 1:
                    result["nextCursor"] = None
                send({"jsonrpc": "2.0", "id": mid, "result": result})
        elif method == "tools/call":
            name = params.get("name")
            if name in die:
                break
            if name in hang:
                continue
            if name in errors:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "error": {
                            "code": errors[name]["code"],
                            "message": errors[name]["message"],
                        },
                    }
                )
                continue
            time.sleep(delays.get(name, 0.0))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "content": [{"type": "text", "text": "result-" + name}],
                        "isError": False,
                    },
                }
            )
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )


main()
"""


def _write_server(tmp_path: Path) -> Path:
    """Return the path to the fake MCP server script in tmp_path."""
    script = tmp_path / "fake_server.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return script


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Write the server config and return its path."""
    path = tmp_path / "server_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


async def _start_client(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    request_timeout: float = 5.0,
    env: dict[str, str] | None = None,
) -> StdioMcpClient:
    """Write the config, spawn the server, and run the handshake."""
    config_path = _write_config(tmp_path, config)
    options: dict[str, Any] = {}
    if env is not None:
        options["env"] = env
    client = StdioMcpClient(
        [sys.executable, str(_write_server(tmp_path)), str(config_path)],
        cwd=tmp_path,
        request_timeout=request_timeout,
        **options,
    )
    await client.start()
    return client


def _read_events(tmp_path: Path) -> list[dict[str, Any]]:
    """Return the server log lines as parsed dicts."""
    events_file = tmp_path / "events.jsonl"
    text = events_file.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_empty_command_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="command"):
        StdioMcpClient([], cwd=tmp_path)


async def test_start_initialize_and_notification(tmp_path: Path) -> None:
    events_file = tmp_path / "events.jsonl"
    client = await _start_client(tmp_path, {"events_file": str(events_file)})
    try:
        assert client.server_info["name"] == "fake-server"
        assert client.server_info["version"] == "1.0"
        assert client.capabilities == {}
    finally:
        await client.close()

    events = _read_events(tmp_path)
    assert events[0]["method"] == "initialize"
    assert events[0]["id"] == 1
    assert events[0]["params"]["protocolVersion"] == "2025-06-18"
    assert events[0]["params"]["clientInfo"] == {
        "name": "python-agent",
        "version": "0.1",
    }
    assert events[1]["method"] == "notifications/initialized"
    assert "id" not in events[1]


async def test_initialize_stores_capabilities(tmp_path: Path) -> None:
    config = {
        "capabilities": {"tools": {"listChanged": True}, "logging": {}},
        "events_file": str(tmp_path / "events.jsonl"),
    }
    client = await _start_client(tmp_path, config)
    try:
        assert client.capabilities == {"tools": {"listChanged": True}, "logging": {}}
    finally:
        await client.close()


async def test_list_tools_single_page(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "tools_pages": [
            [
                {
                    "name": "echo",
                    "description": "Echo text back.",
                    "inputSchema": {"properties": {"text": {"type": "string"}}},
                },
            ]
        ],
    }
    client = await _start_client(tmp_path, config)
    try:
        tools = await client.list_tools()
    finally:
        await client.close()

    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, McpTool)
    assert tool.name == "echo"
    assert tool.description == "Echo text back."
    assert tool.input_schema == {"properties": {"text": {"type": "string"}}}


async def test_list_tools_two_pages(tmp_path: Path) -> None:
    page_one = [{"name": "first", "description": "First page tool.", "inputSchema": {}}]
    page_two = [{"name": "second", "description": "Second page tool.", "inputSchema": {}}]
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "tools_pages": [page_one, page_two],
    }
    client = await _start_client(tmp_path, config)
    try:
        tools = await client.list_tools()
    finally:
        await client.close()

    assert [tool.name for tool in tools] == ["first", "second"]
    names = [event["params"].get("cursor") for event in _read_events(tmp_path)]
    assert [cursor for cursor in names if cursor is not None] == ["page-2"]


async def test_list_tools_page_limit(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "infinite_pages": True,
    }
    client = await _start_client(tmp_path, config)
    try:
        tools = await client.list_tools()
    finally:
        await client.close()

    assert tools == []
    # The client must stop after the page cap, not loop forever.
    requests = [event for event in _read_events(tmp_path) if event.get("method") == "tools/list"]
    assert len(requests) == 50


async def test_list_tools_empty(tmp_path: Path) -> None:
    client = await _start_client(tmp_path, {"events_file": str(tmp_path / "events.jsonl")})
    try:
        tools = await client.list_tools()
    finally:
        await client.close()

    assert tools == []


async def test_call_tool_returns_result(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "delays": {"echo": 0.05},
    }
    client = await _start_client(tmp_path, config)
    try:
        result = await client.call_tool("echo", {"text": "hello"})
    finally:
        await client.close()

    assert result["content"] == [{"type": "text", "text": "result-echo"}]
    assert result["isError"] is False


async def test_call_tool_jsonrpc_error(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "error_on": {"broken": {"code": -32000, "message": "boom"}},
    }
    client = await _start_client(tmp_path, config)
    try:
        with pytest.raises(McpError) as exc_info:
            await client.call_tool("broken", {})
    finally:
        await client.close()

    assert exc_info.value.code == -32000
    assert exc_info.value.message == "boom"


async def test_call_tool_timeout(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "hang_on": ["slow_call"],
    }
    client = await _start_client(tmp_path, config, request_timeout=0.3)
    try:
        with pytest.raises(McpError) as exc_info:
            await client.call_tool("slow_call", {})
    finally:
        await client.close()

    assert exc_info.value.code is None
    assert exc_info.value.message == "timed out"


async def test_call_tool_timeout_then_other_requests_still_work(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "hang_on": ["slow_call"],
    }
    client = await _start_client(tmp_path, config, request_timeout=0.2)
    try:
        with pytest.raises(McpError, match="timed out"):
            await client.call_tool("slow_call", {})
        result = await client.call_tool("echo", {})
    finally:
        await client.close()

    assert result["content"][0]["text"] == "result-echo"


async def test_server_dies_mid_request(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "die_on": ["explode"],
    }
    client = await _start_client(tmp_path, config)
    try:
        with pytest.raises(McpError) as exc_info:
            await client.call_tool("explode", {})
    finally:
        await client.close()

    assert exc_info.value.message in ("server closed", "read failed: ")


async def test_server_request_gets_method_not_found(tmp_path: Path) -> None:
    events_file = tmp_path / "events.jsonl"
    config = {"events_file": str(events_file), "server_request": True}
    client = await _start_client(tmp_path, config)
    try:
        result = await client.call_tool("echo", {})
    finally:
        await client.close()

    assert result["content"][0]["text"] == "result-echo"
    events = _read_events(tmp_path)
    replies = [event for event in events if event.get("id") == 777]
    assert len(replies) == 1
    assert replies[0]["error"]["code"] == -32601


async def test_parallel_calls_reverse_order(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
        "delays": {"a": 0.3, "b": 0.2, "c": 0.1},
    }
    client = await _start_client(tmp_path, config)
    try:
        results = await asyncio.gather(
            client.call_tool("a", {}),
            client.call_tool("b", {}),
            client.call_tool("c", {}),
        )
    finally:
        await client.close()

    assert [r["content"][0]["text"] for r in results] == [
        "result-a",
        "result-b",
        "result-c",
    ]


async def test_frozen_tool_attributes(tmp_path: Path) -> None:
    client = await _start_client(
        tmp_path,
        {
            "events_file": str(tmp_path / "events.jsonl"),
            "tools_pages": [[{"name": "echo", "description": "Echo.", "inputSchema": {}}]],
        },
    )
    try:
        tools = await client.list_tools()
    finally:
        await client.close()

    with pytest.raises(AttributeError):
        tools[0].name = "other"  # type: ignore[misc]


async def test_close_is_idempotent_and_no_orphan(tmp_path: Path) -> None:
    client = await _start_client(tmp_path, {"events_file": str(tmp_path / "events.jsonl")})
    await client.close()
    await client.close()

    # Wait so a leaked child process would be observable.
    await asyncio.sleep(0.1)
    current = asyncio.current_task()
    assert current is not None
    pending = [task for task in asyncio.all_tasks() if not task.done() and task is not current]
    assert pending == []


async def test_context_manager(tmp_path: Path) -> None:
    async with StdioMcpClient(
        [sys.executable, str(_write_server(tmp_path)), str(_write_config(tmp_path, {}))],
        cwd=tmp_path,
    ) as client:
        assert client.server_info["name"] == "fake-server"
    await asyncio.sleep(0.1)
    current = asyncio.current_task()
    assert current is not None
    pending = [task for task in asyncio.all_tasks() if not task.done() and task is not current]
    assert pending == []


async def test_call_tool_returns_error_flag(tmp_path: Path) -> None:
    config = {
        "events_file": str(tmp_path / "events.jsonl"),
    }
    client = await _start_client(tmp_path, config)
    try:
        result = await client.call_tool("failing", {})
    finally:
        await client.close()

    assert result["isError"] is False


# --- review fixes: limits, broken connection, bad payloads, cleanup ---------------

_INIT_REPLY = (
    "import json, os, sys, time\n"
    "def send(obj):\n"
    "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
    "    sys.stdout.flush()\n"
    "def read():\n"
    "    return json.loads(sys.stdin.readline())\n"
    "msg = read()\n"
    "send({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'serverInfo': {'name': 's'}}})\n"
    "read()  # notifications/initialized\n"
)


def _script_server(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "script_server.py"
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def _pid_is_dead(pid_file: Path) -> bool:
    import os
    import time

    deadline = time.monotonic() + 5
    pid = int(pid_file.read_text(encoding="utf-8"))
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


async def test_line_without_end_is_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.mcp_client as mcp_client

    monkeypatch.setattr(mcp_client, "_MAX_LINE_BYTES", 1000)
    body = (
        _INIT_REPLY + "read()\nsys.stdout.write('x' * 5000)\nsys.stdout.flush()\ntime.sleep(30)\n"
    )
    async with StdioMcpClient(
        _script_server(tmp_path, body), cwd=tmp_path, request_timeout=10
    ) as client:
        started = asyncio.get_running_loop().time()
        with pytest.raises(McpError, match="line too long"):
            await client.list_tools()
        assert asyncio.get_running_loop().time() - started < 3


async def test_request_after_server_exit_fails_at_once(tmp_path: Path) -> None:
    body = _INIT_REPLY + "sys.exit(0)\n"
    async with StdioMcpClient(
        _script_server(tmp_path, body), cwd=tmp_path, request_timeout=10
    ) as client:
        await asyncio.sleep(0.3)
        started = asyncio.get_running_loop().time()
        with pytest.raises(McpError):
            await client.list_tools()
        assert asyncio.get_running_loop().time() - started < 1


@pytest.mark.parametrize(
    ("reply", "method"),
    [("[1, 2]", "call"), ("{'tools': [{'description': 'no name'}]}", "list")],
)
async def test_unexpected_result_shape_is_an_mcp_error(
    tmp_path: Path, reply: str, method: str
) -> None:
    body = (
        _INIT_REPLY
        + "msg = read()\n"
        + f"send({{'jsonrpc': '2.0', 'id': msg['id'], 'result': {reply}}})\n"
        + "time.sleep(30)\n"
    )
    async with StdioMcpClient(_script_server(tmp_path, body), cwd=tmp_path) as client:
        with pytest.raises(McpError, match="protocol error"):
            if method == "call":
                await client.call_tool("x", {})
            else:
                await client.list_tools()


async def test_failed_initialize_stops_the_server(tmp_path: Path) -> None:
    pid_file = tmp_path / "server.pid"
    body = (
        "import json, os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "msg = json.loads(sys.stdin.readline())\n"
        "sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'error': {'code': -32000, 'message': 'no'}}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    client = StdioMcpClient(_script_server(tmp_path, body), cwd=tmp_path)
    with pytest.raises(McpError, match="no"):
        await client.start()
    assert _pid_is_dead(pid_file)


async def test_close_kills_a_child_that_outlives_the_server(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    body = (
        _INIT_REPLY
        + "import subprocess\n"
        + "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        + f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        + "sys.exit(0)\n"
    )
    client = StdioMcpClient(_script_server(tmp_path, body), cwd=tmp_path)
    await client.start()
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text(encoding="utf-8"):
            break
        await asyncio.sleep(0.05)
    await client.close()
    assert _pid_is_dead(pid_file)
