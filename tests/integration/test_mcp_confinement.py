"""Exercise the MCP subprocess inside the real operating system sandbox."""

import json
import sys
from pathlib import Path

import pytest

from agent.mcp_client import StdioMcpClient
from agent.sandbox import SandboxPolicy, available

_SERVER = r"""
import json
import os
import sys
from pathlib import Path

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    result = {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {}}
    elif method == "tools/call":
        params = request["params"]
        arguments = params["arguments"]
        try:
            if params["name"] == "write":
                Path(arguments["path"]).write_text("written", encoding="utf-8")
                text = "written"
            elif params["name"] == "read":
                text = Path(arguments["path"]).read_text(encoding="utf-8")
            else:
                text = json.dumps({key: os.environ.get(key) for key in (
                    "HOME", "PATH", "TMPDIR", "XDG_CONFIG_HOME", "KEEP"
                )})
        except PermissionError:
            text = "denied"
        result = {"content": [{"type": "text", "text": text}]}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


@pytest.mark.skipif(not available(), reason="no usable sandbox backend")
async def test_required_sandbox_confines_mcp_and_reaps_process(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    environment = {"HOME": str(tmp_path), "PATH": "/unsafe", "KEEP": "kept"}
    original = dict(environment)
    policy = SandboxPolicy(mode="required", temp_root=tmp_path / "sandbox")
    client = StdioMcpClient(
        [str(Path(sys.executable).resolve()), "-u", "-c", _SERVER],
        cwd=project,
        env=environment,
        sandbox=policy,
        session_id="mcp-session",
        request_timeout=5,
    )
    async with client:
        process = client._process
        assert process is not None
        allowed = project / "allowed.txt"
        result = await client.call_tool("write", {"path": str(allowed)})
        assert result["content"][0]["text"] == "written"
        assert allowed.read_text(encoding="utf-8") == "written"
        for operation in ("read", "write"):
            result = await client.call_tool(operation, {"path": str(outside)})
            assert result["content"][0]["text"] == "denied"
        result = await client.call_tool("environment", {})
        actual = json.loads(result["content"][0]["text"])
        private_home = str((tmp_path / "sandbox" / "mcp-session").resolve())
        assert actual["HOME"] == actual["TMPDIR"] == private_home
        assert actual["XDG_CONFIG_HOME"] == str(Path(private_home) / ".config")
        assert actual["PATH"] != "/unsafe"
        assert actual["KEEP"] == "kept"
    assert outside.read_text(encoding="utf-8") == "private"
    assert environment == original
    assert process.returncode is not None
    assert client._process is None
