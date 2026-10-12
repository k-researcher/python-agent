import sys
from pathlib import Path

import pytest

from agent.tools.base import ToolContext
from agent.tools.builtin import MAX_COMMAND_OUTPUT, ShellTool


async def test_shell_does_not_inherit_agent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_LLM_API_KEY", "must-not-leak")
    command = (
        f"{sys.executable} -c \"import os; print(os.environ.get('AGENT_LLM_API_KEY', 'missing'))\""
    )
    result = await ShellTool().execute(
        ToolContext(session_id="test", project_root=tmp_path), {"command": command}
    )

    assert result["stdout"].strip() == "missing"


async def test_shell_kills_process_when_output_limit_is_exceeded(tmp_path: Path) -> None:
    command = f"{sys.executable} -c \"print('x' * {MAX_COMMAND_OUTPUT * 2})\""
    result = await ShellTool().execute(
        ToolContext(session_id="test", project_root=tmp_path), {"command": command}
    )

    assert result["truncated"] is True
    assert len(result["stdout"].encode()) <= MAX_COMMAND_OUTPUT


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX-only")
async def test_cancelled_shell_kills_spawned_children(tmp_path: Path) -> None:
    import asyncio
    import os

    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "time.sleep(60)"
    )
    command = f"{sys.executable} -c {script!r}"
    shell_context = ToolContext(session_id="test", project_root=tmp_path)
    task = asyncio.create_task(ShellTool().execute(shell_context, {"command": command}))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    child_pid = int(pid_file.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(50):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    os.kill(child_pid, 9)
    raise AssertionError("grandchild process survived cancellation")
