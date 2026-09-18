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
        f'{sys.executable} -c "import os; '
        "print(os.environ.get('AGENT_LLM_API_KEY', 'missing'))\""
    )
    result = await ShellTool().execute(
        ToolContext(session_id="test", project_root=tmp_path), {"command": command}
    )

    assert result["stdout"].strip() == "missing"


async def test_shell_kills_process_when_output_limit_is_exceeded(tmp_path: Path) -> None:
    command = f'{sys.executable} -c "print(\'x\' * {MAX_COMMAND_OUTPUT * 2})"'
    result = await ShellTool().execute(
        ToolContext(session_id="test", project_root=tmp_path), {"command": command}
    )

    assert result["truncated"] is True
    assert len(result["stdout"].encode()) <= MAX_COMMAND_OUTPUT
