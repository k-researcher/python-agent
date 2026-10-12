from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent.security import PathSecurityError
from agent.tools.base import ToolContext
from agent.tools.builtin import EditFileTool, WriteFileTool


def context(root: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=root)


async def test_write_file_refuses_dangling_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "victim.txt"
    (project / "notes.txt").symlink_to(outside)

    with pytest.raises(PathSecurityError):
        await WriteFileTool().execute(context(project), {"path": "notes.txt", "content": "x"})
    assert not outside.exists()


async def test_write_file_creates_nested_directories(tmp_path: Path) -> None:
    result = await WriteFileTool().execute(
        context(tmp_path), {"path": "src/new/module.py", "content": "print('ok')\n"}
    )

    assert result["path"] == "src/new/module.py"
    assert (tmp_path / "src/new/module.py").read_text() == "print('ok')\n"
    assert not list((tmp_path / "src/new").glob(".*.tmp"))


async def test_edit_file_preserves_permissions(tmp_path: Path) -> None:
    script = tmp_path / "run.sh"
    script.write_text("echo old\n")
    script.chmod(0o755)

    await EditFileTool().execute(
        context(tmp_path), {"path": "run.sh", "old_text": "old", "new_text": "new"}
    )

    assert script.read_text() == "echo new\n"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
