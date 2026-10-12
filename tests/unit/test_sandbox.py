"""Shell sandbox: profile rules, safe temporary directories and real confinement on macOS."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from agent import sandbox
from agent.sandbox import (
    SandboxPolicy,
    SandboxUnavailableError,
    prepare,
    profile,
    session_temp_dir,
)
from agent.tools.base import ToolContext, ToolError
from agent.tools.builtin import ShellTool

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="no usable sandbox backend")
# The base interpreter is a trusted runtime path; the agent virtual environment is not.
PYTHON = os.path.realpath(getattr(sys, "_base_executable", sys.executable))


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = (tmp_path / "project").resolve()
    outside = (tmp_path / "outside").resolve()
    temp_root = (tmp_path / "sandbox").resolve()
    project.mkdir()
    outside.mkdir()
    (project / "inside.txt").write_text("inside", encoding="utf-8")
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    return project, outside, temp_root


def _context(project: Path, temp_root: Path) -> ToolContext:
    policy = SandboxPolicy(mode="required", temp_root=temp_root)
    return ToolContext(session_id="s1", project_root=project, sandbox=policy)


def _run(context: ToolContext, *argv: str) -> dict[str, object]:
    arguments = {"command": " ".join(argv), "timeout_seconds": 60}
    return asyncio.run(ShellTool().execute(context, arguments))


def _python(context: ToolContext, script: str) -> dict[str, object]:
    return _run(context, shlex.quote(PYTHON), "-c", shlex.quote(script))


# Profile and policy rules (no sandbox backend needed).


def test_profile_denies_by_default_and_closes_risky_operations() -> None:
    text = profile(Path("/p"), Path("/t"), ["/r"])
    assert text.startswith("(version 1)\n(deny default)")
    assert "network" not in text
    assert "(deny process-info*)" in text
    assert "ipc-posix" not in text
    assert "(allow sysctl-read)" not in text
    assert "(allow file-read-metadata)" not in text
    assert '(subpath "/p")' in text and '(subpath "/t")' in text and '(subpath "/r")' in text


def test_profile_network_is_explicit() -> None:
    assert "(allow network*)" in profile(Path("/p"), Path("/t"), [], allow_network=True)


def test_profile_quotes_paths_and_refuses_control_characters() -> None:
    assert '"/p\\"x"' in profile(Path('/p"x'), Path("/t"), [])
    with pytest.raises(SandboxUnavailableError):
        profile(Path("/p\nx"), Path("/t"), [])


def test_project_files_never_widen_the_profile(
    layout: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _, temp_root = layout
    (project / ".venv").mkdir()
    (project / ".venv" / "pyvenv.cfg").write_text("home = /bin\n", encoding="utf-8")
    (project / ".git").write_text("gitdir: /\n", encoding="utf-8")
    monkeypatch.setattr(sandbox, "available", lambda: True)
    prepared = prepare(SandboxPolicy(mode="required", temp_root=temp_root), project, "s1")
    assert prepared is not None
    assert '(subpath "/")' not in prepared[0][2]


def test_off_mode_has_no_prefix(tmp_path: Path) -> None:
    assert prepare(SandboxPolicy(mode="off"), tmp_path, "s1") is None


def test_required_mode_without_backend_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(SandboxUnavailableError):
        prepare(SandboxPolicy(mode="required", temp_root=tmp_path), tmp_path, "s1")
    assert prepare(SandboxPolicy(mode="auto", temp_root=tmp_path), tmp_path, "s1") is None


def test_shell_refuses_to_run_without_required_sandbox(
    layout: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _, temp_root = layout
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(ToolError, match="sandbox"):
        _run(_context(project, temp_root), "echo", "hi")


# Temporary directory checks.


def test_temp_dir_is_private(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    path = session_temp_dir(temp_root, "s1", project)
    assert path.is_dir() and not path.is_symlink()
    assert (path.stat().st_mode & 0o777) == 0o700


def test_temp_dir_link_is_refused(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    temp_root.mkdir()
    (temp_root / "s1").symlink_to(outside)
    with pytest.raises(SandboxUnavailableError):
        session_temp_dir(temp_root, "s1", project)


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o500, 0o600, 0o1700])
def test_temp_dir_with_other_mode_is_refused(layout: tuple[Path, Path, Path], mode: int) -> None:
    project, _, temp_root = layout
    (temp_root / "s1").mkdir(parents=True)
    os.chmod(temp_root / "s1", mode)
    with pytest.raises(SandboxUnavailableError):
        session_temp_dir(temp_root, "s1", project)


def test_temp_root_inside_project_is_refused(layout: tuple[Path, Path, Path]) -> None:
    project, _, _ = layout
    with pytest.raises(SandboxUnavailableError, match="outside the project"):
        session_temp_dir(project / "data" / "sandbox", "s1", project)


@pytest.mark.parametrize("session_id", ["../x", "/abs", "", "a/b", ".hidden", "x" * 65])
def test_bad_session_id_is_refused(layout: tuple[Path, Path, Path], session_id: str) -> None:
    project, _, temp_root = layout
    with pytest.raises(SandboxUnavailableError):
        session_temp_dir(temp_root, session_id, project)


# Real confinement (macOS with a usable sandbox backend).


@needs_sandbox
def test_project_read_and_write_are_allowed(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    context = _context(project, temp_root)
    assert _run(context, "cat", "inside.txt")["stdout"] == "inside"
    assert _run(context, "touch", "created.txt")["exit_code"] == 0
    assert (project / "created.txt").exists()


@needs_sandbox
def test_reading_outside_the_project_is_denied(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    result = _run(_context(project, temp_root), "cat", str(outside / "secret.txt"))
    assert result["exit_code"] != 0
    assert "secret" not in str(result["stdout"])


@needs_sandbox
def test_link_to_outside_file_is_denied(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    (project / "link.txt").symlink_to(outside / "secret.txt")
    result = _run(_context(project, temp_root), "cat", "link.txt")
    assert result["exit_code"] != 0
    assert "secret" not in str(result["stdout"])


@needs_sandbox
def test_hard_link_to_outside_file_cannot_be_created(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    result = _run(_context(project, temp_root), "ln", str(outside / "secret.txt"), "hard.txt")
    assert result["exit_code"] != 0
    assert not (project / "hard.txt").exists()


@needs_sandbox
def test_writing_outside_the_project_is_denied(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    result = _run(_context(project, temp_root), "touch", str(outside / "new.txt"))
    assert result["exit_code"] != 0
    assert not (outside / "new.txt").exists()


@needs_sandbox
def test_system_data_volume_path_is_denied(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    alias = Path("/System/Volumes/Data") / str(outside / "secret.txt").lstrip("/")
    result = _run(_context(project, temp_root), "cat", str(alias))
    assert result["exit_code"] != 0
    assert "secret" not in str(result["stdout"])


@needs_sandbox
def test_home_is_private(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    result = _python(_context(project, temp_root), "import os; print(os.environ['HOME'])")
    assert str(result["stdout"]).strip() == str(temp_root / "s1")


@needs_sandbox
def test_network_is_denied(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    script = "import socket; socket.create_connection(('127.0.0.1', 9), 2)"
    result = _python(_context(project, temp_root), script)
    assert result["exit_code"] != 0
    assert "Operation not permitted" in str(result["stderr"])


@needs_sandbox
def test_other_process_arguments_are_hidden(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    result = _run(_context(project, temp_root), "ps", "-o", "command=", "-p", str(os.getpid()))
    assert "pytest" not in str(result["stdout"])


@needs_sandbox
def test_shared_memory_is_denied(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    script = "from multiprocessing import shared_memory as m; m.SharedMemory(create=True, size=8)"
    result = _python(_context(project, temp_root), script)
    assert result["exit_code"] != 0
    assert "PermissionError" in str(result["stderr"])


@needs_sandbox
def test_input_is_closed(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    result = _python(_context(project, temp_root), "import sys; print(repr(sys.stdin.read()))")
    assert str(result["stdout"]).strip() == "''"


@needs_sandbox
def test_child_processes_stay_in_the_sandbox(layout: tuple[Path, Path, Path]) -> None:
    project, outside, temp_root = layout
    script = (
        "import subprocess, sys; "
        f"sys.exit(subprocess.run(['cat', '{outside / 'secret.txt'}']).returncode)"
    )
    result = _python(_context(project, temp_root), script)
    assert result["exit_code"] != 0
    assert "secret" not in str(result["stdout"])
    assert "Operation not permitted" in str(result["stderr"])


@needs_sandbox
def test_agent_virtual_environment_is_not_runnable(layout: tuple[Path, Path, Path]) -> None:
    project, _, temp_root = layout
    if Path(sys.prefix) == Path(sys.base_prefix):
        pytest.skip("tests do not run in a virtual environment")
    result = _run(_context(project, temp_root), str(Path(sys.prefix) / "bin" / "python"), "-V")
    assert result["exit_code"] != 0


def test_profile_uses_checked_temp_path_without_resolution(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    text = profile(tmp_path / "project", link, [])
    assert f'(subpath "{link}")' in text
    assert f'(subpath "{real}")' not in text
