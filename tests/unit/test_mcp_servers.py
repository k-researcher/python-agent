"""Checks of an MCP server definition from the UI, and its start without a sandbox."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agent import sandbox
from agent.config import Settings
from agent.mcp_servers import (
    McpServerError,
    check_command,
    check_env_key,
    check_read_path,
    check_server,
    check_server_id,
    runtime_read_paths,
    sandbox_policy,
    to_source,
)
from agent.model_overrides import encrypt_secret
from agent.models import McpServer


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def test_server_id_is_short_and_plain() -> None:
    assert check_server_id("files_2-x") == "files_2-x"
    for value in ("", "Files", "-x", "a" * 33, "a b", "a/b"):
        with pytest.raises(McpServerError):
            check_server_id(value)


def test_command_must_have_a_program_and_sane_parts() -> None:
    assert check_command(("npx", "-y", "server")) == ["npx", "-y", "server"]
    for command in ([], [" "], ["npx", ""], ["npx", "a\x00b"], ["x"] * 65, ["x" * 4097]):
        with pytest.raises(McpServerError):
            check_command(command)


@pytest.mark.parametrize("key", ["PATH", "HOME", "TMPDIR", "DYLD_INSERT_LIBRARIES", "LD_PRELOAD"])
def test_sandbox_variables_cannot_be_replaced(key: str) -> None:
    with pytest.raises(McpServerError, match="песочница"):
        check_env_key(key)


def test_plain_variable_names_are_accepted() -> None:
    assert check_env_key("GITHUB_TOKEN") == "GITHUB_TOKEN"
    for key in ("1TOKEN", "A-B", "", "A B"):
        with pytest.raises(McpServerError):
            check_env_key(key)


def test_read_path_must_be_narrow_and_away_from_the_agent_data(
    settings: Settings, tmp_path: Path
) -> None:
    allowed = tmp_path / "tools"
    assert check_read_path(f"{allowed}/./", settings) == str(allowed)
    refused = [
        "relative/path",
        "/",
        str(Path.home()),
        str(Path.home().parent),
        f"{allowed}/../tools",
        str(settings.data_dir),
        str(settings.data_dir / "secret-area"),
        str(tmp_path),
    ]
    for value in refused:
        with pytest.raises(McpServerError):
            check_read_path(value, settings)


def test_read_path_through_a_link_into_the_data_is_refused(
    settings: Settings, tmp_path: Path
) -> None:
    settings.data_dir.mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(settings.data_dir)
    with pytest.raises(McpServerError, match="папку данных"):
        check_read_path(str(link), settings)


def test_install_directory_of_the_program_is_opened(settings: Settings, tmp_path: Path) -> None:
    program = tmp_path / "toolbox" / "bin" / "server"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\n")
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    assert runtime_read_paths([str(program)], settings) == [str(tmp_path / "toolbox")]
    # A program at the root of the file system would open everything: nothing is opened.
    assert runtime_read_paths(["/bin/sh"], settings) == []


def _virtual_environment(tmp_path: Path) -> tuple[Path, Path]:
    """Build a server script in a virtual environment whose Python lives elsewhere."""
    runtime = tmp_path / "runtimes" / "cpython-3.13" / "bin" / "python3.13"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    environment = tmp_path / "server" / ".venv" / "bin"
    environment.mkdir(parents=True)
    (environment.parent / "pyvenv.cfg").write_text(f"home = {runtime.parent}\n")
    (environment / "python").symlink_to(runtime)
    script = environment / "docs-server"
    script.write_text(f"#!{environment / 'python'}\nimport docs\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, runtime


def test_the_real_python_of_a_virtual_environment_is_opened(
    settings: Settings, tmp_path: Path
) -> None:
    script, _ = _virtual_environment(tmp_path)
    # The project folder holds the environment and the code of a development install.
    assert runtime_read_paths([str(script), "--stdio"], settings) == [
        str(tmp_path / "server"),
        str(tmp_path / "runtimes" / "cpython-3.13"),
    ]


def test_a_linked_program_opens_its_install_and_its_real_folder(
    settings: Settings, tmp_path: Path
) -> None:
    # The layout of npx in Homebrew: a link from bin into lib/node_modules.
    real = tmp_path / "brew" / "lib" / "node_modules" / "npm" / "bin" / "npx-cli.js"
    real.parent.mkdir(parents=True)
    real.write_text("")
    (tmp_path / "brew" / "bin").mkdir()
    (tmp_path / "brew" / "bin" / "npx").symlink_to(real)
    assert runtime_read_paths([str(tmp_path / "brew" / "bin" / "npx")], settings) == [
        str(tmp_path / "brew"),
        str(tmp_path / "brew" / "lib" / "node_modules" / "npm"),
    ]


def test_an_interpreter_found_through_env_is_opened(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, runtime = _virtual_environment(tmp_path)
    runtime.chmod(runtime.stat().st_mode | stat.S_IXUSR)
    (runtime.parent / "node").symlink_to(runtime)
    monkeypatch.setenv("PATH", str(runtime.parent))
    script = tmp_path / "tool" / "bin" / "server"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env -S node --no-warnings\n")
    assert runtime_read_paths([str(script)], settings) == [
        str(tmp_path / "tool"),
        str(tmp_path / "runtimes" / "cpython-3.13"),
    ]


def test_runtime_paths_never_open_the_agent_data(settings: Settings, tmp_path: Path) -> None:
    script, runtime = _virtual_environment(tmp_path)
    settings.data_dir.mkdir(parents=True)
    moved = settings.data_dir / "cpython" / "bin" / "python3.13"
    moved.parent.mkdir(parents=True)
    moved.write_text("")
    (script.parent / "python").unlink()
    (script.parent / "python").symlink_to(moved)
    assert runtime_read_paths([str(script)], settings) == [str(tmp_path / "server")]
    assert runtime_read_paths(["no-such-program-anywhere"], settings) == []


def test_the_sandbox_of_a_server_gets_its_runtime_without_settings(
    settings: Settings, tmp_path: Path
) -> None:
    script, _ = _virtual_environment(tmp_path)
    source = to_source(server(settings, command=[str(script)], read_paths=["/opt/extra"]))
    paths = sandbox_policy(settings, source).extra_read_paths
    assert "/opt/extra" in paths
    assert str(tmp_path / "runtimes" / "cpython-3.13") in paths


def test_an_environment_in_the_home_folder_opens_only_itself(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The folder above an environment can be too wide to open.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "server")
    script, _ = _virtual_environment(tmp_path)
    assert runtime_read_paths([str(script)], settings)[0] == str(tmp_path / "server" / ".venv")


def test_a_program_inside_the_agent_data_is_not_opened(settings: Settings) -> None:
    program = settings.data_dir / "bin" / "server"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\n")
    assert runtime_read_paths([str(program)], settings) == []


def server(settings: Settings, **values: object) -> McpServer:
    fields: dict[str, object] = {
        "id": "files",
        "command": ["server", "--stdio"],
        "env_secrets": {"TOKEN": encrypt_secret("value", settings)},
        "project_ids": [],
        "allowed_modes": ["dev"],
        "risk_level": "network_access",
        "allow_network": False,
        "read_paths": [],
        "enabled": True,
        "call_timeout": 120,
        **values,
    }
    return McpServer(**fields)


def test_ui_server_becomes_a_source_without_a_workspace(settings: Settings) -> None:
    source = to_source(server(settings, project_ids=["p1"], allow_network=True))
    assert source.workspace_id is None
    assert source.project_ids == frozenset({"p1"})
    assert source.env_keys == ("TOKEN",)
    assert source.allow_network
    assert source.command == ("server", "--stdio")


async def test_check_without_a_sandbox_explains_and_starts_nothing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "started"
    monkeypatch.setattr(sandbox, "available", lambda: False)
    result = await check_server(
        settings, server(settings, command=["/bin/sh", "-c", f"touch {marker}"])
    )
    assert result["ok"] is False
    assert "Песочница" in result["error"]
    assert not marker.exists()
    assert os.path.isdir(settings.data_dir / "mcp-check" / "files")
