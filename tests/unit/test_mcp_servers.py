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
    suggested_read_paths,
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


def test_install_directory_of_the_program_is_suggested(settings: Settings, tmp_path: Path) -> None:
    program = tmp_path / "toolbox" / "bin" / "server"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\n")
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    assert suggested_read_paths([str(program)], settings) == [str(tmp_path / "toolbox")]
    # A program at the root of the file system would open everything: nothing is suggested.
    assert suggested_read_paths(["/bin/sh"], settings) == []
    assert suggested_read_paths(["no-such-program-anywhere"], settings) == []


def test_a_program_inside_the_agent_data_is_not_opened(settings: Settings) -> None:
    program = settings.data_dir / "bin" / "server"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\n")
    assert suggested_read_paths([str(program)], settings) == []


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
