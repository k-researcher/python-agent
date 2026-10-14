"""Tests for project check detection and execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent.tools.base import ToolContext, ToolError
from agent.tools.builtin import ShellTool
from agent.tools.check_tools import (
    MAX_CONFIG_BYTES,
    CheckCommand,
    RunChecksTool,
    detect_checks,
)


def write(path: Path, content: str) -> Path:
    """Write a file inside a project directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_project(tmp_path: Path) -> Path:
    """Create an empty project directory."""
    project = tmp_path / "project"
    project.mkdir()
    return project


def context_for(project: Path) -> ToolContext:
    """Build a tool context for the given project root."""
    return ToolContext(session_id="test", project_root=project)


class RecordingShell(ShellTool):
    """A fake ShellTool that records commands and returns fixed results."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.commands: list[str] = []
        self._results = list(results or [])
        self._default: dict[str, Any] = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "truncated": False,
        }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        del context
        self.commands.append(str(arguments["command"]))
        if self._results:
            return self._results.pop(0)
        return dict(self._default)


# --- detect_checks: pyproject.toml ---


def test_detect_checks_ruff_table(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n")

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="lint", argv=("ruff", "check", "."), reason="pyproject.toml declares ruff"
    )
    assert checks == [expected]


def test_detect_checks_ruff_in_dependencies(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "pyproject.toml",
        '[project]\ndependencies = ["ruff>=0.8"]\n',
    )

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="lint", argv=("ruff", "check", "."), reason="pyproject.toml declares ruff"
    )
    assert checks == [expected]


def test_detect_checks_ruff_in_optional_dependencies(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "pyproject.toml",
        '[project.optional-dependencies]\ndev = ["ruff>=0.8"]\n',
    )

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="lint", argv=("ruff", "check", "."), reason="pyproject.toml declares ruff"
    )
    assert checks == [expected]


def test_detect_checks_mypy_table(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.mypy]\n")

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="typecheck", argv=("mypy", "."), reason="pyproject.toml declares mypy"
    )
    assert checks == [expected]


def test_detect_checks_mypy_in_dependencies(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "pyproject.toml",
        '[project]\ndependencies = ["mypy>=1.0"]\n',
    )

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="typecheck", argv=("mypy", "."), reason="pyproject.toml declares mypy"
    )
    assert checks == [expected]


def test_detect_checks_mypy_with_packages_omits_path(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", '[tool.mypy]\npackages = ["src"]\n')

    checks = detect_checks(project)

    expected = CheckCommand(kind="typecheck", argv=("mypy",), reason="pyproject.toml declares mypy")
    assert checks == [expected]


def test_detect_checks_mypy_with_files_omits_path(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", '[tool.mypy]\nfiles = ["src/main.py"]\n')

    checks = detect_checks(project)

    expected = CheckCommand(kind="typecheck", argv=("mypy",), reason="pyproject.toml declares mypy")
    assert checks == [expected]


def test_detect_checks_both_ruff_and_mypy(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n[tool.mypy]\n")

    checks = detect_checks(project)

    assert [check.kind for check in checks] == ["lint", "typecheck"]


# --- detect_checks: package.json ---


def test_detect_checks_package_json_lint(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "package.json",
        json.dumps({"scripts": {"lint": "eslint src"}}),
    )

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="lint",
        argv=("npm", "run", "--silent", "lint"),
        reason="package.json has a lint script",
    )
    assert checks == [expected]


def test_detect_checks_package_json_typecheck(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "package.json",
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
    )

    checks = detect_checks(project)

    expected = CheckCommand(
        kind="typecheck",
        argv=("npm", "run", "--silent", "typecheck"),
        reason="package.json has a typecheck script",
    )
    assert checks == [expected]


def test_detect_checks_combines_sources(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n")
    write(
        project / "package.json",
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
    )

    checks = detect_checks(project)

    assert [check.kind for check in checks] == ["lint", "typecheck"]
    assert checks[0].argv == ("ruff", "check", ".")
    assert checks[1].argv == ("npm", "run", "--silent", "typecheck")


# --- detect_checks: edge cases ---


def test_detect_checks_empty_project(tmp_path: Path) -> None:
    project = make_project(tmp_path)

    checks = detect_checks(project)

    assert checks == []


def test_detect_checks_broken_pyproject(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "this is not valid [[ toml")

    checks = detect_checks(project)

    assert checks == []


def test_detect_checks_broken_package_json(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "package.json", "{not valid json")

    checks = detect_checks(project)

    assert checks == []


def test_detect_checks_large_pyproject_ignored(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "pyproject.toml",
        "[tool.ruff]\n" + "x" * (MAX_CONFIG_BYTES + 1),
    )

    checks = detect_checks(project)

    assert checks == []


def test_detect_checks_large_package_json_ignored(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "package.json",
        json.dumps({"scripts": {"lint": "eslint"}}) + "x" * (MAX_CONFIG_BYTES + 1),
    )

    checks = detect_checks(project)

    assert checks == []


@pytest.mark.skipif(os.name != "posix", reason="symlinks are POSIX-only")
def test_detect_checks_ignores_symlink_outside_project(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    outside = tmp_path / "outside.toml"
    outside.write_text("[tool.ruff]\n", encoding="utf-8")
    os.symlink(outside, project / "pyproject.toml")

    checks = detect_checks(project)

    assert checks == []


@pytest.mark.skipif(os.name != "posix", reason="symlinks are POSIX-only")
def test_detect_checks_ignores_package_json_symlink_outside(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"scripts": {"lint": "eslint"}}', encoding="utf-8")
    os.symlink(outside, project / "package.json")

    checks = detect_checks(project)

    assert checks == []


# --- RunChecksTool ---


async def test_run_checks_tool_runs_all_in_order(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n[tool.mypy]\n")
    shell = RecordingShell()
    tool = RunChecksTool(shell_tool=shell)

    result = await tool.execute(context_for(project), {"kind": "all"})

    assert shell.commands == ["ruff check .", "mypy ."]
    assert len(result["results"]) == 2
    assert result["results"][0]["kind"] == "lint"
    assert result["results"][0]["command"] == "ruff check ."
    assert result["results"][1]["kind"] == "typecheck"
    assert result["results"][1]["command"] == "mypy ."


async def test_run_checks_tool_filters_by_kind(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n[tool.mypy]\n")
    shell = RecordingShell()
    tool = RunChecksTool(shell_tool=shell)

    result = await tool.execute(context_for(project), {"kind": "lint"})

    assert shell.commands == ["ruff check ."]
    assert len(result["results"]) == 1
    assert result["results"][0]["kind"] == "lint"


async def test_run_checks_tool_raises_when_nothing_detected(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    tool = RunChecksTool(shell_tool=RecordingShell())

    with pytest.raises(ToolError, match="No project check commands"):
        await tool.execute(context_for(project), {"kind": "all"})


async def test_run_checks_tool_propagates_shell_result(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n")
    shell = RecordingShell(
        [
            {
                "exit_code": 2,
                "stdout": "error: line 1",
                "stderr": "",
                "truncated": False,
            }
        ]
    )
    tool = RunChecksTool(shell_tool=shell)

    result = await tool.execute(context_for(project), {"kind": "all"})

    item = result["results"][0]
    assert item["exit_code"] == 2
    assert item["stdout"] == "error: line 1"
    assert item["stderr"] == ""
    assert item["truncated"] is False


async def test_run_checks_tool_runs_package_json_commands(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(
        project / "package.json",
        json.dumps({"scripts": {"lint": "eslint", "typecheck": "tsc"}}),
    )
    shell = RecordingShell()
    tool = RunChecksTool(shell_tool=shell)

    result = await tool.execute(context_for(project), {"kind": "all"})

    assert shell.commands == [
        "npm run --silent lint",
        "npm run --silent typecheck",
    ]
    assert len(result["results"]) == 2


@pytest.mark.parametrize("extra", [0, 1])
def test_detect_checks_size_limit_is_exact(tmp_path: Path, extra: int) -> None:
    # Valid TOML padded with a comment: only the size limit decides the result.
    project = make_project(tmp_path)
    body = "[tool.ruff]\n#"
    padding = MAX_CONFIG_BYTES - len(body.encode()) - 1 + extra
    write(project / "pyproject.toml", body + "x" * padding + "\n")
    commands = [check.argv for check in detect_checks(project)]
    assert commands == ([] if extra else [("ruff", "check", ".")])


@pytest.mark.skipif(os.name != "posix", reason="symlinks are POSIX-only")
def test_detect_checks_ignores_symlink_inside_project(tmp_path: Path) -> None:
    # A symlink is never followed, even when it points to a file inside the project.
    project = make_project(tmp_path)
    write(project / "real.toml", "[tool.ruff]\n")
    os.symlink(project / "real.toml", project / "pyproject.toml")
    assert detect_checks(project) == []


async def test_run_checks_tool_filters_typecheck(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    write(project / "pyproject.toml", "[tool.ruff]\n[tool.mypy]\nfiles = ['src']\n")
    shell = RecordingShell()
    tool = RunChecksTool(shell_tool=shell)

    result = await tool.execute(context_for(project), {"kind": "typecheck"})

    assert shell.commands == ["mypy"]
    assert [item["kind"] for item in result["results"]] == ["typecheck"]
