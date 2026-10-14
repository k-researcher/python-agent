"""Project check detection and execution (lint and typecheck tools)."""

from __future__ import annotations

import json
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.safe_fs import SafeProjectFS
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.builtin import ShellTool

MAX_CONFIG_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class CheckCommand:
    """One check command to run inside the project."""

    kind: str
    argv: tuple[str, ...]
    reason: str


def _read_project_config(name: str, root: Path) -> str | None:
    """Read a config file in the project root without following symlinks.

    SafeProjectFS opens every path component with O_NOFOLLOW and checks the size on the
    open descriptor, so the file cannot be swapped between the check and the read.
    A symlink, a file larger than 1 MB and a missing or unreadable file return None.
    """
    try:
        return SafeProjectFS(root).read_text(name, max_bytes=MAX_CONFIG_BYTES)
    except (OSError, ValueError, ToolError):
        return None


def _dependency_name(value: str) -> str:
    """Extract the package name from a PEP 508 requirement string."""
    name = value
    for separator in ("[", "<", ">", "=", "~", "!", ";", " "):
        name = name.split(separator, 1)[0]
    return name.replace("_", "-").lower()


def _has_dependency(section: Any, target: str) -> bool:
    """Return True when a dependency list contains the target package."""
    if not isinstance(section, list):
        return False
    return any(isinstance(entry, str) and _dependency_name(entry) == target for entry in section)


def _detect_pyproject_checks(data: dict[str, Any]) -> list[CheckCommand]:
    """Build check commands from a parsed pyproject.toml document."""
    checks: list[CheckCommand] = []
    tool_section = data.get("tool", {})
    if not isinstance(tool_section, dict):
        tool_section = {}

    project_section = data.get("project", {})
    if not isinstance(project_section, dict):
        project_section = {}

    dependencies = project_section.get("dependencies", [])
    optional = project_section.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        optional = {}

    has_ruff_tool = isinstance(tool_section.get("ruff"), dict)
    has_mypy_tool = isinstance(tool_section.get("mypy"), dict)
    has_ruff_dep = _has_dependency(dependencies, "ruff") or any(
        _has_dependency(group, "ruff") for group in optional.values()
    )
    has_mypy_dep = _has_dependency(dependencies, "mypy") or any(
        _has_dependency(group, "mypy") for group in optional.values()
    )

    if has_ruff_tool or has_ruff_dep:
        reason = "pyproject.toml declares ruff"
        checks.append(CheckCommand(kind="lint", argv=("ruff", "check", "."), reason=reason))

    if has_mypy_tool or has_mypy_dep:
        reason = "pyproject.toml declares mypy"
        mypy_section = tool_section.get("mypy")
        if isinstance(mypy_section, dict) and (
            "packages" in mypy_section or "files" in mypy_section
        ):
            checks.append(CheckCommand(kind="typecheck", argv=("mypy",), reason=reason))
        else:
            checks.append(CheckCommand(kind="typecheck", argv=("mypy", "."), reason=reason))

    return checks


def _detect_package_json_checks(data: Any) -> list[CheckCommand]:
    """Build check commands from a parsed package.json document."""
    checks: list[CheckCommand] = []
    if not isinstance(data, dict):
        return checks
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return checks
    if "lint" in scripts:
        reason = "package.json has a lint script"
        checks.append(
            CheckCommand(kind="lint", argv=("npm", "run", "--silent", "lint"), reason=reason)
        )
    if "typecheck" in scripts:
        reason = "package.json has a typecheck script"
        checks.append(
            CheckCommand(
                kind="typecheck", argv=("npm", "run", "--silent", "typecheck"), reason=reason
            )
        )
    return checks


def detect_checks(project_root: Path) -> list[CheckCommand]:
    """Detect lint and typecheck commands from project config files.

    The function inspects files only. It never runs a command.
    A file with invalid TOML or JSON produces no checks for that source.
    """
    checks: list[CheckCommand] = []

    content = _read_project_config("pyproject.toml", project_root)
    if content is not None:
        try:
            data = tomllib.loads(content)
        except tomllib.TOMLDecodeError:
            data = None
        if isinstance(data, dict):
            checks.extend(_detect_pyproject_checks(data))

    content = _read_project_config("package.json", project_root)
    if content is not None:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
        checks.extend(_detect_package_json_checks(data))

    return checks


class RunChecksTool(Tool):
    """Run project lint and typecheck commands."""

    name = "run_checks"
    description = "Run project lint and typecheck commands in order."
    risk_level = RiskLevel.process_execution
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["lint", "typecheck", "all"]},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }

    def __init__(self, shell_tool: ShellTool | None = None) -> None:
        self._shell = shell_tool or ShellTool()

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        kind = str(arguments["kind"])
        checks = detect_checks(context.project_root)
        if kind != "all":
            checks = [check for check in checks if check.kind == kind]
        if not checks:
            raise ToolError("No project check commands were detected for the requested kind")

        results: list[dict[str, Any]] = []
        for check in checks:
            command = shlex.join(check.argv)
            result = await self._shell.execute(context, {"command": command})
            results.append(
                {
                    "kind": check.kind,
                    "command": command,
                    "exit_code": result["exit_code"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "truncated": result["truncated"],
                }
            )
        return {"results": results}
