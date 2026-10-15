"""Load and validate skill definitions from skill directories.

A skill is a directory that contains a SKILL.md file. The file starts
with a YAML header between two "---" lines, followed by instruction
text. The module is standalone and does not depend on the runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_MAX_SKILL_BYTES = 256 * 1024
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SKILL_MD = "SKILL.md"


class SkillError(ValueError):
    """The skill definition is invalid."""


@dataclass(frozen=True)
class Skill:
    """A loaded skill definition."""

    name: str
    description: str
    version: str
    tools: tuple[str, ...]
    root: Path
    body: str


def _read_skill_md(directory: Path) -> str:
    """Read and validate the SKILL.md file in a skill directory."""
    skill_file = directory / _SKILL_MD
    if skill_file.is_symlink():
        raise SkillError(f"{_SKILL_MD} is a symbolic link in {directory}")
    if not skill_file.exists():
        raise SkillError(f"No {_SKILL_MD} file in {directory}")
    if not skill_file.is_file():
        raise SkillError(f"{_SKILL_MD} is not a regular file in {directory}")
    size = skill_file.stat().st_size
    if size > _MAX_SKILL_BYTES:
        raise SkillError(f"{_SKILL_MD} is larger than {_MAX_SKILL_BYTES} bytes in {directory}")
    try:
        return skill_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError(f"{_SKILL_MD} is not UTF-8 text in {directory}") from exc


def _parse_header(content: str, directory: Path) -> tuple[dict[str, object], str]:
    """Split the YAML header from the instruction body."""
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{_SKILL_MD} in {directory} has no YAML header start")
    end_index = -1
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index == -1:
        raise SkillError(f"{_SKILL_MD} in {directory} has no closing header delimiter")
    header_text = "\n".join(lines[1:end_index]).strip()
    if not header_text:
        raise SkillError(f"{_SKILL_MD} in {directory} has an empty YAML header")
    try:
        data = yaml.safe_load(header_text)
    except yaml.YAMLError as exc:
        raise SkillError(f"Invalid YAML header in {_SKILL_MD} in {directory}: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillError(f"YAML header in {_SKILL_MD} in {directory} is not a mapping")
    body = "\n".join(lines[end_index + 1 :]).strip()
    return data, body


def _validate_header(
    data: dict[str, object], directory: Path
) -> tuple[str, str, str, tuple[str, ...]]:
    """Validate header fields and return normalized values."""
    name = data.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise SkillError(f"Invalid skill name in {directory}: {name!r}")
    description = data.get("description")
    if not isinstance(description, str):
        raise SkillError(f"Skill {name} has no description in {directory}")
    if not (1 <= len(description) <= 1024):
        raise SkillError(f"Skill {name} description length is out of range in {directory}")
    version = data.get("version", "")
    if not isinstance(version, str):
        raise SkillError(f"Skill {name} has a non-string version in {directory}")
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        raise SkillError(f"Skill {name} tools is not a list in {directory}")
    tool_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, str):
            raise SkillError(f"Skill {name} has a non-string tool in {directory}")
        tool_names.append(tool)
    return name, description, version, tuple(tool_names)


def load_skill(directory: Path) -> Skill:
    """Load a skill from its directory."""
    root = directory.resolve()
    content = _read_skill_md(directory)
    data, body = _parse_header(content, directory)
    name, description, version, tools = _validate_header(data, directory)
    return Skill(
        name=name,
        description=description,
        version=version,
        tools=tools,
        root=root,
        body=body,
    )


def discover_skills(root: Path) -> list[Skill]:
    """Discover all skills in first-level subdirectories of root."""
    if not root.is_dir():
        raise SkillError(f"Skill root is not a directory: {root}")
    skills: list[Skill] = []
    seen: set[str] = set()
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if not (entry / _SKILL_MD).is_file():
            continue
        skill = load_skill(entry)
        if skill.name in seen:
            raise SkillError(f"Duplicate skill name: {skill.name}")
        seen.add(skill.name)
        skills.append(skill)
    skills.sort(key=lambda s: s.name)
    return skills


def skill_file(skill: Skill, relative: str) -> Path:
    """Return the path of a file inside the skill directory.

    The relative path must stay inside the skill root. Parent
    traversal, absolute paths and symlink escapes are rejected.
    """
    if not relative or "\x00" in relative:
        raise SkillError("Invalid relative path for skill file")
    candidate = skill.root / relative
    try:
        resolved = candidate.resolve()
    except (FileNotFoundError, RuntimeError) as exc:
        raise SkillError(f"Cannot resolve path inside skill root: {relative}") from exc
    try:
        resolved.relative_to(skill.root)
    except ValueError as exc:
        raise SkillError(f"Path escapes skill root: {relative}") from exc
    # Return the checked path, not the original one: a link could change after the check.
    return resolved


def skill_index(skills: list[Skill]) -> str:
    """Build a short skill list for a system prompt."""
    # One line per skill: line breaks in a description must not break the list.
    lines = [f"- {item.name}: {' '.join(item.description.split())}" for item in skills]
    return "\n".join(lines)
