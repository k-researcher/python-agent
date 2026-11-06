"""Load and validate skill definitions from skill directories.

A skill is a directory that contains a SKILL.md file. The file starts
with a YAML header between two "---" lines, followed by instruction
text. The module is standalone and does not depend on the runtime.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_SKILL_BYTES = 256 * 1024
MAX_LISTED_FILES = 200
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SKILL_MD = "SKILL.md"


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
    # Other files of the skill, relative to its root, read when the skill is loaded.
    files: tuple[str, ...] = ()


def _read_skill_md(directory: Path) -> str:
    """Read and validate the SKILL.md file in a skill directory."""
    skill_file = directory / SKILL_MD
    if skill_file.is_symlink():
        raise SkillError(f"{SKILL_MD} is a symbolic link in {directory}")
    if not skill_file.exists():
        raise SkillError(f"No {SKILL_MD} file in {directory}")
    if not skill_file.is_file():
        raise SkillError(f"{SKILL_MD} is not a regular file in {directory}")
    size = skill_file.stat().st_size
    if size > MAX_SKILL_BYTES:
        raise SkillError(f"{SKILL_MD} is larger than {MAX_SKILL_BYTES} bytes in {directory}")
    try:
        return skill_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError(f"{SKILL_MD} is not UTF-8 text in {directory}") from exc


def parse_skill_text(content: str, directory: Path) -> tuple[dict[str, object], str]:
    """Split the YAML header from the instruction body; ``directory`` names the source."""
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{SKILL_MD} in {directory} has no YAML header start")
    end_index = -1
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index == -1:
        raise SkillError(f"{SKILL_MD} in {directory} has no closing header delimiter")
    header_text = "\n".join(lines[1:end_index]).strip()
    if not header_text:
        raise SkillError(f"{SKILL_MD} in {directory} has an empty YAML header")
    try:
        data = yaml.safe_load(header_text)
    except yaml.YAMLError as exc:
        raise SkillError(f"Invalid YAML header in {SKILL_MD} in {directory}: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillError(f"YAML header in {SKILL_MD} in {directory} is not a mapping")
    body = "\n".join(lines[end_index + 1 :]).strip()
    return data, body


def validate_name(name: str) -> str:
    """Return the name when it is a valid skill name: lowercase letters, digits and hyphens."""
    if not _NAME_RE.fullmatch(name):
        raise SkillError(f"Invalid skill name: {name!r}")
    return name


def validate_header(
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
    data, body = parse_skill_text(content, directory)
    name, description, version, tools = validate_header(data, directory)
    return Skill(
        name=name,
        description=description,
        version=version,
        tools=tools,
        root=root,
        body=body,
        files=_list_files(root),
    )


def _list_files(root: Path) -> tuple[str, ...]:
    """List the files of a skill except SKILL.md, without hidden files and links."""
    found: list[str] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith("."))
        for name in sorted(files):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if name.startswith(".") or relative == SKILL_MD or path.is_symlink():
                continue
            found.append(relative)
            if len(found) >= MAX_LISTED_FILES:
                return tuple(found)
    return tuple(found)


def scan_skills(root: Path) -> tuple[list[Skill], list[str]]:
    """Load every skill in first-level subdirectories of root.

    A broken skill does not hide the other skills: its error is returned instead.
    Hidden directories are skipped, because an import prepares a skill there.
    """
    if not root.is_dir():
        return [], [f"Skill root is not a directory: {root}"]
    skills: list[Skill] = []
    errors: list[str] = []
    seen: set[str] = set()
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or not entry.is_dir() or entry.is_symlink():
            continue
        if not (entry / SKILL_MD).is_file():
            continue
        try:
            skill = load_skill(entry)
        except SkillError as exc:
            errors.append(str(exc))
            continue
        if skill.name in seen:
            errors.append(f"Duplicate skill name: {skill.name}")
            continue
        seen.add(skill.name)
        skills.append(skill)
    skills.sort(key=lambda s: s.name)
    return skills, errors


def discover_skills(root: Path) -> list[Skill]:
    """Discover all skills in first-level subdirectories of root; raise on the first error."""
    if not root.is_dir():
        raise SkillError(f"Skill root is not a directory: {root}")
    skills, errors = scan_skills(root)
    if errors:
        raise SkillError(errors[0])
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
