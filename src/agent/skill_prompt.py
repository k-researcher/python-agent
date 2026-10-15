"""Build a skills section for a system prompt.

The module loads skills from project and shared directories. It
builds a short section and returns skill bodies by name. The module
is standalone and does not connect to sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.skills import Skill, SkillError, discover_skills, skill_index

_PROJECT_SKILLS_REL = Path(".agent/skills")
_HEADER = "Available skills:"
_FOOTER = "Ask for a skill by name to get its full instructions."
_MORE_FMT = "... and {count} more skills"


@dataclass(frozen=True)
class SkillSet:
    """A validated collection of skills and their load errors."""

    skills: tuple[Skill, ...]
    errors: tuple[str, ...]


def _is_safe_project_skills(project_root: Path) -> bool:
    """Check that the project skills directory is a real subdirectory."""
    skills_dir = project_root / _PROJECT_SKILLS_REL
    if skills_dir.is_symlink():
        return False
    try:
        resolved = skills_dir.resolve()
    except (FileNotFoundError, RuntimeError):
        return False
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        return False
    return True


def _load_directory(directory: Path) -> tuple[tuple[Skill, ...], str]:
    """Load skills from one directory and return skills and one error."""
    if not directory.exists():
        return (), ""
    try:
        found = discover_skills(directory)
    except SkillError as exc:
        return (), str(exc)
    return tuple(found), ""


def load_skill_set(project_root: Path, shared_root: Path | None) -> SkillSet:
    """Load skills from the project and shared directories."""
    errors: list[str] = []
    project_skills: list[Skill] = []

    project_dir = project_root / _PROJECT_SKILLS_REL
    if project_dir.exists():
        if not _is_safe_project_skills(project_root):
            errors.append(f"Project skills directory is not safe: {project_dir}")
        else:
            found, error = _load_directory(project_dir)
            project_skills = list(found)
            if error:
                errors.append(error)

    shared_skills: list[Skill] = []
    if shared_root is not None:
        found, error = _load_directory(shared_root)
        shared_skills = list(found)
        if error:
            errors.append(error)

    by_name: dict[str, Skill] = {}
    for skill in shared_skills:
        by_name[skill.name] = skill
    for skill in project_skills:
        by_name[skill.name] = skill

    merged = tuple(sorted(by_name.values(), key=lambda s: s.name))
    return SkillSet(skills=merged, errors=tuple(errors))


def skill_prompt_section(skill_set: SkillSet, *, max_chars: int = 4000) -> str:
    """Build a compact skills section for a system prompt."""
    if not skill_set.skills:
        return ""
    lines = skill_index(list(skill_set.skills)).split("\n")
    # Keep skills that fit entirely. Room for the "... and N more" line is kept, so the
    # whole section never exceeds max_chars.
    reserve = len(_MORE_FMT.format(count=len(lines))) + 1
    total = len(_HEADER) + 1 + len(_FOOTER)
    kept: list[str] = []
    for index, line in enumerate(lines):
        needed = len(line) + 1
        last = index == len(lines) - 1
        if total + needed + (0 if last else reserve) > max_chars:
            break
        kept.append(line)
        total += needed
    parts = [_HEADER, *kept]
    if len(kept) < len(lines):
        more = len(lines) - len(kept)
        parts.append(_MORE_FMT.format(count=more))
    parts.append(_FOOTER)
    return "\n".join(parts)


def skill_instructions(skill_set: SkillSet, name: str) -> str:
    """Return the instruction body of a skill by name."""
    for skill in skill_set.skills:
        if skill.name == name:
            return skill.body
    raise SkillError(f"Unknown skill name: {name}")
