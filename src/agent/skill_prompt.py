"""Build a skills section for a system prompt.

The module loads skills from project and shared directories. It
builds a short section and returns skill bodies by name. The module
does not read the database: the switches of each project are applied
in ``skill_library``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agent.skills import Skill, SkillError, scan_skills, skill_index

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


def _load_directory(directory: Path) -> tuple[tuple[Skill, ...], tuple[str, ...]]:
    """Load the valid skills of one directory and the errors of the broken ones."""
    if not directory.exists():
        return (), ()
    found, errors = scan_skills(directory)
    return tuple(found), tuple(errors)


def project_skills_dir(project_root: Path) -> Path:
    """Return the directory of the skills that belong to one project."""
    return project_root / _PROJECT_SKILLS_REL


def project_skills_safe(project_root: Path) -> bool:
    """Check that the project skills directory, when it exists, stays inside the project."""
    return not project_skills_dir(project_root).exists() or _is_safe_project_skills(project_root)


def load_scoped_skills(
    project_root: Path | None, shared_root: Path | None
) -> tuple[tuple[Skill, ...], tuple[Skill, ...], tuple[str, ...]]:
    """Return the shared skills, the project skills and the load errors, without merging."""
    errors: list[str] = []
    project_skills: tuple[Skill, ...] = ()
    if project_root is not None:
        project_dir = project_skills_dir(project_root)
        if not project_skills_safe(project_root):
            errors.append(f"Project skills directory is not safe: {project_dir}")
        else:
            project_skills, found_errors = _load_directory(project_dir)
            errors.extend(found_errors)
    shared_skills: tuple[Skill, ...] = ()
    if shared_root is not None:
        shared_skills, found_errors = _load_directory(shared_root)
        errors.extend(found_errors)
    return shared_skills, project_skills, tuple(errors)


def merge_skills(shared: Sequence[Skill], project: Sequence[Skill]) -> tuple[Skill, ...]:
    """Merge two skill lists; a project skill replaces a shared skill with the same name."""
    by_name = {skill.name: skill for skill in shared}
    by_name.update((skill.name, skill) for skill in project)
    return tuple(sorted(by_name.values(), key=lambda s: s.name))


def load_skill_set(project_root: Path, shared_root: Path | None) -> SkillSet:
    """Load skills from the project and shared directories."""
    shared, project, errors = load_scoped_skills(project_root, shared_root)
    return SkillSet(skills=merge_skills(shared, project), errors=errors)


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
