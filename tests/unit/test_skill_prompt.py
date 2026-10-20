"""Unit tests for the skill prompt module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.skill_prompt import (
    SkillSet,
    load_skill_set,
    skill_instructions,
    skill_prompt_section,
)
from agent.skills import Skill, SkillError

_HEADER = "---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def _write_skill(root: Path, name: str, description: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = _HEADER.format(name=name, description=description, body=body)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def _make_skill(name: str, body: str = "body") -> Skill:
    return Skill(
        name=name,
        description="tool",
        version="",
        tools=(),
        root=Path(f"/skills/{name}"),
        body=body,
    )


def test_load_shared_only(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_skill(shared, "alpha", "Alpha tool", "alpha body")

    result = load_skill_set(tmp_path, shared)

    assert [s.name for s in result.skills] == ["alpha"]
    assert result.errors == ()


def test_load_project_only(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / ".agent" / "skills").mkdir(parents=True)
    _write_skill(project_root / ".agent" / "skills", "beta", "Beta tool", "beta body")

    result = load_skill_set(project_root, None)

    assert [s.name for s in result.skills] == ["beta"]
    assert result.errors == ()


def test_project_overrides_shared(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_skill(shared, "common", "Shared version", "shared body")
    project_root = tmp_path
    (project_root / ".agent" / "skills").mkdir(parents=True)
    _write_skill(project_root / ".agent" / "skills", "common", "Project version", "project body")

    result = load_skill_set(project_root, shared)

    assert len(result.skills) == 1
    assert result.skills[0].name == "common"
    assert result.skills[0].body == "project body"


def test_missing_directories_are_not_errors(tmp_path: Path) -> None:
    result = load_skill_set(tmp_path, None)

    assert result.skills == ()
    assert result.errors == ()


def test_missing_shared_only(tmp_path: Path) -> None:
    missing = tmp_path / "missing-shared"
    result = load_skill_set(tmp_path, missing)

    assert result.skills == ()
    assert result.errors == ()


def test_error_in_shared_keeps_project(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_skill(shared, "good-shared", "Good shared", "shared body")
    bad_shared = shared / "bad"
    bad_shared.mkdir()
    (bad_shared / "SKILL.md").write_text("not a valid skill", encoding="utf-8")

    project_root = tmp_path
    (project_root / ".agent" / "skills").mkdir(parents=True)
    _write_skill(project_root / ".agent" / "skills", "good-proj", "Good project", "p body")

    result = load_skill_set(project_root, shared)

    assert [s.name for s in result.skills] == ["good-proj"]
    assert len(result.errors) == 1


def test_error_in_project_keeps_shared(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_skill(shared, "good-shared", "Good shared", "shared body")

    project_root = tmp_path
    (project_root / ".agent" / "skills").mkdir(parents=True)
    bad_dir = project_root / ".agent" / "skills" / "bad"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_text("not a valid skill", encoding="utf-8")

    result = load_skill_set(project_root, shared)

    assert [s.name for s in result.skills] == ["good-shared"]
    assert len(result.errors) == 1


def test_project_skills_symlink_is_an_error(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_skill(shared, "good-shared", "Good shared", "shared body")

    project_root = tmp_path
    (project_root / ".agent").mkdir()
    os.symlink(tmp_path / "outside", project_root / ".agent" / "skills", target_is_directory=True)
    (tmp_path / "outside").mkdir()

    result = load_skill_set(project_root, shared)

    assert [s.name for s in result.skills] == ["good-shared"]
    assert len(result.errors) == 1


def test_project_skills_escape_is_an_error(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_skill(shared, "good-shared", "Good shared", "shared body")

    project_root = tmp_path
    outside = tmp_path / "outside-skills"
    outside.mkdir(exist_ok=True)
    (project_root / ".agent").mkdir()
    os.symlink(outside, project_root / ".agent" / "skills", target_is_directory=True)

    result = load_skill_set(project_root, shared)

    assert [s.name for s in result.skills] == ["good-shared"]
    assert len(result.errors) == 1


def test_empty_section() -> None:
    skill_set = SkillSet(skills=(), errors=())
    assert skill_prompt_section(skill_set) == ""


def test_section_with_skills() -> None:
    skill_set = SkillSet(
        skills=(
            _make_skill("alpha", "alpha body"),
            _make_skill("beta", "beta body"),
        ),
        errors=(),
    )
    section = skill_prompt_section(skill_set)
    assert section.startswith("Available skills:")
    assert "- alpha: tool" in section
    assert "- beta: tool" in section
    assert section.endswith("Ask for a skill by name to get its full instructions.")


def test_section_truncation() -> None:
    skills = tuple(_make_skill(f"skill-{i}", "b") for i in range(50))
    skill_set = SkillSet(skills=skills, errors=())
    section = skill_prompt_section(skill_set, max_chars=200)
    assert section.startswith("Available skills:")
    assert "... and" in section
    assert "more skills" in section


def test_section_truncation_keeps_only_full_skills() -> None:
    skills = tuple(_make_skill(f"skill-{i}", "b") for i in range(10))
    skill_set = SkillSet(skills=skills, errors=())
    section = skill_prompt_section(skill_set, max_chars=150)
    for line in section.split("\n"):
        if line.startswith("- "):
            assert "skill-" in line
    assert "... and" in section


def test_instructions_returns_body() -> None:
    skill_set = SkillSet(
        skills=(_make_skill("alpha", "alpha instructions"),),
        errors=(),
    )
    assert skill_instructions(skill_set, "alpha") == "alpha instructions"


def test_instructions_unknown_name() -> None:
    skill_set = SkillSet(skills=(), errors=())
    with pytest.raises(SkillError, match="Unknown skill"):
        skill_instructions(skill_set, "missing")


@pytest.mark.parametrize("max_chars", [120, 200, 333, 500, 1000])
def test_section_never_exceeds_the_limit(tmp_path: Path, max_chars: int) -> None:
    shared = tmp_path / "shared"
    for index in range(30):
        folder = shared / f"skill-{index:02d}"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: skill-{index:02d}\n"
            f"description: Skill number {index} does a task\n---\nBody.\n",
            encoding="utf-8",
        )
    section = skill_prompt_section(load_skill_set(tmp_path, shared), max_chars=max_chars)
    assert len(section) <= max_chars
    assert section.endswith("Ask for a skill by name to get its full instructions.")
