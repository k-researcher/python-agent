"""Unit tests for the skill loading module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.skills import Skill, SkillError, discover_skills, load_skill, skill_file, skill_index

_VALID_HEADER = """---
name: example-skill
description: An example skill for testing
version: "1.0.0"
tools:
  - read
  - write
---

Do the thing step by step.
Follow the instructions.
"""


def _write_skill(directory: Path, name: str, content: str) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def test_load_valid_skill(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "alpha", _VALID_HEADER)
    skill = load_skill(skill_dir)
    assert isinstance(skill, Skill)
    assert skill.name == "example-skill"
    assert skill.description == "An example skill for testing"
    assert skill.version == "1.0.0"
    assert skill.tools == ("read", "write")
    assert skill.root == skill_dir.resolve()
    assert "Do the thing step by step." in skill.body


def test_skill_defaults_when_optional_fields_missing(tmp_path: Path) -> None:
    content = "---\nname: minimal\ndescription: Minimal skill\n---\nBody text here.\n"
    skill_dir = _write_skill(tmp_path, "minimal", content)
    skill = load_skill(skill_dir)
    assert skill.version == ""
    assert skill.tools == ()


def test_missing_skill_md(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SkillError, match="No SKILL.md"):
        load_skill(empty_dir)


def test_header_missing(tmp_path: Path) -> None:
    content = "name: x\ndescription: y\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "no-header", content)
    with pytest.raises(SkillError, match="no YAML header"):
        load_skill(skill_dir)


def test_header_no_closing_delimiter(tmp_path: Path) -> None:
    content = "---\nname: x\ndescription: y\nbody\n"
    skill_dir = _write_skill(tmp_path, "no-close", content)
    with pytest.raises(SkillError, match="no closing header"):
        load_skill(skill_dir)


def test_empty_yaml_header(tmp_path: Path) -> None:
    content = "---\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "empty-header", content)
    with pytest.raises(SkillError, match="empty YAML header"):
        load_skill(skill_dir)


def test_invalid_yaml(tmp_path: Path) -> None:
    content = "---\nname: [unclosed\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "bad-yaml", content)
    with pytest.raises(SkillError, match="Invalid YAML"):
        load_skill(skill_dir)


def test_header_not_mapping(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "not-map", "---\n- a\n- b\n---\nbody\n")
    with pytest.raises(SkillError, match="not a mapping"):
        load_skill(skill_dir)


def test_invalid_name_pattern(tmp_path: Path) -> None:
    content = "---\nname: BadName\ndescription: x\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "bad-name", content)
    with pytest.raises(SkillError, match="Invalid skill name"):
        load_skill(skill_dir)


def test_missing_name(tmp_path: Path) -> None:
    content = "---\ndescription: x\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "no-name", content)
    with pytest.raises(SkillError, match="Invalid skill name"):
        load_skill(skill_dir)


def test_missing_description(tmp_path: Path) -> None:
    content = "---\nname: valid-name\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "no-desc", content)
    with pytest.raises(SkillError, match="no description"):
        load_skill(skill_dir)


def test_non_string_description(tmp_path: Path) -> None:
    content = "---\nname: valid-name\ndescription: 123\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "desc-num", content)
    with pytest.raises(SkillError, match="no description"):
        load_skill(skill_dir)


def test_empty_description(tmp_path: Path) -> None:
    content = "---\nname: valid-name\ndescription: ''\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "desc-empty", content)
    with pytest.raises(SkillError, match="description length"):
        load_skill(skill_dir)


def test_tool_too_long_description(tmp_path: Path) -> None:
    content = f"---\nname: valid-name\ndescription: {'x' * 1025}\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "desc-long", content)
    with pytest.raises(SkillError, match="description length"):
        load_skill(skill_dir)


def test_non_string_version(tmp_path: Path) -> None:
    content = "---\nname: valid-name\ndescription: ok\nversion: 1.0\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "ver-num", content)
    with pytest.raises(SkillError, match="non-string version"):
        load_skill(skill_dir)


def test_tools_not_list(tmp_path: Path) -> None:
    content = "---\nname: valid-name\ndescription: ok\ntools: read\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "tools-str", content)
    with pytest.raises(SkillError, match="tools is not a list"):
        load_skill(skill_dir)


def test_tools_non_string_entry(tmp_path: Path) -> None:
    content = "---\nname: valid-name\ndescription: ok\ntools:\n  - 42\n---\nbody\n"
    skill_dir = _write_skill(tmp_path, "tools-int", content)
    with pytest.raises(SkillError, match="non-string tool"):
        load_skill(skill_dir)


def test_skill_too_large(tmp_path: Path) -> None:
    skill_dir = tmp_path / "big"
    skill_dir.mkdir()
    content = "---\nname: big-skill\ndescription: big\n---\n" + "x" * (256 * 1024 + 1)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    with pytest.raises(SkillError, match="larger than"):
        load_skill(skill_dir)


def test_skill_md_symlink(tmp_path: Path) -> None:
    skill_dir = tmp_path / "linked"
    skill_dir.mkdir()
    target = tmp_path / "SKILL.md"
    target.write_text(_VALID_HEADER, encoding="utf-8")
    os.symlink(target, skill_dir / "SKILL.md")
    with pytest.raises(SkillError, match="symbolic link"):
        load_skill(skill_dir)


def test_discover_sorts_skills(tmp_path: Path) -> None:
    header = "---\nname: b-skill\ndescription: B\n---\nbody\n"
    _write_skill(tmp_path, "b-skill", header)
    header_a = "---\nname: a-skill\ndescription: A\n---\nbody\n"
    _write_skill(tmp_path, "a-skill", header_a)
    header_c = "---\nname: c-skill\ndescription: C\n---\nbody\n"
    _write_skill(tmp_path, "c-skill", header_c)
    # A non-skill directory must be ignored.
    (tmp_path / "plain-dir").mkdir()
    (tmp_path / "plain-dir" / "notes.txt").write_text("data", encoding="utf-8")

    skills = discover_skills(tmp_path)
    assert [s.name for s in skills] == ["a-skill", "b-skill", "c-skill"]


def test_discover_duplicate_skill_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "one", _VALID_HEADER)
    _write_skill(tmp_path, "two", _VALID_HEADER)
    with pytest.raises(SkillError, match="Duplicate skill name"):
        discover_skills(tmp_path)


def test_discover_skips_symlink_directories(tmp_path: Path) -> None:
    _write_skill(tmp_path, "real", _VALID_HEADER)
    target = tmp_path / "real"
    os.symlink(target, tmp_path / "alias")
    skills = discover_skills(tmp_path)
    assert [s.name for s in skills] == ["example-skill"]


def test_discover_error_is_not_hidden(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "good",
        "---\nname: alpha-good\ndescription: fine\n---\nbody text\n",
    )
    _write_skill(
        tmp_path,
        "bad",
        "---\nname: broken\ndescription: bad\n",
    )
    with pytest.raises(SkillError, match="no closing header"):
        discover_skills(tmp_path)


def test_skill_file_returns_internal_path(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "alpha", _VALID_HEADER)
    skill = load_skill(skill_dir)
    (skill.root / "data.txt").write_text("hello", encoding="utf-8")
    path = skill_file(skill, "data.txt")
    assert path == skill.root / "data.txt"


def test_skill_file_parent_traversal_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "alpha", _VALID_HEADER)
    skill = load_skill(skill_dir)
    with pytest.raises(SkillError, match="escapes skill root"):
        skill_file(skill, "../outside.txt")


def test_skill_file_absolute_path_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "alpha", _VALID_HEADER)
    skill = load_skill(skill_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(SkillError, match="escapes skill root"):
        skill_file(skill, str(outside))


def test_skill_file_symlink_escape_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "alpha", _VALID_HEADER)
    skill = load_skill(skill_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, skill.root / "link.txt")
    with pytest.raises(SkillError, match="escapes skill root"):
        skill_file(skill, "link.txt")


def test_skill_file_empty_path_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "alpha", _VALID_HEADER)
    skill = load_skill(skill_dir)
    with pytest.raises(SkillError, match="Invalid relative path"):
        skill_file(skill, "")


def test_skill_index_produces_lines() -> None:
    skills = [
        Skill(
            name="alpha",
            description="Alpha tool",
            version="",
            tools=(),
            root=Path("/tmp/skills/alpha"),
            body="body",
        ),
        Skill(
            name="beta",
            description="Beta tool",
            version="1.0",
            tools=("run",),
            root=Path("/tmp/skills/beta"),
            body="body",
        ),
    ]
    assert skill_index(skills) == "- alpha: Alpha tool\n- beta: Beta tool"


def test_skill_index_empty_list() -> None:
    assert skill_index([]) == ""


def test_skill_file_returns_the_checked_path(tmp_path: Path) -> None:
    skill = load_skill(_write_skill(tmp_path, "alpha", _VALID_HEADER))
    (skill.root / "real.txt").write_text("data", encoding="utf-8")
    os.symlink(skill.root / "real.txt", skill.root / "inner-link.txt")
    # A link inside the skill is allowed, and the result is the resolved target.
    assert skill_file(skill, "inner-link.txt") == (skill.root / "real.txt").resolve()


def test_non_utf8_skill_md_is_a_skill_error(tmp_path: Path) -> None:
    skill_dir = tmp_path / "binary"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_bytes(b"---\nname: x\n---\n\xff\xfe")
    with pytest.raises(SkillError, match="UTF-8"):
        load_skill(skill_dir)


def test_skill_index_keeps_one_line_per_skill(tmp_path: Path) -> None:
    content = '---\nname: multi\ndescription: "first line\\nsecond line"\n---\nBody.\n'
    skill = load_skill(_write_skill(tmp_path, "multi", content))
    assert skill_index([skill]) == "- multi: first line second line"
