from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import agent.tools.skill_tools as skill_tools
from agent.skill_prompt import SkillSet, load_skill_set
from agent.skills import Skill
from agent.tools.base import ToolContext, ToolError
from agent.tools.skill_tools import ReadSkillTool


def context(root: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=root)


async def test_read_skill_metadata() -> None:
    skill_set = SkillSet(skills=(), errors=())
    tool = ReadSkillTool(skill_set)

    assert tool.name == "read_skill"
    assert tool.risk_level == "read_only"
    assert tool.parallel_safe is True
    assert tool.allowed_modes == frozenset({"dev", "ask"})
    assert tool.input_schema["required"] == ["name"]
    assert tool.input_schema["additionalProperties"] is False


async def test_read_skill_success() -> None:
    skill = Skill(
        name="test_skill",
        description="desc",
        version="1.0",
        tools=(),
        root=Path("."),
        body="Instruction content",
    )
    skill_set = SkillSet(skills=(skill,), errors=())
    tool = ReadSkillTool(skill_set)

    result = await tool.execute(context(Path(".")), {"name": "test_skill"})
    assert result == {"name": "test_skill", "instructions": "Instruction content"}


async def test_read_skill_unknown_name() -> None:
    skill_set = SkillSet(skills=(), errors=())
    tool = ReadSkillTool(skill_set)

    with pytest.raises(ToolError) as excinfo:
        await tool.execute(context(Path(".")), {"name": "unknown"})
    assert "Unknown skill name" in str(excinfo.value)


async def test_read_skill_malformed_input() -> None:
    skill_set = SkillSet(skills=(), errors=())
    tool = ReadSkillTool(skill_set)

    with pytest.raises(ToolError):
        await tool.execute(context(Path(".")), {})

    with pytest.raises(ToolError):
        await tool.execute(context(Path(".")), {"name": 123})

    with pytest.raises(ToolError):
        await tool.execute(context(Path(".")), {"name": "skill", "extra": "value"})


async def test_read_skill_empty_set() -> None:
    skill_set = SkillSet(skills=(), errors=())
    tool = ReadSkillTool(skill_set)

    with pytest.raises(ToolError):
        await tool.execute(context(Path(".")), {"name": "any"})


async def test_read_skill_snapshot_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    skill_dir = project / ".agent" / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: test-skill\ndescription: desc\nversion: '1'\ntools: []\n---\nOriginal Body"
    )

    skill_set = load_skill_set(project, None)
    tool = ReadSkillTool(skill_set)

    skill_file.write_text("Modified Body")

    def fail_read(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("Disk read attempted")

    monkeypatch.setattr(Path, "read_text", fail_read)
    monkeypatch.setattr(Path, "open", fail_read)

    result = await tool.execute(context(project), {"name": "test-skill"})
    assert result["instructions"] == "Original Body"


async def test_read_skill_does_not_mask_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    skill = Skill(
        name="test_skill",
        description="desc",
        version="1.0",
        tools=(),
        root=Path("."),
        body="Instruction content",
    )
    skill_set = SkillSet(skills=(skill,), errors=())
    tool = ReadSkillTool(skill_set)

    def raise_runtime(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("Unexpected system failure")

    monkeypatch.setattr(skill_tools, "skill_instructions", raise_runtime)

    with pytest.raises(RuntimeError, match="Unexpected system failure") as caught:
        await tool.execute(context(Path(".")), {"name": "test_skill"})
    assert type(caught.value) is RuntimeError


def _skill_with_files(tmp_path: Path) -> SkillSet:
    project = tmp_path / "project"
    skill_dir = project / ".agent" / "skills" / "pdf"
    (skill_dir / "docs").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: pdf\ndescription: PDF\n---\nRead docs/a.md")
    (skill_dir / "docs" / "a.md").write_text("Page rules", encoding="utf-8")
    (skill_dir / "binary.bin").write_bytes(b"\xff\xfe")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    os.symlink(tmp_path / "secret.txt", skill_dir / "link.txt")
    return load_skill_set(project, None)


async def test_read_skill_lists_files_and_reads_one(tmp_path: Path) -> None:
    tool = ReadSkillTool(_skill_with_files(tmp_path))

    main = await tool.execute(context(tmp_path), {"name": "pdf"})
    # Links and SKILL.md itself are not in the list.
    assert main["files"] == ["binary.bin", "docs/a.md"]
    extra = await tool.execute(context(tmp_path), {"name": "pdf", "file": "docs/a.md"})
    assert extra == {"name": "pdf", "file": "docs/a.md", "content": "Page rules"}


@pytest.mark.parametrize(
    ("file", "message"),
    [
        ("link.txt", "has no file"),
        ("../../../../secret.txt", "has no file"),
        ("missing.md", "has no file"),
        ("binary.bin", "not UTF-8"),
    ],
)
async def test_read_skill_file_stays_inside_the_skill(
    tmp_path: Path, file: str, message: str
) -> None:
    tool = ReadSkillTool(_skill_with_files(tmp_path))

    with pytest.raises(ToolError, match=message):
        await tool.execute(context(tmp_path), {"name": "pdf", "file": file})
