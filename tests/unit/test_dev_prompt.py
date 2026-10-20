from __future__ import annotations

from pathlib import Path

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "dev.md"


def test_dev_prompt_exists() -> None:
    assert PROMPT_FILE.is_file()


def _prompt_text() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def test_dev_prompt_not_longer_than_60_lines() -> None:
    lines = [line for line in _prompt_text().splitlines() if line.strip()]
    assert len(lines) <= 60


def test_dev_prompt_mentions_key_tools() -> None:
    content = _prompt_text()
    for tool in ("search_files", "read_file", "edit_file", "todo_write"):
        assert tool in content


def test_dev_prompt_requires_edit_and_check_before_finish() -> None:
    content = _prompt_text()
    assert "правк" in content
    assert "проверк" in content
    assert "не закончена" in content
