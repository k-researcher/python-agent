from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.config import Settings
from agent.modes import MODES, ROLE_PROFILES, tool_profile
from agent.tools import ToolRegistry
from agent.tools.base import ToolError

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def test_every_mode_has_exactly_one_known_profile() -> None:
    assert set(MODES) == set(ROLE_PROFILES)
    assert set(ROLE_PROFILES.values()) == {"dev", "ask"}
    assert tool_profile("arch") == "ask"
    assert tool_profile("pm") == "dev"
    # An unknown mode keeps its own name, so no tool allows it.
    assert tool_profile("root") == "root"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_has_a_prompt(mode: str) -> None:
    assert (PROMPTS / f"{mode}.md").is_file()


@pytest.mark.parametrize("mode", ["arch", "pm", "qa", "devops", "cybersec", "analyst"])
def test_role_prompt_lists_only_available_tools(mode: str) -> None:
    registry = ToolRegistry(settings=Settings(allow_network_tools=True))
    text = (PROMPTS / f"{mode}.md").read_text()
    tool_groups = re.findall(r"^- ([a-z_]+(?:, [a-z_]+)*) —", text, re.MULTILINE)
    assert tool_groups
    for group in tool_groups:
        for name in group.split(", "):
            assert registry.get(name, mode).name == name


def _tool_names(registry: ToolRegistry, mode: str) -> set[str]:
    return {item["function"]["name"] for item in registry.definitions(mode)}


@pytest.mark.parametrize("mode", [mode for mode, profile in ROLE_PROFILES.items()])
def test_a_role_gets_exactly_the_tools_of_its_profile(mode: str) -> None:
    registry = ToolRegistry(settings=Settings(allow_network_tools=True))
    assert _tool_names(registry, mode) == _tool_names(registry, ROLE_PROFILES[mode])


def test_read_only_roles_cannot_change_files_or_run_commands() -> None:
    registry = ToolRegistry()
    for mode in ("arch", "cybersec", "analyst"):
        names = _tool_names(registry, mode)
        assert not {"write_file", "edit_file", "shell", "run_agent"} & names
        with pytest.raises(ToolError):
            registry.get("write_file", mode)


def test_an_unknown_mode_gets_no_tools() -> None:
    registry = ToolRegistry()
    assert registry.definitions("root") == []
    with pytest.raises(ToolError):
        registry.get("read_file", "root")
