import pytest

from agent.config import Settings
from agent.tools import ToolRegistry
from agent.tools.base import RiskLevel, ToolError


def test_read_only_tool_does_not_require_confirmation() -> None:
    registry = ToolRegistry()

    assert registry.requires_confirmation("read_file", "dev") is False


def test_run_registry_keeps_settings_and_does_not_modify_base() -> None:
    from agent.skill_prompt import SkillSet
    from agent.tools.skill_tools import ReadSkillTool

    settings = Settings(allow_network_tools=False)
    registry = ToolRegistry(settings=settings)
    scoped = registry.with_tools((ReadSkillTool(SkillSet(skills=(), errors=())),))

    assert scoped is not registry
    assert scoped.settings is settings
    assert scoped.requires_confirmation("read_skill", "ask") is False
    with pytest.raises(ToolError):
        registry.get("read_skill", "ask")
    with pytest.raises(ToolError):
        scoped.get("write_file", "ask")
    with pytest.raises(ToolError):
        scoped.get("http_request", "dev")


def test_run_registry_refuses_to_replace_an_existing_tool() -> None:
    registry = ToolRegistry()
    original = registry.get("read_file", "dev")

    with pytest.raises(ToolError, match="already registered"):
        registry.with_tools((original,))
    assert registry.get("read_file", "dev") is original


@pytest.mark.parametrize("name", ["write_file", "edit_file", "remove_file", "shell"])
def test_side_effect_tools_require_confirmation(name: str) -> None:
    registry = ToolRegistry()

    assert registry.requires_confirmation(name, "dev") is True


def test_ask_mode_cannot_access_mutating_tools() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolError):
        registry.get("write_file", "ask")


def test_ask_definitions_are_read_only() -> None:
    registry = ToolRegistry()
    names = {item["function"]["name"] for item in registry.definitions("ask")}

    assert {
        "list_dir",
        "read_file",
        "search_files",
        "glob",
        "read_many",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "todo_write",
        "todo_read",
        "excel_read",
        "excel_sheets",
    } == names
    assert all(registry.risk(name, "ask") is RiskLevel.read_only for name in names)


def test_run_checks_is_a_dev_tool_that_needs_approval() -> None:
    registry = ToolRegistry()
    assert "run_checks" in {item["function"]["name"] for item in registry.definitions("dev")}
    assert registry.risk("run_checks", "dev") is RiskLevel.process_execution
    assert registry.requires_confirmation("run_checks", "dev") is True
    with pytest.raises(ToolError):
        registry.get("run_checks", "ask")


def test_plan_tools_need_no_approval_in_both_modes() -> None:
    registry = ToolRegistry()
    for mode in ("ask", "dev"):
        for name in ("todo_write", "todo_read"):
            assert registry.requires_confirmation(name, mode) is False


def test_disabled_capabilities_are_not_exposed_to_model() -> None:
    settings = Settings(allow_network_tools=False, knowledge_base_enabled=False)
    registry = ToolRegistry(settings=settings)
    names = {item["function"]["name"] for item in registry.definitions("dev")}

    assert "http_request" not in names
    assert "kb_search" not in names
    assert "run_agent" in names


def test_enabled_local_knowledge_is_read_only_but_writes_require_approval() -> None:
    registry = ToolRegistry(settings=Settings(knowledge_base_enabled=True))

    assert registry.requires_confirmation("kb_search", "ask") is False
    assert registry.requires_confirmation("kb_save", "dev") is True


def test_child_agent_schema_lists_configured_model_profiles() -> None:
    settings = Settings(
        default_llm_profile="fast",
        llm_profiles={
            "fast": {"base_url": "https://a.example/v1", "model": "a"},
            "deep": {"base_url": "https://b.example/v1", "model": "b"},
        },
    )
    registry = ToolRegistry(settings=settings)
    definition = next(
        item for item in registry.definitions("dev") if item["function"]["name"] == "run_agent"
    )

    profile_schema = definition["function"]["parameters"]["properties"]["llm_profile"]
    assert profile_schema["enum"] == ["deep", "fast"]


def test_shell_operators_are_refused_before_approval() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolError, match="Shell operators are not supported: &&"):
        registry.validate("shell", "dev", {"command": "pytest -q && ruff check"})
    registry.validate("shell", "dev", {"command": "pytest -q tests"})
