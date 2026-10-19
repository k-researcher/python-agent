from __future__ import annotations

import pytest

from agent.source_binding import (
    SourceDisabledError,
    mcp_tool_policy,
    source_scope,
)
from agent.source_policy import SourceScope
from agent.tool_sources import ToolSource
from agent.tools.base import RiskLevel
from agent.tools.mcp_tools import McpToolPolicy


def _source(**overrides: object) -> ToolSource:
    defaults: dict[str, object] = {
        "source_id": "server-1",
        "workspace_id": "Workspace-UUID",
        "command": ("server",),
        "enabled": False,
    }
    return ToolSource.model_validate(defaults | overrides)


def test_default_disabled_source_scope() -> None:
    scope = source_scope(_source())
    assert scope.source_id == "server-1"
    assert scope.workspace_id == "Workspace-UUID"
    assert scope.project_ids == frozenset()
    assert scope.enabled is False
    assert scope.allowed_modes == frozenset({"dev"})


def test_default_disabled_source_rejected_by_policy() -> None:
    with pytest.raises(SourceDisabledError):
        mcp_tool_policy(_source())
    with pytest.raises(ValueError):
        mcp_tool_policy(_source())


def test_enabled_ask_and_dev_mode_subsets() -> None:
    for modes in (frozenset({"dev"}), frozenset({"ask"}), frozenset({"dev", "ask"})):
        source = _source(enabled=True, allowed_modes=modes)
        scope = source_scope(source)
        policy = mcp_tool_policy(source)
        assert scope.allowed_modes == modes
        assert policy.allowed_modes == modes


def test_trusted_risk_level_is_not_downgraded_or_widened() -> None:
    for risk in (
        RiskLevel.read_only,
        RiskLevel.local_write,
        RiskLevel.process_execution,
        RiskLevel.network_access,
        RiskLevel.remote_write,
        RiskLevel.destructive,
        RiskLevel.secret_access,
    ):
        policy = mcp_tool_policy(_source(enabled=True, risk_level=risk))
        assert policy.risk_level is risk


def test_result_bound_is_preserved() -> None:
    source = _source(enabled=True, max_result_chars=1)
    assert mcp_tool_policy(source).max_result_chars == 1
    source = _source(enabled=True, max_result_chars=1_000_000)
    assert mcp_tool_policy(source).max_result_chars == 1_000_000


def test_exact_scope_and_policy_bounds() -> None:
    source = _source(
        source_id="server-2",
        workspace_id="workspace-B",
        project_ids=frozenset({"project-1", "project-2"}),
        enabled=True,
        allowed_modes=frozenset({"ask"}),
        risk_level=RiskLevel.local_write,
        max_result_chars=4096,
    )
    scope = source_scope(source)
    assert scope == SourceScope(
        source_id="server-2",
        workspace_id="workspace-B",
        project_ids=frozenset({"project-1", "project-2"}),
        enabled=True,
        allowed_modes=frozenset({"ask"}),
    )
    policy = mcp_tool_policy(source)
    assert policy == McpToolPolicy(
        server_id="server-2",
        risk_level=RiskLevel.local_write,
        allowed_modes=frozenset({"ask"}),
        allowed_tools=None,
        max_result_chars=4096,
    )


def test_immutable_sets_are_preserved_without_aliasing() -> None:
    projects = frozenset({"project-1"})
    modes = frozenset({"ask"})
    source = _source(
        enabled=True,
        project_ids=projects,
        allowed_modes=modes,
    )
    scope = source_scope(source)
    policy = mcp_tool_policy(source)
    assert scope.project_ids == projects
    assert scope.allowed_modes == modes
    assert policy.allowed_modes == modes
    for instance in (scope, policy):
        with pytest.raises((AttributeError, TypeError)):
            instance.source_id = "other"
        assert not hasattr(instance, "__dict__")


def test_source_scope_keeps_disabled_flag() -> None:
    scope = source_scope(_source())
    assert scope.enabled is False
    enabled = source_scope(_source(enabled=True))
    assert enabled.enabled is True
