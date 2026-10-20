"""Map trusted tool-source configuration to scopes and MCP policies.

These adapters are pure: they do not read the network, the database, the
environment, or any registry or session state. They only reshape the trusted
configuration carried by a ToolSource. The risk level and the result-size bound
are trusted from the operator configuration and never widened or dropped here.
"""

from __future__ import annotations

from agent.source_policy import SourceScope
from agent.tool_sources import ToolSource
from agent.tools.mcp_tools import McpToolPolicy


class SourceDisabledError(ValueError):
    """Report that an MCP policy cannot be built for a disabled source."""


def source_scope(source: ToolSource) -> SourceScope:
    """Build the access scope that mirrors the trusted source configuration.

    Every scope-relevant field is preserved unchanged: the source id, the
    workspace id, the project allowlist, the enabled flag, and the allowed
    modes. The trusted risk level and result bound belong to the MCP policy and
    are handed through by ``mcp_tool_policy`` rather than being changed here.
    """
    return SourceScope(
        source_id=source.source_id,
        workspace_id=source.workspace_id,
        project_ids=source.project_ids,
        enabled=source.enabled,
        allowed_modes=source.allowed_modes,
    )


def mcp_tool_policy(source: ToolSource) -> McpToolPolicy:
    """Build the MCP exposure policy for an explicitly enabled source.

    A disabled source is rejected. The policy keeps the source's server id,
    trusts its risk level without downgrading it, copies the allowed modes
    without widening them, and preserves the result-size bound.
    """
    if not source.enabled:
        raise SourceDisabledError(f"Source is disabled: {source.source_id}")
    return McpToolPolicy(
        server_id=source.source_id,
        risk_level=source.risk_level,
        allowed_modes=source.allowed_modes,
        max_result_chars=source.max_result_chars,
    )
