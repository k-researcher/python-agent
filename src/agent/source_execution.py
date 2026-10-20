"""Guard source-backed tool execution against missing, stale, or replaced sources.

These checks are read-only: they never write, flush, commit, touch the
environment, or launch processes. They only compare the approved revision bound
to an approval against the current tool snapshot and then require a fresh
session grant for the source.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from agent.source_access import can_use_session_source
from agent.source_binding import source_scope
from agent.tools.base import Tool, ToolError
from agent.tools.mcp_tools import McpBackedTool


def capture_source_revision(tool: Tool) -> str | None:
    """Capture the approval revision bound to a tool, or None for a builtin.

    A source-backed MCP tool returns its exact bound approval revision and raises
    ToolError when it is unbound, mismatched against its trusted source, or
    disabled. Builtin tools have no backing source and return None.
    """
    if isinstance(tool, McpBackedTool):
        return tool.approval_revision()
    if tool.name.startswith("mcp__"):
        raise ToolError("MCP tool namespace requires a trusted source")
    return None


async def validate_source_execution(
    db: AsyncSession,
    session_id: str,
    tool: Tool,
    approved_revision: str | None,
) -> None:
    """Deny execution when the source-backed approval is missing or stale.

    Builtin tools reject any nonnull stored revision, which prevents a
    source-backed approval from being replaced by a builtin tool, and otherwise
    perform no SQL. The reserved MCP namespace cannot be used by a builtin,
    including legacy calls with a null revision. MCP tools deny a null revision
    and a revision or source mismatch before SQL, then require a fresh session grant for the trusted
    source; a denied grant raises a generic ToolError.
    """
    if not isinstance(tool, McpBackedTool):
        capture_source_revision(tool)
        if approved_revision is not None:
            raise ToolError("Source-backed approval cannot run as a builtin tool")
        return
    if approved_revision is None:
        raise ToolError("Missing approval revision for MCP tool")
    current = capture_source_revision(tool)
    if current != approved_revision:
        raise ToolError("Approval revision does not match the current source")
    source = tool.source
    if source is None:
        raise ToolError("MCP tool has no trusted source")
    if not await can_use_session_source(db, session_id, source_scope(source)):
        raise ToolError("Source execution is not permitted for this session")
