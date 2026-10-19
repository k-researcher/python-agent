"""Select tool sources that the current session is allowed to use."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from agent.source_access import can_use_session_source
from agent.source_binding import source_scope
from agent.tool_sources import ToolSource, ToolSourceError


async def select_session_sources(
    db: AsyncSession,
    session_id: str,
    sources: tuple[ToolSource, ...],
) -> tuple[ToolSource, ...]:
    """Return, in original order, only the sources the session may use.

    Duplicate (workspace_id, source_id) identities are rejected before any SQL,
    even when the duplicates are disabled. The authorization check reads fresh
    session facts and never writes or flushes.
    """
    identities = {(source.workspace_id, source.source_id) for source in sources}
    if len(identities) != len(sources):
        raise ToolSourceError("Duplicate tool source")
    allowed: list[ToolSource] = []
    for source in sources:
        if await can_use_session_source(db, session_id, source_scope(source)):
            allowed.append(source)
    return tuple(allowed)
