"""Own trusted MCP clients and their registry for one session execution."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import TracebackType

from sqlalchemy import select

from agent.config import Settings
from agent.database import session_factory
from agent.mcp_client import StdioMcpClient
from agent.models import Project, Session
from agent.sandbox import policy_from_settings
from agent.session_fence import CURRENT_FENCE, TERMINAL_STATUSES, FenceLostError, fence_check
from agent.source_access import can_use_session_source
from agent.source_binding import mcp_tool_policy, source_scope
from agent.tool_sources import ToolSource, load_tool_sources
from agent.tools.base import ToolError
from agent.tools.mcp_tools import build_mcp_tools
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def _may_connect(session_id: str, project_root: Path, source: ToolSource) -> bool:
    async with session_factory() as db:
        row = (
            await db.execute(
                select(Session.status, Session.stop_requested, Project.root_path)
                .join(Project, Project.id == Session.project_id)
                .where(Session.id == session_id)
            )
        ).first()
        if (
            row is None
            or row.status in TERMINAL_STATUSES
            or row.stop_requested
            or Path(row.root_path).resolve(strict=True) != project_root
        ):
            return False
        allowed = await can_use_session_source(db, session_id, source_scope(source))
        fence = CURRENT_FENCE.get()
        if allowed and fence is not None:
            await fence_check(db, fence)
            await db.commit()
        return allowed


async def _close_client(client: StdioMcpClient) -> None:
    closing = asyncio.create_task(client.close())
    cancelled = False
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    try:
        closing.result()
    except Exception:
        if cancelled:
            raise asyncio.CancelledError from None
        raise ToolError("Trusted MCP client could not be closed") from None
    if cancelled:
        raise asyncio.CancelledError


def _close_on_exit(
    client: StdioMcpClient,
) -> Callable[
    [type[BaseException] | None, BaseException | None, TracebackType | None],
    Awaitable[bool],
]:
    """Close the client on exit; a close failure never hides an exception in flight."""

    async def close(
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            await _close_client(client)
        except ToolError:
            if exc is None:
                raise
            logger.warning("Trusted MCP client could not be closed")
        return False

    return close


@asynccontextmanager
async def source_registry(
    settings: Settings,
    session_id: str,
    project_root: Path,
    registry: ToolRegistry,
) -> AsyncIterator[ToolRegistry]:
    """Discover only operator-selected sources; close all clients on every exit."""
    sources = load_tool_sources(settings.tool_sources_file)
    async with AsyncExitStack() as stack:
        try:
            for source in sources:
                if not await _may_connect(session_id, project_root, source):
                    continue
                environment = {
                    key: os.environ[key]
                    for key in ("PATH", "LANG", *source.env_keys)
                    if key in os.environ
                }
                client = StdioMcpClient(
                    list(source.command),
                    cwd=project_root,
                    env=environment,
                    sandbox=replace(policy_from_settings(settings), mode="required"),
                    session_id=session_id,
                )
                stack.push_async_exit(_close_on_exit(client))
                await client.start()
                tools = await build_mcp_tools(client, mcp_tool_policy(source), source=source)
                for tool in tools:
                    tool.approval_revision()
                if not await _may_connect(session_id, project_root, source):
                    raise ToolError("Source access changed during discovery")
                registry = registry.with_tools(tuple(tools))
        except FenceLostError:
            raise
        except Exception:
            raise ToolError("Trusted MCP sources could not be initialized") from None
        yield registry
