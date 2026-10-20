"""Audit session MCP invocations without retaining payloads or remote errors."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from agent.session_fence import FenceLostError
from agent.tools.base import ToolContext, ToolError
from agent.tools.mcp_tools import McpBackedTool

logger = logging.getLogger(__name__)


async def _persist(record: Coroutine[Any, Any, None]) -> bool:
    recording = asyncio.create_task(record)
    cancelled = False
    while not recording.done():
        try:
            await asyncio.shield(recording)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    try:
        recording.result()
    except BaseException:
        if cancelled:
            raise asyncio.CancelledError from None
        raise
    return cancelled


async def execute_mcp_attempt(
    tool: McpBackedTool,
    context: ToolContext,
    arguments: dict[str, Any],
    *,
    tool_call_id: str,
    before_send: Callable[[], Awaitable[None]],
) -> dict[str, Any]:
    """Record linked audit events around one guarded call, never retrying it."""
    callback = context.audit_egress
    if callback is None:
        raise ToolError("MCP invocation requires outbound audit")
    revision = tool.approval_revision()
    source = tool.source
    assert source is not None
    try:
        snapshot = copy.deepcopy(arguments)
        payload = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, RecursionError):
        raise ToolError("Invalid MCP argument snapshot") from None
    metadata = {
        "attempt_id": uuid.uuid4().hex,
        "tool_call_sha256": hashlib.sha256(
            tool_call_id.encode("utf-8", errors="replace")
        ).hexdigest(),
        "source_revision": revision,
    }

    async def record(status: str, outcome: str) -> None:
        detail = json.dumps({**metadata, "outcome": outcome}, sort_keys=True, separators=(",", ":"))
        await callback("mcp", source.source_id, tool.name, payload, status, detail)

    async def record_failure(status: str, outcome: str) -> bool:
        try:
            return await _persist(record(status, outcome))
        except Exception:
            logger.warning("MCP terminal audit could not be recorded")
            return False

    try:
        cancelled = await _persist(record("started", "started"))
    except FenceLostError:
        raise
    except Exception:
        raise ToolError("MCP outbound audit could not be recorded") from None
    if cancelled:
        await record_failure("cancelled", "cancelled")
        raise asyncio.CancelledError

    phase = "guard_denied"
    try:
        await before_send()
        phase = "invocation_error"
        result = await tool.execute(context, snapshot)
    except asyncio.CancelledError:
        await record_failure("cancelled", "cancelled")
        raise
    except FenceLostError:
        await record_failure("error", "fence_lost")
        raise
    except Exception:
        cancelled = await record_failure("error", phase)
        if cancelled:
            raise asyncio.CancelledError from None
        raise ToolError("MCP invocation failed") from None

    status = "error" if result.get("is_error") else "completed"
    outcome = "tool_error" if result.get("is_error") else "completed"
    try:
        cancelled = await _persist(record(status, outcome))
    except FenceLostError:
        raise
    except Exception:
        raise ToolError("MCP outbound audit could not be recorded") from None
    if cancelled:
        raise asyncio.CancelledError
    return result
