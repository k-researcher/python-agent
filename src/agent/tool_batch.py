"""Batched parallel execution of read-only tool calls.

Run independent tool calls concurrently with a bounded concurrency.
Keep results in the original call order. Isolate each call failure.
This module is decoupled from the agent runtime.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent.tools.base import RiskLevel, Tool


@dataclass(frozen=True, slots=True)
class BatchCall:
    """A single tool invocation inside a batch."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BatchResult:
    """The outcome of a single tool invocation."""

    call_id: str
    ok: bool
    output: str
    error: str
    duration_ms: int


def is_parallel_safe(tool: Tool) -> bool:
    """Tell whether a tool may run in parallel with other tools.

    A tool is safe only when both conditions hold.
    The risk level must be read_only.
    The class must set parallel_safe to True.
    """
    marked_safe = getattr(type(tool), "parallel_safe", False)
    return tool.risk_level is RiskLevel.read_only and marked_safe


def split_parallel(
    calls: list[BatchCall],
    is_safe: Callable[[str], bool],
) -> list[list[BatchCall]]:
    """Split calls into groups for parallel or sequential execution.

    Consecutive safe calls merge into one group.
    Each unsafe call forms its own group.
    The original order is preserved.
    """
    groups: list[list[BatchCall]] = []
    current: list[BatchCall] = []
    for call in calls:
        if is_safe(call.name):
            current.append(call)
        else:
            if current:
                groups.append(current)
                current = []
            groups.append([call])
    if current:
        groups.append(current)
    return groups


async def run_batch(
    calls: list[BatchCall],
    execute: Callable[[BatchCall], Awaitable[str]],
    *,
    max_concurrency: int = 4,
    timeout_seconds: float = 60.0,
) -> list[BatchResult]:
    """Run tool calls and return results in the original order.

    Each call runs as its own asyncio task.
    A semaphore bounds the number of concurrent calls.
    A per-call timeout turns the call into a failed result.
    An exception in one call does not affect the other calls.
    An external cancellation stops all unfinished tasks and propagates.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if not calls:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(call: BatchCall) -> BatchResult:
        async with semaphore:
            started = time.monotonic()
            timeout = asyncio.timeout(timeout_seconds)
            try:
                async with timeout:
                    output = await execute(call)
            except TimeoutError as exc:
                # Only the batch deadline reports "timed out"; a TimeoutError that the tool
                # raises itself keeps its own message.
                message = "timed out" if timeout.expired() else str(exc) or "TimeoutError"
                return BatchResult(call.call_id, False, "", message, _elapsed_ms(started))
            except Exception as exc:
                return BatchResult(call.call_id, False, "", str(exc), _elapsed_ms(started))
            return BatchResult(call.call_id, True, output, "", _elapsed_ms(started))

    tasks = [asyncio.create_task(run_one(call)) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _elapsed_ms(started: float) -> int:
    """Return the elapsed time in whole milliseconds."""
    return int((time.monotonic() - started) * 1000)
