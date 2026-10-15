"""Unit tests for batched parallel tool execution."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.tool_batch import (
    BatchCall,
    BatchResult,
    is_parallel_safe,
    run_batch,
    split_parallel,
)
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError


class ReadSafeTool(Tool):
    name = "read_safe"
    description = "Read only tool marked safe."
    risk_level = RiskLevel.read_only
    parallel_safe = True
    input_schema: dict[str, Any] = {}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}


class ReadUnsafeTool(Tool):
    name = "read_unsafe"
    description = "Read only tool without the safe flag."
    risk_level = RiskLevel.read_only
    input_schema: dict[str, Any] = {}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}


class WriteSafeTool(Tool):
    name = "write_safe"
    description = "Write tool with the safe flag."
    risk_level = RiskLevel.local_write
    parallel_safe = True
    input_schema: dict[str, Any] = {}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}


def _call(call_id: str, name: str = "read") -> BatchCall:
    return BatchCall(call_id=call_id, name=name, arguments={})


def test_batch_call_is_frozen() -> None:
    call = _call("c1")

    with pytest.raises(AttributeError):
        call.call_id = "other"  # type: ignore[misc]


def test_batch_result_fields() -> None:
    result = BatchResult(call_id="c1", ok=True, output="out", error="", duration_ms=5)

    assert result.call_id == "c1"
    assert result.ok is True
    assert result.output == "out"
    assert result.error == ""
    assert result.duration_ms == 5


def test_is_parallel_safe_read_only_with_flag() -> None:
    assert is_parallel_safe(ReadSafeTool()) is True


def test_is_parallel_safe_read_only_without_flag() -> None:
    assert is_parallel_safe(ReadUnsafeTool()) is False


def test_is_parallel_safe_write_with_flag() -> None:
    assert is_parallel_safe(WriteSafeTool()) is False


async def test_run_batch_empty() -> None:
    async def execute(call: BatchCall) -> str:
        return "unused"

    assert await run_batch([], execute) == []


async def test_run_batch_invalid_concurrency() -> None:
    async def execute(call: BatchCall) -> str:
        return "unused"

    calls = [_call("c1")]
    with pytest.raises(ValueError, match="max_concurrency"):
        await run_batch(calls, execute, max_concurrency=0)


async def test_run_batch_preserves_order_with_different_durations() -> None:
    delays = {"slow": 0.1, "fast": 0.01, "medium": 0.05}

    async def execute(call: BatchCall) -> str:
        await asyncio.sleep(delays[call.name])
        return f"out-{call.call_id}"

    calls = [
        _call("slow1", "slow"),
        _call("fast1", "fast"),
        _call("medium1", "medium"),
        _call("slow2", "slow"),
        _call("fast2", "fast"),
    ]
    results = await run_batch(calls, execute)

    assert [r.call_id for r in results] == ["slow1", "fast1", "medium1", "slow2", "fast2"]
    assert all(r.ok for r in results)
    assert [r.output for r in results] == [f"out-{c.call_id}" for c in calls]


async def test_run_batch_enforces_concurrency_limit() -> None:
    active = 0
    max_active = 0

    async def execute(call: BatchCall) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return f"out-{call.call_id}"

    calls = [_call(f"c{i}") for i in range(8)]
    results = await run_batch(calls, execute, max_concurrency=2)

    assert max_active == 2
    assert len(results) == 8
    assert all(r.ok for r in results)


async def test_run_batch_isolates_tool_errors() -> None:
    async def execute(call: BatchCall) -> str:
        if call.call_id == "bad_tool":
            raise ToolError("tool boom")
        if call.call_id == "bad_value":
            raise ValueError("value boom")
        if call.call_id == "bad_generic":
            raise RuntimeError("generic boom")
        await asyncio.sleep(0.01)
        return f"out-{call.call_id}"

    calls = [
        _call("a"),
        _call("bad_tool"),
        _call("b"),
        _call("bad_value"),
        _call("c"),
        _call("bad_generic"),
        _call("d"),
    ]
    results = await run_batch(calls, execute)

    assert [r.call_id for r in results] == [c.call_id for c in calls]
    assert [r.ok for r in results] == [True, False, True, False, True, False, True]
    assert [r.error for r in results] == ["", "tool boom", "", "value boom", "", "generic boom", ""]
    assert [r.output for r in results] == ["out-a", "", "out-b", "", "out-c", "", "out-d"]


async def test_run_batch_timeout() -> None:
    async def execute(call: BatchCall) -> str:
        await asyncio.sleep(10)
        return "late"

    calls = [_call("c1")]
    results = await run_batch(calls, execute, timeout_seconds=0.05)

    assert len(results) == 1
    assert not results[0].ok
    assert results[0].error == "timed out"
    assert results[0].output == ""


async def test_run_batch_timeout_does_not_block_others() -> None:
    async def execute(call: BatchCall) -> str:
        if call.call_id == "slow":
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(0.01)
        return f"out-{call.call_id}"

    calls = [_call("a"), _call("slow"), _call("b")]
    results = await run_batch(calls, execute, timeout_seconds=0.05)

    assert [r.call_id for r in results] == ["a", "slow", "b"]
    assert [r.ok for r in results] == [True, False, True]
    assert results[1].error == "timed out"


async def test_run_batch_external_cancellation_cancels_all_tasks() -> None:
    release = asyncio.Event()
    completed: list[str] = []
    started = asyncio.Event()

    async def execute(call: BatchCall) -> str:
        if call.call_id == "c0":
            started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            raise
        completed.append(call.call_id)
        return f"out-{call.call_id}"

    calls = [_call(f"c{i}") for i in range(4)]
    task = asyncio.create_task(run_batch(calls, execute, max_concurrency=2))

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # All execute calls must have been cancelled; none may complete.
    release.set()
    await asyncio.sleep(0.01)
    assert completed == []

    # No leaked pending tasks must remain in the event loop.
    current = asyncio.current_task()
    assert current is not None
    pending = [t for t in asyncio.all_tasks() if not t.done() and t is not current]
    assert pending == []


def test_split_parallel_empty() -> None:
    assert split_parallel([], lambda name: name in {"read"}) == []


def test_split_parallel_all_safe() -> None:
    calls = [_call("a", "read"), _call("b", "read"), _call("c", "read")]
    groups = split_parallel(calls, lambda name: name in {"read"})

    assert groups == [calls]


def test_split_parallel_all_unsafe() -> None:
    calls = [_call("a", "write"), _call("b", "write"), _call("c", "write")]
    groups = split_parallel(calls, lambda name: name in {"read"})

    assert groups == [[calls[0]], [calls[1]], [calls[2]]]


def test_split_parallel_mixed_sequence() -> None:
    safe = {"read", "glob"}

    def is_safe(name: str) -> bool:
        return name in safe

    calls = [
        _call("a", "read"),
        _call("b", "read"),
        _call("c", "write"),
        _call("d", "read"),
        _call("e", "shell"),
        _call("f", "glob"),
        _call("g", "read"),
    ]
    groups = split_parallel(calls, is_safe)

    assert groups == [
        [calls[0], calls[1]],
        [calls[2]],
        [calls[3]],
        [calls[4]],
        [calls[5], calls[6]],
    ]


async def test_tool_timeout_error_keeps_its_own_message() -> None:
    async def execute(_call: BatchCall) -> str:
        raise TimeoutError("upstream service timed out after 3 s")

    results = await run_batch([BatchCall("c1", "read_file", {})], execute, timeout_seconds=5)
    assert results[0].ok is False
    assert results[0].error == "upstream service timed out after 3 s"


async def test_run_batch_without_timeout_waits_for_the_call() -> None:
    async def execute(call: BatchCall) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"call": call.call_id}

    results = await run_batch([BatchCall("c1", "read_file", {})], execute, timeout_seconds=None)
    assert results[0].ok is True
    assert results[0].output == {"call": "c1"}


def test_only_plain_file_readers_are_parallel_safe() -> None:
    from agent.tools import ToolRegistry

    registry = ToolRegistry()
    safe = sorted(name for name, tool in registry._tools.items() if is_parallel_safe(tool))
    # Git tools may take the index lock and todo tools write the session row: never parallel.
    assert safe == ["glob", "list_dir", "read_file", "read_many", "search_files"]
