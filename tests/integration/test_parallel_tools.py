"""Parallel read-only tool calls in the runtime: real overlap, stored in call order."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent.api import supervisor
from agent.llm import LLMResponse
from agent.tools.base import ToolContext
from agent.tools.builtin import GlobTool, ReadFileTool
from tests.integration.helpers import authed_client


def _call(index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"provider-{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeToolsLLM:
    """First reply asks for the given tool calls, the second reply ends the task."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.tool_calls = calls
        self.replies = 0

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]], **_options: Any
    ) -> LLMResponse:
        self.replies += 1
        if self.replies == 1:
            return LLMResponse(
                content=None,
                tool_calls=self.tool_calls,
                finish_reason="tool_calls",
                prompt_tokens=10,
                completion_tokens=5,
            )
        return LLMResponse(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
        )


def _wait_for_status(client: TestClient, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Session did not reach {expected}")


def _run(root: Path, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run one session with the fake model; return its tool messages and the call ids."""
    original_llm = supervisor.llm
    supervisor.llm = FakeToolsLLM(calls)  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Parallel", "root_path": str(root)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={
                    "project_id": project["id"],
                    "prompt": "Read the files",
                    "mode": "dev",
                    "auto_start": True,
                },
            ).json()
            _wait_for_status(client, session["id"], "completed")
            return list(client.get(f"/api/sessions/{session['id']}/messages").json())
    finally:
        supervisor.llm = original_llm


def _tool_results(messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    assistant = next(item for item in messages if item["role"] == "assistant")
    call_ids = [call["id"] for call in assistant["tool_calls"]]
    tools = [item for item in messages if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in tools] == call_ids
    return call_ids, [json.loads(item["content"]) for item in tools]


def test_read_only_calls_run_at_the_same_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    original = ReadFileTool.execute
    state: dict[str, Any] = {"started": 0, "both": None}

    async def execute(
        self: ReadFileTool, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        # Each call waits until the other one has started: in sequence this times out.
        if state["both"] is None:
            state["both"] = asyncio.Event()
        state["started"] += 1
        if state["started"] == 2:
            state["both"].set()
        await asyncio.wait_for(state["both"].wait(), 3)
        return await original(self, context, arguments)

    monkeypatch.setattr(ReadFileTool, "execute", execute)
    messages = _run(
        tmp_path,
        [_call(1, "read_file", {"path": "a.txt"}), _call(2, "read_file", {"path": "b.txt"})],
    )
    _, results = _tool_results(messages)
    assert [result["success"] for result in results] == [True, True]
    assert "alpha" in json.dumps(results[0], ensure_ascii=False)
    assert "beta" in json.dumps(results[1], ensure_ascii=False)


def test_unsafe_call_splits_the_group_and_order_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    events: list[str] = []
    read_original = ReadFileTool.execute
    glob_original = GlobTool.execute

    async def slow_read(
        self: ReadFileTool, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        events.append(f"start {arguments['path']}")
        # The first read is slow, so a parallel neighbour finishes before it.
        await asyncio.sleep(0.2 if arguments["path"] == "a.txt" else 0)
        events.append(f"end {arguments['path']}")
        return await read_original(self, context, arguments)

    async def fast_glob(
        self: GlobTool, context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        events.append("start glob")
        events.append("end glob")
        return await glob_original(self, context, arguments)

    monkeypatch.setattr(ReadFileTool, "execute", slow_read)
    monkeypatch.setattr(GlobTool, "execute", fast_glob)
    messages = _run(
        tmp_path,
        [
            _call(1, "read_file", {"path": "a.txt"}),
            _call(2, "glob", {"pattern": "*.txt"}),
            # git_status is read-only but not parallel safe: it runs alone.
            _call(3, "git_status", {}),
            _call(4, "read_file", {"path": "b.txt"}),
        ],
    )
    _, results = _tool_results(messages)
    assert len(results) == 4
    assert results[0]["success"] and results[1]["success"] and results[3]["success"]
    # The glob call overlapped the slow read; the last read started after the group ended.
    assert events.index("end glob") < events.index("end a.txt")
    assert events.index("start b.txt") > events.index("end a.txt")
