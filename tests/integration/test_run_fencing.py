"""Run fencing in the runtime, cut-off replies and the provider history of broken tool calls."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from agent.api import supervisor
from agent.database import session_factory
from agent.llm import LLMResponse
from agent.session_fence import RunFence, claim
from tests.integration.helpers import authed_client
from tests.integration.test_runtime_robustness import project_session, wait_for


class StaleRunLLM:
    """While the run waits for the model, its lease expires and another executor reclaims it.

    No Stop is requested and the status stays active, so only the fence can stop the stale run.
    """

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.session_id = ""
        self.new_owner: RunFence | None = None

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        async with session_factory() as db:
            # These writes are not part of the stale run: they must pass without its fence.
            db.sync_session.info["fence_exempt"] = True
            sessions = sa.table(
                "sessions", sa.column("id"), sa.column("lease_until", sa.DateTime(timezone=True))
            )
            await db.execute(
                sa.update(sessions)
                .where(sessions.c.id == self.session_id)
                .values(lease_until=datetime.now(UTC) - timedelta(hours=1))
            )
            await db.commit()
            self.new_owner = await claim(db, self.session_id, lease_seconds=600)
        return LLMResponse("late answer from a stale run", [], "stop", 1, 1)


def test_reclaimed_run_cannot_write(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = StaleRunLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "fence", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={
                    "project_id": project["id"],
                    "prompt": "fence",
                    "mode": "dev",
                    "auto_start": False,
                },
            ).json()
            fake.session_id = session["id"]
            assert client.post(f"/api/sessions/{session['id']}/start").status_code == 202
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                task = supervisor._tasks.get(session["id"])
                if fake.new_owner is not None and task is not None and task.done():
                    break
                time.sleep(0.02)
            messages = client.get(f"/api/sessions/{session['id']}/messages").json()
            status = client.get(f"/api/sessions/{session['id']}").json()["status"]
        assert fake.new_owner is not None and fake.new_owner.epoch == 2
        assert all(message["content"] != "late answer from a stale run" for message in messages)
        assert status not in {"completed", "error"}
    finally:
        supervisor.llm = original


class EmptyCutOffLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        return LLMResponse(None, [], "length", 1, 16_000)


def test_empty_cut_off_reply_is_an_error_not_completion(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EmptyCutOffLLM()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "cutoff")
            wait_for(client, session_id, "error")
            session = client.get(f"/api/sessions/{session_id}").json()
        assert "cut off by the output limit" in session["error"]
    finally:
        supervisor.llm = original


class BrokenArgumentsLLM:
    """First reply: a tool call with cut-off JSON arguments. Second reply: a final answer."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.histories: list[list[dict[str, Any]]] = []

    async def chat(
        self, messages: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any
    ) -> Any:
        self.histories.append(messages)
        if len(self.histories) == 1:
            broken = {
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.txt", "start'},
            }
            return LLMResponse(None, [broken], "tool_calls", 1, 1)
        return LLMResponse("done", [], "stop", 1, 1)


def test_broken_tool_arguments_never_reach_the_provider_history(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = BrokenArgumentsLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "broken-args")
            wait_for(client, session_id, "completed")
            messages = client.get(f"/api/sessions/{session_id}/messages").json()
        assert len(fake.histories) == 2
        for item in fake.histories[1]:
            for call in item.get("tool_calls") or []:
                json.loads(call["function"]["arguments"])  # valid JSON for the provider
        tool_results = [m for m in messages if m["role"] == "tool"]
        assert "not valid JSON" in tool_results[0]["content"]
    finally:
        supervisor.llm = original


class CutOffToolCallLLM:
    """A reply cut off by the output limit in the middle of a write_file call."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        self.calls += 1
        call = {
            "type": "function",
            "function": {"name": "write_file", "arguments": '{"path": "big.txt", "content": "x"}'},
        }
        return LLMResponse(None, [call], "length", 1, 16_000)


def test_cut_off_tool_call_runs_no_tool(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = CutOffToolCallLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "cutoff-tool")
            wait_for(client, session_id, "error")
            approvals = client.get(f"/api/sessions/{session_id}/approvals").json()
            messages = client.get(f"/api/sessions/{session_id}/messages").json()
        assert approvals == []
        assert not (tmp_path / "big.txt").exists()
        assert all(message["role"] != "tool" for message in messages)
        assert fake.calls == 1
    finally:
        supervisor.llm = original
