"""Hard session budget limits enforced in the runtime loop.

These tests drive the real AgentSupervisor run through scripted LLM replies (no network)
and assert that exhausted limits stop the session with the durable ``stop_reason``,
without ``session.error`` and without a late HTTP attempt. A new message after a budget
stop starts a fresh run with fresh steps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from agent.api import settings, supervisor
from agent.database import init_database, session_factory
from agent.llm import LLMResponse
from agent.model_config import PricingSpec
from agent.models import Message, OutboundAudit, Project, Session, SessionStatus
from agent.session_budget_store import mark_budget_stop
from tests.integration.helpers import authed_client


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"provider-{call_id}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _read_call(path: str) -> dict[str, Any]:
    return _call("read", "read_file", {"path": path})


def _write_call(path: str) -> dict[str, Any]:
    return _call("write", "write_file", {"path": path, "content": "x"})


class ScriptedLLM:
    """Return the scripted replies in order and record the number of HTTP requests."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, replies: list[LLMResponse]) -> None:
        self.replies = replies
        self.calls = 0

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]], **_options: Any
    ) -> LLMResponse:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def wait_for(client: Any, session_id: str, expected: str, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/sessions/{session_id}").json()
        if body["status"] == expected:
            return
        time.sleep(0.02)
    body = client.get(f"/api/sessions/{session_id}").json()
    raise AssertionError(f"Session did not reach {expected}: {body}")


def _create_run(
    client: Any,
    root: Path,
    prompt: str,
    fake: ScriptedLLM,
    *,
    budget: dict[str, object] | None = None,
    auto_start: bool = True,
) -> str:
    project = client.post("/api/projects", json={"name": prompt, "root_path": str(root)}).json()
    payload: dict[str, object] = {
        "project_id": project["id"],
        "prompt": prompt,
        "mode": "dev",
        "auto_start": auto_start,
    }
    if budget is not None:
        payload["budget"] = budget
    return str(client.post("/api/sessions", json=payload).json()["id"])


@contextlib.contextmanager
def _fake_llm(fake: ScriptedLLM) -> Any:
    previous = supervisor.llm
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        yield fake
    finally:
        supervisor.llm = previous


async def _audit_rows(session_id: str) -> list[OutboundAudit]:
    async with session_factory() as db:
        from sqlalchemy import select

        return list(
            await db.scalars(
                select(OutboundAudit).where(
                    OutboundAudit.session_id == session_id,
                    OutboundAudit.category == "llm",
                )
            )
        )


async def _manual_session(root: Path, name: str) -> str:
    """Create a pending project/session and its first user message directly in the DB."""
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name=name, root_path=str(root))
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            title=name,
            status=SessionStatus.pending.value,
            configuration={},
        )
        db.add(session)
        await db.flush()
        db.add(Message(session_id=session.id, role="user", content="Start"))
        await db.commit()
        return session.id


async def _session_row(session_id: str) -> Session:
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        return row


def test_max_steps_runs_two_inferences_and_stops_the_third(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    replies = [
        LLMResponse(None, [_read_call("a.txt")], "tool_calls", 10, 5),
        LLMResponse(None, [_read_call("b.txt")], "tool_calls", 10, 5),
        LLMResponse("never", [], "stop", 10, 5),
    ]
    with authed_client() as client, _fake_llm(ScriptedLLM(replies)) as fake:
        session_id = _create_run(client, tmp_path, "read twice", fake, budget={"max_steps": 2})
        wait_for(client, session_id, "stopped")

        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["status"] == "stopped"
        assert body["error"] is None
        assert body["budget"]["stop_reason"] == "steps_exhausted"
        assert body["budget"]["steps"] == 2
        assert fake.calls == 2  # the third inference was never attempted

        messages = client.get(f"/api/sessions/{session_id}/messages").json()
        tools = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
        flattened = json.dumps(tools, ensure_ascii=False)
        # Tools of the second step ran: both reads succeeded and returned real content.
        assert "alpha" in flattened and "beta" in flattened
        assert len(tools) == 2
        assert all(item["success"] for item in tools)


def test_token_limit_blocks_attempt_without_started_audit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    original = settings.llm_max_tokens
    settings.llm_max_tokens = 300  # a small worst-case output bound for the attempt
    try:
        replies = [
            LLMResponse(None, [_read_call("a.txt")], "tool_calls", 25000, 100),
            LLMResponse("done", [], "stop", 25000, 100),
        ]
        with authed_client() as client, _fake_llm(ScriptedLLM(replies)) as fake:
            session_id = _create_run(client, tmp_path, "tokens", fake, budget={"max_tokens": 30000})
            wait_for(client, session_id, "stopped")

            body = client.get(f"/api/sessions/{session_id}").json()
            assert body["status"] == "stopped"
            assert body["budget"]["stop_reason"] == "tokens_exhausted"
            assert fake.calls == 1  # the next attempt was not sent

            rows = asyncio.run(_audit_rows(session_id))
            assert len(rows) == 1  # only the first attempt was recorded
            assert rows[0].status == "completed"
    finally:
        settings.llm_max_tokens = original


def test_money_limit_without_pricing_is_cost_unknown_and_sends_nothing(
    tmp_path: Path,
) -> None:
    # The legacy model has no pricing: a money ceiling makes any attempt boundary unknown,
    # so the very first HTTP attempt is refused.
    with (
        authed_client() as client,
        _fake_llm(ScriptedLLM([LLMResponse("hello", [], "stop", 5, 5)])) as fake,
    ):
        session_id = _create_run(client, tmp_path, "costs", fake, budget={"max_cost": "0.10"})
        wait_for(client, session_id, "stopped")

        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["status"] == "stopped"
        assert body["budget"]["stop_reason"] == "cost_unknown"
        assert fake.calls == 0
        assert asyncio.run(_audit_rows(session_id)) == []


def test_money_exhausted_after_first_answer_stops_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model is priced and the first answer overruns the money ceiling: the second
    # inference boundary refuses before any retry, fallback or HTTP attempt.
    import agent.model_config as model_config

    real = model_config._resolve

    def with_pricing(*args: Any, **kwargs: Any) -> Any:
        registry = real(*args, **kwargs)
        for model in registry.models.values():
            object.__setattr__(model, "pricing", PricingSpec(input_per_mtok=1, output_per_mtok=1))
        return registry

    monkeypatch.setattr(model_config, "_resolve", with_pricing)
    original = settings.llm_max_tokens
    settings.llm_max_tokens = 200
    try:
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        replies = [
            LLMResponse(None, [_read_call("a.txt")], "tool_calls", 40000, 10000),
            LLMResponse("done", [], "stop", 1, 1),
        ]
        with authed_client() as client, _fake_llm(ScriptedLLM(replies)) as fake:
            session_id = _create_run(client, tmp_path, "costs", fake, budget={"max_cost": "0.05"})
            wait_for(client, session_id, "stopped")

            body = client.get(f"/api/sessions/{session_id}").json()
            assert body["status"] == "stopped"
            assert body["budget"]["stop_reason"] == "cost_exhausted"
            assert body["error"] is None
            assert fake.calls == 1  # no retry, no fallback request
    finally:
        settings.llm_max_tokens = original


def test_deadline_in_the_past_stops_before_the_first_request(tmp_path: Path) -> None:
    with _fake_llm(ScriptedLLM([LLMResponse("hello", [], "stop", 1, 1)])) as fake:
        session_id = asyncio.run(_manual_session(tmp_path, "deadline"))

        async def seed() -> None:
            stale = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat()
            async with session_factory() as db:
                row = await db.get(Session, session_id)
                assert row is not None
                row.budget_state = {
                    "limits": {"max_steps": 10, "deadline_seconds": 5.0},
                    "run_generation": row.run_generation or 0,
                    "started_at": stale,
                }
                await db.commit()

        asyncio.run(seed())
        asyncio.run(supervisor._run(session_id))

        row = asyncio.run(_session_row(session_id))
        assert row.status == SessionStatus.stopped.value
        assert row.error is None
        assert (row.budget_state or {}).get("stop_reason") == "deadline_exceeded"
        assert fake.calls == 0


def test_budget_stop_has_no_error_cancels_pending_approval_and_new_run_gets_fresh_steps(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    replies = [
        LLMResponse(None, [_read_call("a.txt")], "tool_calls", 10, 5),
        LLMResponse(None, [_write_call("out.txt")], "tool_calls", 10, 5),
        LLMResponse("done", [], "stop", 10, 5),
    ]
    with authed_client() as client, _fake_llm(ScriptedLLM(replies)) as fake:
        session_id = _create_run(client, tmp_path, "steps", fake, budget={"max_steps": 2})
        # The second inference asked for a write_file approval: the run waits for it.
        wait_for(client, session_id, "awaiting_confirmation")
        approvals = client.get(f"/api/sessions/{session_id}/approvals").json()
        assert approvals, "expected a pending approval"

        # Budget accounting ran; a new message after a stopped budget would start a fresh
        # run, but here the session is still waiting. Exercise the budget stop that leaves
        # a pending approval behind: it must withdraw it without writing session.error.
        async def budget_stop_with_pending() -> None:
            async with session_factory() as db:
                row = await db.get(Session, session_id)
                assert row is not None
                await mark_budget_stop(db, session_id, row.run_generation or 0, "steps_exhausted")
                await db.commit()
            async with session_factory() as db:
                row = await db.get(Session, session_id)
                assert row is not None
                await supervisor._stop_session_by_budget(
                    db, session_id, row.run_generation or 0, "steps_exhausted"
                )

        asyncio.run(budget_stop_with_pending())
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["status"] == "stopped"
        assert body["error"] is None
        assert body["budget"]["stop_reason"] == "steps_exhausted"
        assert client.get(f"/api/sessions/{session_id}/approvals").json() == []

        # A new message through the API starts a new run with fresh steps and the reply
        # script continues where it stopped.
        sent = client.post(f"/api/sessions/{session_id}/messages", json={"content": "continue"})
        assert sent.status_code == 201
        wait_for(client, session_id, "completed")
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["status"] == "completed"
        assert body["budget"]["stop_reason"] is None


def test_budget_exceeded_from_summary_call_stops_session(tmp_path: Path) -> None:
    with _fake_llm(ScriptedLLM([LLMResponse("done", [], "stop", 1, 1)])) as fake:
        session_id = asyncio.run(_manual_session(tmp_path, "summary"))

        async def seed() -> None:
            async with session_factory() as db:
                row = await db.get(Session, session_id)
                assert row is not None
                row.budget_state = {"limits": {"max_steps": 10, "max_cost": "0.10"}}
                # A long compressible history forces the context manager to summarize.
                for index in range(14):
                    db.add(
                        Message(
                            session_id=session_id,
                            role="assistant",
                            content=None,
                            tool_calls=[
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps(
                                            {"path": "a.txt", "content": "x" * 800}
                                        ),
                                    },
                                }
                            ],
                        )
                    )
                    db.add(
                        Message(
                            session_id=session_id,
                            role="tool",
                            tool_call_id=f"call-{index}",
                            content=json.dumps({"ok": True, "data": "r" * 800}),
                        )
                    )
                await db.commit()

        asyncio.run(seed())
        asyncio.run(supervisor._run(session_id))

        row = asyncio.run(_session_row(session_id))
        assert row.status == SessionStatus.stopped.value
        assert row.error is None
        assert (row.budget_state or {}).get("stop_reason") == "cost_unknown"
        # The money ceiling with no pricing refused the boundary before any HTTP attempt.
        assert fake.calls == 0


def test_regular_completion_without_limits_is_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    replies = [
        LLMResponse(None, [_read_call("a.txt")], "tool_calls", 10, 5),
        LLMResponse("Done.", [], "stop", 20, 5),
    ]
    with authed_client() as client, _fake_llm(ScriptedLLM(replies)) as fake:
        session_id = _create_run(client, tmp_path, "plain", fake)
        wait_for(client, session_id, "completed")

        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["status"] == "completed"
        assert body["error"] is None
        assert fake.calls == 2
        messages = client.get(f"/api/sessions/{session_id}/messages").json()
        assert [item["content"] for item in messages if item["role"] == "assistant"] == [
            None,
            "Done.",
        ]


def test_resumed_approved_tool_is_refused_after_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent import api

    replies = [
        LLMResponse(None, [_write_call("out.txt")], "tool_calls", 10, 5),
        LLMResponse("done", [], "stop", 10, 5),
    ]
    with authed_client() as client, _fake_llm(ScriptedLLM(replies)):
        session_id = _create_run(
            client, tmp_path, "deadline-tool", ScriptedLLM([]), budget={"deadline_seconds": 3600}
        )
        wait_for(client, session_id, "awaiting_confirmation")
        approval = client.get(f"/api/sessions/{session_id}/approvals").json()[0]

        async def expire_run() -> None:
            async with session_factory() as db:
                row = await db.get(Session, session_id)
                assert row is not None
                state = dict(row.budget_state or {})
                state["started_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
                row.budget_state = state
                await db.commit()

        asyncio.run(expire_run())

        # Only the runtime check stands between the approval and the tool body here.
        async def not_blocked(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(api, "_session_blocked", not_blocked)
        resolved = client.post(f"/api/approvals/{approval['id']}", json={"decision": "approve"})
        assert resolved.status_code == 200, resolved.text
        wait_for(client, session_id, "stopped")
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["error"] is None
        assert body["budget"]["stop_reason"] == "deadline_exceeded"
    assert not (tmp_path / "out.txt").exists()
