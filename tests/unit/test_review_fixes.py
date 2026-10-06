"""Regression tests for the findings of the phase 1 review."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.llm import LLMResponse, LLMTransientError, redact
from agent.model_config import ReasoningSpec, ResolvedModel, ResolvedModelRegistry
from agent.models import Project, Session, SessionStatus, ToolCall, ToolCallStatus
from agent.runtime import AgentSupervisor
from agent.worker import wait_for_stop


def model(model_id: str, *, window: int = 16_000, max_tokens: int = 2_000) -> ResolvedModel:
    return ResolvedModel(
        id=model_id,
        provider_id="p",
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test-secret-value",
        configured=True,
        model=model_id,
        context_window=window,
        max_tokens=max_tokens,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
    )


def registry(*models: ResolvedModel, fallback: list[str] | None = None) -> ResolvedModelRegistry:
    return ResolvedModelRegistry(
        models={item.id: item for item in models},
        default_id=models[0].id,
        fallback=fallback or [],
        roles={},
        warnings=[],
        source="yaml",
        checksum="test",
    )


async def make_session(tmp_path: Path, **values: Any) -> str:
    await init_database()
    async with session_factory() as db:
        project = Project(name=f"review-{tmp_path.name}", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="review", configuration={}, **values)
        db.add(session)
        await db.commit()
        return session.id


def test_redact_removes_configured_and_generic_keys() -> None:
    text = 'Invalid key sk-test-secret-value; header "Bearer abcdefgh12345"; key=xyz-123456789'
    cleaned = redact(text, ("xyz-123456789",))
    assert "sk-test-secret-value" not in cleaned
    assert "abcdefgh12345" not in cleaned
    assert "xyz-123456789" not in cleaned


def test_container_bind_with_local_http_origin_is_accepted() -> None:
    Settings(
        host="0.0.0.0", public_origin="http://127.0.0.1:8080", extra_origins=[]
    ).validate_runtime_security()


async def test_claim_refuses_a_call_after_stop(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path, stop_requested=True)
    async with session_factory() as db:
        call = ToolCall(
            id="call_review_claim",
            session_id=session_id,
            name="write_file",
            arguments={},
            risk_level="local_write",
            status=ToolCallStatus.pending.value,
        )
        db.add(call)
        await db.commit()
        claimed = await AgentSupervisor(Settings())._claim_tool_call(db, call)
        assert not claimed
        assert call.status == ToolCallStatus.pending.value


async def test_worker_stop_watch_returns_after_stop(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path, status=SessionStatus.running.value)
    watcher = asyncio.create_task(wait_for_stop(session_id, interval=0.01))
    await asyncio.sleep(0.05)
    assert not watcher.done()
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        session.stop_requested = True
        await db.commit()
    await asyncio.wait_for(watcher, timeout=1)


class ScriptedClient:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.budgets: list[int | None] = []
        self.endpoint = "https://llm.example/v1/chat/completions"

    async def chat(self, _m: Any, _t: Any, **options: Any) -> LLMResponse:
        self.budgets.append(options.get("max_tokens"))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply  # type: ignore[no-any-return]


async def test_truncation_retry_stays_inside_the_context_window(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    supervisor = AgentSupervisor(Settings())
    small = model("small", window=16_000, max_tokens=2_000)
    client = ScriptedClient([LLMResponse("cut", [], "length", 1, 1)] * 3)
    await supervisor._complete(
        session_id, "chat:small", client, small, [], [], "{}", None, 13_000  # type: ignore[arg-type]
    )
    assert all(budget is not None and 13_000 + budget <= 16_000 for budget in client.budgets)


async def test_fallback_model_takes_the_request_when_the_primary_is_down(
    tmp_path: Path,
) -> None:
    session_id = await make_session(tmp_path)
    supervisor = AgentSupervisor(Settings())
    primary, light = model("main"), model("light")
    fallback_client = ScriptedClient([LLMResponse("from light", [], "stop", 1, 1)])
    supervisor._llm_clients["light"] = fallback_client  # type: ignore[assignment]
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        await db.refresh(session, attribute_names=["messages"])
        response = await supervisor._complete_with_fallback(
            db,
            session,
            registry(primary, light, fallback=["light"]),
            primary,
            [],
            LLMTransientError("HTTP 503"),
        )
    assert response.content == "from light"


async def test_without_fallback_the_transient_error_stays(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    supervisor = AgentSupervisor(Settings())
    primary = model("main")
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        await db.refresh(session, attribute_names=["messages"])
        with pytest.raises(LLMTransientError):
            await supervisor._complete_with_fallback(
                db, session, registry(primary), primary, [], LLMTransientError("HTTP 503")
            )
