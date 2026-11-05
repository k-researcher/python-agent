from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agent.api import supervisor
from agent.events import StreamPublisher, broker
from agent.llm import LLMResponse
from agent.sse import ChatStreamAccumulator
from tests.integration.helpers import authed_client
from tests.integration.test_reasoning_flow import wait_for


class StreamingLLM:
    """Send three reasoning fragments and one text fragment, as a streaming provider does."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, reasoning_tokens: list[int | None]) -> None:
        self.reasoning_tokens = reasoning_tokens

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **options: Any) -> Any:
        on_delta = options.get("on_delta")
        if on_delta is not None:
            for part in ("Счи", "таю ", "шаги"):
                await on_delta("reasoning", part)
                await asyncio.sleep(0.01)
            await on_delta("content", "ответ")
        return LLMResponse(
            "ответ",
            [],
            "stop",
            1,
            9,
            reasoning="Считаю шаги",
            reasoning_tokens=self.reasoning_tokens.pop(0),
        )


def assistants(client: Any, session_id: str) -> list[dict[str, Any]]:
    messages = client.get(f"/api/sessions/{session_id}/messages").json()
    return [m for m in messages if m["role"] == "assistant"]


def test_reply_keeps_reasoning_tokens_and_time(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = StreamingLLM([None, 42])  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Stats", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={"project_id": project["id"], "prompt": "Подумай", "mode": "ask"},
            ).json()
            wait_for(client, session["id"], "completed")
            client.post(f"/api/sessions/{session['id']}/messages", json={"content": "Ещё"})
            wait_for(client, session["id"], "completed")

            estimated, reported = (m["reasoning_stats"] for m in assistants(client, session["id"]))
            # Without a provider count, the three stream fragments are the estimate.
            assert estimated["tokens"] == 3 and estimated["exact"] is False
            assert reported["tokens"] == 42 and reported["exact"] is True
            for stats in (estimated, reported):
                assert stats["seconds"] is not None and 0 < stats["seconds"] < 5
    finally:
        supervisor.llm = original


def test_reply_without_reasoning_has_no_stats(tmp_path: Path) -> None:
    class Plain:
        endpoint = StreamingLLM.endpoint

        async def chat(self, *_: Any, **__: Any) -> Any:
            return LLMResponse("ответ", [], "stop", 1, 1)

    original = supervisor.llm
    supervisor.llm = Plain()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Plain", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={"project_id": project["id"], "prompt": "Привет", "mode": "ask"},
            ).json()
            wait_for(client, session["id"], "completed")
            assert [m["reasoning_stats"] for m in assistants(client, session["id"])] == [None]
    finally:
        supervisor.llm = original


async def test_stream_deltas_count_fragments_of_each_kind() -> None:
    publisher = StreamPublisher("s")
    assert publisher.reasoning_seconds is None
    async with broker.subscribe() as queue:
        await publisher.add("reasoning", "a")
        await publisher.add("reasoning", "b")
        await publisher.add("content", "c")
        await publisher.flush()
        await publisher.add("reasoning", "d")
        await publisher.flush()
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    counts = [(event["type"], event["payload"]["fragments"]) for event in events]
    assert counts == [("reasoning.delta", 2), ("message.delta", 1), ("reasoning.delta", 3)]
    assert publisher.reasoning_fragments == 3
    assert publisher.reasoning_seconds is not None and publisher.reasoning_seconds >= 0


def test_stream_result_reads_reported_reasoning_tokens() -> None:
    def result(usage: dict[str, Any]) -> LLMResponse:
        accumulator = ChatStreamAccumulator()
        accumulator.add(json.dumps({"choices": [{"delta": {"reasoning_content": "мысль"}}]}))
        accumulator.add(json.dumps({"choices": [{"delta": {"content": "ответ"}}]}))
        accumulator.add(
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage})
        )
        return accumulator.result()

    details = {"completion_tokens_details": {"reasoning_tokens": 20}}
    assert result({"completion_tokens": 30, **details}).reasoning_tokens == 20
    assert result({"completion_tokens": 30}).reasoning_tokens is None
