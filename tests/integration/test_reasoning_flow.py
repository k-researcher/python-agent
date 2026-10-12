from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.api import settings, supervisor
from agent.llm import LLMResponse
from tests.integration.helpers import authed_client


class RecordingLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.efforts: list[str | None] = []

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **options: Any) -> Any:
        self.efforts.append(options.get("reasoning_effort"))
        return LLMResponse("answer", [], "stop", 1, 1, reasoning="thinking text")


def wait_for(client: Any, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == expected:
            return
        time.sleep(0.02)
    raise AssertionError(expected)


def test_per_message_reasoning_level_and_stored_reasoning(tmp_path: Path, monkeypatch: Any) -> None:
    from agent.model_config import ReasoningSpec

    original = supervisor.llm
    fake = RecordingLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    monkeypatch.setattr(settings, "llm_profiles", {})
    import agent.model_config as model_config

    real_resolve = model_config._resolve

    def with_reasoning(*args: Any, **kwargs: Any) -> Any:
        registry = real_resolve(*args, **kwargs)
        for model_id, model in list(registry.models.items()):
            object.__setattr__(
                model,
                "reasoning",
                ReasoningSpec(
                    supported=True,
                    allowed_efforts=["none", "low", "high"],
                    default_effort="low",
                    wire_parameter="reasoning_effort",
                ),
            )
            registry.models[model_id] = model
        return registry

    monkeypatch.setattr(model_config, "_resolve", with_reasoning)
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Reasoning", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={"project_id": project["id"], "prompt": "Think", "mode": "ask"},
            ).json()
            wait_for(client, session["id"], "completed")

            bad = client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"content": "x", "reasoning_effort": "ultra"},
            )
            assert bad.status_code == 422
            sent = client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"content": "Think harder", "reasoning_effort": "high"},
            )
            assert sent.status_code == 201
            wait_for(client, session["id"], "completed")
            history = client.get(f"/api/sessions/{session['id']}/messages").json()
        assert fake.efforts == ["low", "high"]
        assistants = [item for item in history if item["role"] == "assistant"]
        assert [item["reasoning_effort"] for item in assistants] == ["low", "high"]
        assert assistants[-1]["reasoning_content"] == "thinking text"
        assert history[-2]["reasoning_effort"] == "high"
    finally:
        supervisor.llm = original
