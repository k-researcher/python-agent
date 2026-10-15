"""A reply cut off by the output limit during reasoning gets one retry at a lower level."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from agent.api import settings, supervisor
from agent.llm import LLMResponse
from tests.integration.helpers import authed_client

CUT = LLMResponse("", [], "length", 10, 100, reasoning="a very long chain of thought")
ANSWER = LLMResponse("answer", [], "stop", 10, 5, reasoning="short thought")


class ScriptedLLM:
    """Return the scripted replies in order and record the requested reasoning levels."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, replies: list[LLMResponse]) -> None:
        self.replies = replies
        self.efforts: list[str | None] = []

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **options: Any) -> Any:
        self.efforts.append(options.get("reasoning_effort"))
        return self.replies[min(len(self.efforts), len(self.replies)) - 1]


@pytest.fixture
def reasoning_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every configured model the levels none, low and high."""
    import agent.model_config as model_config
    from agent.model_config import ReasoningSpec

    monkeypatch.setattr(settings, "llm_profiles", {})
    real_resolve = model_config._resolve

    def with_reasoning(*args: Any, **kwargs: Any) -> Any:
        registry = real_resolve(*args, **kwargs)
        for model in registry.models.values():
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
        return registry

    monkeypatch.setattr(model_config, "_resolve", with_reasoning)


def _run(root: Path, fake: ScriptedLLM, effort: str) -> tuple[str, list[dict[str, Any]]]:
    original = supervisor.llm
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Retry", "root_path": str(root)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={
                    "project_id": project["id"],
                    "prompt": "Think",
                    "mode": "ask",
                    "reasoning_effort": effort,
                },
            ).json()
            deadline = time.monotonic() + 5
            status = ""
            while time.monotonic() < deadline:
                status = client.get(f"/api/sessions/{session['id']}").json()["status"]
                if status in {"completed", "error", "stopped"}:
                    break
                time.sleep(0.02)
            messages = client.get(f"/api/sessions/{session['id']}/messages").json()
            return status, messages
    finally:
        supervisor.llm = original


@pytest.mark.usefixtures("reasoning_models")
def test_cut_reasoning_is_retried_once_at_a_lower_level(tmp_path: Path) -> None:
    fake = ScriptedLLM([CUT, ANSWER])
    status, messages = _run(tmp_path, fake, "high")
    assert status == "completed"
    assert fake.efforts == ["high", "low"]
    assistant = [item for item in messages if item["role"] == "assistant"]
    assert assistant[-1]["content"] == "answer"
    assert assistant[-1]["reasoning_effort"] == "low"
    # The cut attempt stays for diagnosis and never reaches the model again.
    interrupted = [item for item in messages if item["kind"] == "interrupted"]
    assert [(item["reasoning_effort"], item["content"]) for item in interrupted] == [("high", "")]


@pytest.mark.usefixtures("reasoning_models")
def test_second_cut_is_an_error_without_a_third_request(tmp_path: Path) -> None:
    fake = ScriptedLLM([CUT, CUT, ANSWER])
    status, _ = _run(tmp_path, fake, "high")
    assert status == "error"
    assert fake.efforts == ["high", "low"]


@pytest.mark.usefixtures("reasoning_models")
def test_no_retry_below_the_lowest_level(tmp_path: Path) -> None:
    fake = ScriptedLLM([CUT, ANSWER])
    status, _ = _run(tmp_path, fake, "none")
    assert status == "error"
    assert fake.efforts == ["none"]


@pytest.mark.usefixtures("reasoning_models")
def test_cut_reply_with_text_is_not_retried(tmp_path: Path) -> None:
    cut_text = LLMResponse("partial text", [], "length", 10, 100, reasoning="thought")
    fake = ScriptedLLM([cut_text, ANSWER])
    status, _ = _run(tmp_path, fake, "high")
    assert status == "error"
    assert fake.efforts == ["high"]
