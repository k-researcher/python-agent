"""A waited child must finish its approved work before the parent continues."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from agent.api import supervisor
from agent.llm import LLMResponse
from tests.integration.helpers import authed_client
from tests.integration.test_runtime_robustness import approve_all, project_session, wait_for


class BlogLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]], **_options: Any
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            name = "run_agent"
            arguments: dict[str, Any] = {
                "mode": "dev",
                "prompt": "Write the blog post",
                "wait": True,
            }
        elif self.calls in {2, 3}:
            name = "write_file"
            arguments = {"path": f"post-{self.calls - 1}.md", "content": "Blog post"}
        elif self.calls == 4:
            return LLMResponse("Both posts written", [], "stop", 1, 1)
        else:
            assert self.calls == 5, "The parent must not duplicate the child's work"
            result = json.loads(messages[-1]["content"])
            assert result["output"]["status"] == "completed"
            assert result["output"]["response"] == "Both posts written"
            return LLMResponse("Child finished", [], "stop", 1, 1)
        call = {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return LLMResponse(None, [call], "tool_calls", 1, 1)


def test_parent_continues_once_after_two_child_approval_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = BlogLLM()
    monkeypatch.setattr(supervisor, "llm", fake)
    monkeypatch.setattr(supervisor, "_semaphore", asyncio.Semaphore(1))
    with authed_client() as client:
        parent_id = project_session(client, tmp_path, "Create blog")
        wait_for(client, parent_id, "awaiting_confirmation")
        approve_all(client, parent_id)
        # The first child is created before its write approval is requested.
        children: list[dict[str, Any]] = []
        deadline = time.monotonic() + 3
        while not children and time.monotonic() < deadline:
            children = client.get(f"/api/sessions/{parent_id}/children").json()
            time.sleep(0.01)
        assert len(children) == 1
        child_id = children[0]["id"]
        for call_count in (2, 3):
            wait_for(client, child_id, "awaiting_confirmation")
            assert fake.calls == call_count
            assert client.get(f"/api/sessions/{parent_id}").json()["status"] == "running"
            messages = client.get(f"/api/sessions/{parent_id}/messages").json()
            assert not [row for row in messages if row["role"] == "tool"]
            assert not (tmp_path / f"post-{call_count - 1}.md").exists()
            approve_all(client, child_id)
        wait_for(client, parent_id, "completed")
        assert fake.calls == 5
        assert (tmp_path / "post-1.md").read_text() == "Blog post"
        assert (tmp_path / "post-2.md").read_text() == "Blog post"
