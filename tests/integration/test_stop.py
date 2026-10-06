from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.api import supervisor
from agent.llm import LLMResponse
from tests.integration.helpers import authed_client


class TwoWritesLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]], **_options: Any
    ) -> Any:
        calls = [
            {
                "id": f"write-{index}",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": f'{{"path":"file{index}.txt","content":"x"}}',
                },
            }
            for index in (1, 2)
        ]
        return LLMResponse(None, calls, "tool_calls", 10, 5)


def wait_for(client: Any, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Session did not reach {expected}")


def test_stop_cancels_pending_approvals_and_blocks_late_approval(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = TwoWritesLLM()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Stop project", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={"project_id": project["id"], "prompt": "Write two files", "mode": "dev"},
            ).json()
            wait_for(client, session["id"], "awaiting_confirmation")
            approvals = client.get(f"/api/sessions/{session['id']}/approvals").json()
            assert len(approvals) == 2

            assert client.post(f"/api/sessions/{session['id']}/stop").status_code == 200
            assert client.get(f"/api/sessions/{session['id']}/approvals").json() == []

            late = client.post(f"/api/approvals/{approvals[0]['id']}", json={"decision": "approve"})
            assert late.status_code == 409
            assert client.get(f"/api/sessions/{session['id']}").json()["status"] == "stopped"

            tool_messages = [
                item
                for item in client.get(f"/api/sessions/{session['id']}/messages").json()
                if item["role"] == "tool"
            ]
            assert len(tool_messages) == 2
            assert all("stopped" in item["content"] for item in tool_messages)
        assert not (tmp_path / "file1.txt").exists()
        assert not (tmp_path / "file2.txt").exists()
    finally:
        supervisor.llm = original_llm


class SimpleAnswerLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]], **_options: Any
    ) -> Any:
        return LLMResponse("continued", [], "stop", 1, 1)


def test_new_message_after_stop_runs_again(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = TwoWritesLLM()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Restart project", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={"project_id": project["id"], "prompt": "Write two files", "mode": "dev"},
            ).json()
            wait_for(client, session["id"], "awaiting_confirmation")
            client.post(f"/api/sessions/{session['id']}/stop")

            supervisor.llm = SimpleAnswerLLM()  # type: ignore[assignment]
            sent = client.post(
                f"/api/sessions/{session['id']}/messages", json={"content": "Continue"}
            )
            assert sent.status_code == 201
            wait_for(client, session["id"], "completed")
    finally:
        supervisor.llm = original_llm
