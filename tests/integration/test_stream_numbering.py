"""Stream numbers stay ordered inside one run, also across an approval pause.

A client keeps the stream with the highest number of the current run and drops a
different stream that has the same number. So a run that continues after an
approval must not start its stream numbers again.
"""

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent.api import supervisor
from agent.llm import LLMResponse
from tests.integration.helpers import authed_client


class WriteThenAnswerLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, writes: int) -> None:
        self.calls = 0
        self.writes = writes

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]], **_options: Any
    ) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.writes:
            return LLMResponse(
                content=None,
                tool_calls=[
                    {
                        "id": f"write-{self.calls}",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": f'{{"path":"file{self.calls}.txt","content":"x"}}',
                        },
                    }
                ],
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


def wait_for_status(client: TestClient, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Session did not reach {expected}")


def test_stream_numbers_keep_growing_after_each_approval(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = WriteThenAnswerLLM(writes=2)  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "Streams", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={
                    "project_id": project["id"],
                    "prompt": "Write two files",
                    "mode": "dev",
                    "auto_start": True,
                },
            ).json()
            for _ in range(2):
                wait_for_status(client, session["id"], "awaiting_confirmation")
                approval = client.get(f"/api/sessions/{session['id']}/approvals").json()
                pending = [item for item in approval if item["status"] == "pending"]
                assert len(pending) == 1
                response = client.post(
                    f"/api/approvals/{pending[0]['id']}", json={"decision": "approve"}
                )
                assert response.status_code == 200
            wait_for_status(client, session["id"], "completed")

            snapshot = client.get(f"/api/sessions/{session['id']}/snapshot/messages").json()
            streams = snapshot["completed_streams"]
            assert len(streams) == 3
            assert len({item["generation"] for item in streams}) == 1
            numbers = [item["stream_no"] for item in streams]
            assert numbers == sorted(numbers)
            assert len(set(numbers)) == len(numbers), numbers
    finally:
        supervisor.llm = original_llm
