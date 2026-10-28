"""Exercise project skills through the authenticated API and runtime tool loop."""

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
from tests.integration.helpers import authed_client


class SkillModel:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, tool_name: str = "read_skill", name: str = "site-copy") -> None:
        self.tool_name = tool_name
        self.name = name
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any
    ) -> LLMResponse:
        self.requests.append((messages, tools))
        if len(self.requests) == 1:
            call = {
                "id": "skill-call",
                "type": "function",
                "function": {
                    "name": self.tool_name,
                    "arguments": json.dumps({"name": self.name}),
                },
            }
            return LLMResponse(None, [call], "tool_calls", 10, 5)
        return LLMResponse("Done.", [], "stop", 20, 5)


def make_skill(root: Path, name: str = "site-copy") -> None:
    directory = root / ".agent" / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Prepare a site draft\n"
        "tools: [untrusted_write]\n---\nUse a clear structure.",
        encoding="utf-8",
    )


def completed_messages(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        status = response.json()["status"]
        if status == "completed":
            messages = client.get(f"/api/sessions/{session_id}/messages")
            assert messages.status_code == 200
            return list(messages.json())
        assert status not in {"error", "awaiting_confirmation"}, response.text
        time.sleep(0.01)
    raise AssertionError("Skill session did not complete")


@pytest.mark.parametrize("mode", ["dev", "ask"])
def test_skill_round_trip_needs_no_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    make_skill(tmp_path)
    model = SkillModel()
    monkeypatch.setattr(supervisor, "llm", model)
    with authed_client() as client:
        project = client.post("/api/projects", json={"name": "Skills", "root_path": str(tmp_path)})
        assert project.status_code == 201, project.text
        session = client.post(
            "/api/sessions",
            json={
                "project_id": project.json()["id"],
                "prompt": "Use the site skill",
                "mode": mode,
                "auto_start": True,
            },
        )
        assert session.status_code == 201, session.text
        session_id = session.json()["id"]
        messages = completed_messages(client, session_id)
        approvals = client.get(f"/api/sessions/{session_id}/approvals?pending_only=false")
        assert approvals.status_code == 200
        assert approvals.json() == []

    result = json.loads(
        next(message["content"] for message in messages if message["role"] == "tool")
    )
    assert result == {
        "success": True,
        "output": {"name": "site-copy", "instructions": "Use a clear structure."},
    }
    assert len(model.requests) == 2
    assert any("Available skills:" in str(message.get("content")) for message in messages)
    definitions = {item["function"]["name"] for item in model.requests[0][1]}
    assert "read_skill" in definitions
    assert "untrusted_write" not in definitions
    assert any(message["role"] == "tool" for message in model.requests[1][0])
    assert "read_skill" not in {
        item["function"]["name"] for item in supervisor.registry.definitions(mode)
    }


def test_skill_manifest_cannot_register_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_skill(tmp_path)
    model = SkillModel(tool_name="untrusted_write")
    monkeypatch.setattr(supervisor, "llm", model)
    with authed_client() as client:
        project = client.post("/api/projects", json={"name": "Skills", "root_path": str(tmp_path)})
        assert project.status_code == 201
        session = client.post(
            "/api/sessions",
            json={
                "project_id": project.json()["id"],
                "prompt": "Try the unregistered tool",
                "mode": "dev",
                "auto_start": True,
            },
        )
        assert session.status_code == 201
        session_id = session.json()["id"]

        async def wait_run():
            task = supervisor._tasks.get(session_id)
            if task is not None:
                async with asyncio.timeout(5):
                    await task

        client.portal.call(wait_run)
        state = client.get(f"/api/sessions/{session_id}").json()
        messages = client.get(f"/api/sessions/{session_id}/messages").json()
        assert state["status"] == "error"
        assert len(model.requests) == 1
        assert not any(message["role"] == "tool" for message in messages)
        assert not any(
            message["role"] == "assistant" and message["kind"] == "normal" for message in messages
        )
