import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent.api import settings, supervisor
from agent.config import LLMProfileSettings
from agent.llm import LLMResponse
from agent.main import app


class FakeApprovalLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "fake-write-call",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"created.txt","content":"unsafe"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
                prompt_tokens=10,
                completion_tokens=5,
            )
        return LLMResponse(
            content="Operation was rejected.",
            tool_calls=[],
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
        )


class FakeChildLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    async def chat(
        self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]
    ) -> LLMResponse:
        user_messages = [item.get("content") for item in messages if item["role"] == "user"]
        is_child = any(content == "Return child result" for content in user_messages)
        has_tool_result = any(item["role"] == "tool" for item in messages)
        if is_child:
            return LLMResponse("Child result", [], "stop", 10, 3)
        if has_tool_result:
            return LLMResponse("Parent received child result", [], "stop", 20, 4)
        return LLMResponse(
            content=None,
            tool_calls=[
                {
                    "id": "child-call",
                    "type": "function",
                    "function": {
                        "name": "run_agent",
                        "arguments": '{"mode":"ask","prompt":"Return child result","wait":true}',
                    },
                }
            ],
            finish_reason="tool_calls",
            prompt_tokens=10,
            completion_tokens=5,
        )


class ConcurrencyProbe:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0


class FakeProfileLLM:
    def __init__(self, name: str, probe: ConcurrencyProbe) -> None:
        self.name = name
        self.probe = probe
        self.endpoint = f"https://{name}.example/v1/chat/completions"

    async def chat(
        self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.probe.active += 1
        self.probe.maximum = max(self.probe.maximum, self.probe.active)
        try:
            import asyncio

            await asyncio.sleep(0.05)
            return LLMResponse(f"Response from {self.name}", [], "stop", 10, 3)
        finally:
            self.probe.active -= 1


def wait_for_status(client: TestClient, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/sessions/{session_id}")
        if response.json()["status"] == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Session did not reach {expected}")


def test_session_lifecycle_and_websocket(tmp_path: Path) -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        project_response = client.post(
            "/api/projects",
            json={"name": "Integration project", "root_path": str(tmp_path)},
        )
        assert project_response.status_code == 201
        project_id = project_response.json()["id"]

        session_response = client.post(
            "/api/sessions",
            json={
                "project_id": project_id,
                "prompt": "Integration smoke test",
                "mode": "ask",
                "auto_start": False,
            },
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["id"]

        messages = client.get(f"/api/sessions/{session_id}/messages")
        assert messages.status_code == 200
        assert [item["role"] for item in messages.json()] == ["system", "user"]

        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json()["type"] == "connected"


def test_mutating_tool_waits_for_confirmation(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = FakeApprovalLLM()  # type: ignore[assignment]
    try:
        with TestClient(app) as client:
            project = client.post(
                "/api/projects",
                json={"name": "Approval project", "root_path": str(tmp_path)},
            ).json()
            session = client.post(
                "/api/sessions",
                json={
                    "project_id": project["id"],
                    "prompt": "Create a file",
                    "mode": "dev",
                    "auto_start": True,
                },
            ).json()

            wait_for_status(client, session["id"], "awaiting_confirmation")
            assert not (tmp_path / "created.txt").exists()

            approvals = client.get(f"/api/sessions/{session['id']}/approvals").json()
            assert len(approvals) == 1
            assert approvals[0]["tool_name"] == "write_file"

            response = client.post(
                f"/api/approvals/{approvals[0]['id']}",
                json={"decision": "reject", "comment": "Not allowed"},
            )
            assert response.status_code == 200
            wait_for_status(client, session["id"], "completed")
            assert not (tmp_path / "created.txt").exists()
    finally:
        supervisor.llm = original_llm


def test_child_agent_and_outbound_audit(tmp_path: Path) -> None:
    original_llm = supervisor.llm
    supervisor.llm = FakeChildLLM()  # type: ignore[assignment]
    try:
        with TestClient(app) as client:
            project = client.post(
                "/api/projects",
                json={"name": "Child project", "root_path": str(tmp_path)},
            ).json()
            parent = client.post(
                "/api/sessions",
                json={
                    "project_id": project["id"],
                    "prompt": "Delegate this task",
                    "mode": "dev",
                    "auto_start": True,
                },
            ).json()

            wait_for_status(client, parent["id"], "awaiting_confirmation")
            approval = client.get(f"/api/sessions/{parent['id']}/approvals").json()[0]
            assert approval["tool_name"] == "run_agent"
            response = client.post(
                f"/api/approvals/{approval['id']}", json={"decision": "approve"}
            )
            assert response.status_code == 200
            wait_for_status(client, parent["id"], "completed")

            children = client.get(f"/api/sessions/{parent['id']}/children").json()
            assert len(children) == 1
            assert children[0]["status"] == "completed"
            child_messages = client.get(
                f"/api/sessions/{children[0]['id']}/messages"
            ).json()
            assert child_messages[-1]["content"] == "Child result"

            audit = client.get(f"/api/sessions/{parent['id']}/outbound-audit").json()
            assert {item["status"] for item in audit} >= {"started", "completed"}
            assert all("payload" not in item for item in audit)
    finally:
        supervisor.llm = original_llm


def test_api_and_websocket_token_authentication() -> None:
    original_token = settings.api_token
    settings.api_token = "integration-secret"
    try:
        with TestClient(app) as client:
            assert client.get("/api/health").status_code == 401
            authorized = client.get(
                "/api/health", headers={"X-Agent-Token": "integration-secret"}
            )
            assert authorized.status_code == 200

            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"type": "auth", "token": "integration-secret"})
                assert websocket.receive_json()["type"] == "connected"
    finally:
        settings.api_token = original_token


def test_different_llm_profiles_run_concurrently(tmp_path: Path) -> None:
    original_default = settings.default_llm_profile
    original_profiles = settings.llm_profiles
    original_llm = supervisor.llm
    original_clients = supervisor._llm_clients
    probe = ConcurrencyProbe()
    try:
        settings.default_llm_profile = "fast"
        settings.llm_profiles = {
            "fast": LLMProfileSettings(
                base_url="https://fast.example/v1", api_key="test", model="fast-model"
            ),
            "reasoning": LLMProfileSettings(
                base_url="https://reasoning.example/v1",
                api_key="test",
                model="reasoning-model",
            ),
        }
        supervisor.llm = FakeProfileLLM("fast", probe)  # type: ignore[assignment]
        supervisor._llm_clients = {  # type: ignore[dict-item]
            "reasoning": FakeProfileLLM("reasoning", probe)
        }

        with TestClient(app) as client:
            project = client.post(
                "/api/projects",
                json={"name": "Multi-model project", "root_path": str(tmp_path)},
            ).json()
            sessions = [
                client.post(
                    "/api/sessions",
                    json={
                        "project_id": project["id"],
                        "prompt": f"Use {profile}",
                        "mode": "ask",
                        "llm_profile": profile,
                        "auto_start": True,
                    },
                ).json()
                for profile in ("fast", "reasoning")
            ]
            for session in sessions:
                wait_for_status(client, session["id"], "completed")

            assert [session["llm_profile"] for session in sessions] == [
                "fast",
                "reasoning",
            ]
            responses = [
                client.get(f"/api/sessions/{session['id']}/messages").json()[-1][
                    "content"
                ]
                for session in sessions
            ]
            assert responses == ["Response from fast", "Response from reasoning"]
            assert probe.maximum == 2
    finally:
        settings.default_llm_profile = original_default
        settings.llm_profiles = original_profiles
        supervisor.llm = original_llm
        supervisor._llm_clients = original_clients
