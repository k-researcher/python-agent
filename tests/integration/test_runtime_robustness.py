from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.api import supervisor
from agent.llm import LLMResponse
from agent.runtime import CUT_OFF_NOTICE
from tests.integration.helpers import authed_client


def wait_for(client: Any, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == expected:
            return
        time.sleep(0.02)
    status = client.get(f"/api/sessions/{session_id}").json()
    raise AssertionError(f"Session did not reach {expected}: {status}")


def write_call(path: str) -> dict[str, Any]:
    # A provider that omits tool-call IDs entirely.
    return {
        "type": "function",
        "function": {"name": "write_file", "arguments": f'{{"path":"{path}","content":"x"}}'},
    }


class RepeatingWritesLLM:
    """Two approval rounds with ID-less tool calls: IDs must never collide."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        self.calls += 1
        if self.calls <= 2:
            return LLMResponse(None, [write_call(f"f{self.calls}.txt")], "tool_calls", 1, 1)
        return LLMResponse("done", [], "stop", 1, 1)


def project_session(client: Any, root: Path, prompt: str) -> str:
    project = client.post("/api/projects", json={"name": prompt, "root_path": str(root)}).json()
    session = client.post(
        "/api/sessions", json={"project_id": project["id"], "prompt": prompt, "mode": "dev"}
    ).json()
    return str(session["id"])


def approve_all(client: Any, session_id: str) -> None:
    for approval in client.get(f"/api/sessions/{session_id}/approvals").json():
        response = client.post(f"/api/approvals/{approval['id']}", json={"decision": "approve"})
        assert response.status_code == 200


def test_tool_call_ids_stay_unique_across_approval_rounds(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = RepeatingWritesLLM()  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "ids")
            wait_for(client, session_id, "awaiting_confirmation")
            approve_all(client, session_id)
            wait_for(client, session_id, "awaiting_confirmation")
            approve_all(client, session_id)
            wait_for(client, session_id, "completed")
        assert (tmp_path / "f1.txt").exists() and (tmp_path / "f2.txt").exists()
    finally:
        supervisor.llm = original


class InvalidArgumentsLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[str] = []

    async def chat(
        self, messages: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            bad = {
                "id": "x",
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path": 5}'},
            }
            return LLMResponse(None, [bad], "tool_calls", 1, 1)
        self.seen = [str(item.get("content")) for item in messages if item["role"] == "tool"]
        return LLMResponse("fixed", [], "stop", 1, 1)


def test_schema_invalid_arguments_fail_without_approval(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = InvalidArgumentsLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "schema")
            wait_for(client, session_id, "completed")
            assert client.get(f"/api/sessions/{session_id}/approvals").json() == []
        assert any("Invalid arguments for write_file" in text for text in fake.seen)
    finally:
        supervisor.llm = original


class TruncatingLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.budgets: list[int | None] = []

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **options: Any) -> Any:
        self.budgets.append(options.get("max_tokens"))
        if len(self.budgets) == 1:
            return LLMResponse("partial", [], "length", 1, 1)
        if len(self.budgets) == 2:
            return LLMResponse("still partial", [], "length", 1, 1)
        return LLMResponse("complete answer", [], "stop", 1, 1)


def test_cut_off_reply_gets_a_notice_and_the_next_step_completes(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = TruncatingLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "truncation")
            wait_for(client, session_id, "completed")
            messages = client.get(f"/api/sessions/{session_id}/messages").json()
        # The partial texts are kept for diagnosis only and never reach the model.
        interrupted = [m["content"] for m in messages if m["kind"] == "interrupted"]
        assert interrupted == ["partial", "still partial"]
        notices = [m for m in messages if m["role"] == "user" and m["content"] == CUT_OFF_NOTICE]
        assert len(notices) == 2
        assert messages[-1]["content"] == "complete answer"
        assert len(fake.budgets) == 3
    finally:
        supervisor.llm = original


class BrokenStreamLLM:
    """Break the stream after the first fragment ``breaks`` times, then answer."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, breaks: int) -> None:
        self.breaks = breaks
        self.calls = 0

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        from agent.llm import error_for
        from agent.llm_errors import LLMErrorKind, ProviderErrorInfo

        self.calls += 1
        if self.calls <= self.breaks:
            info = ProviderErrorInfo(LLMErrorKind.timeout, None, None, "ReadTimeout", True)
            raise error_for(info, "LLM stream was interrupted: ReadTimeout", started=True)
        return LLMResponse("complete answer", [], "stop", 1, 1)


def test_broken_stream_is_sent_again(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = BrokenStreamLLM(breaks=2)
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "broken-stream")
            wait_for(client, session_id, "completed")
            messages = client.get(f"/api/sessions/{session_id}/messages").json()
        assert messages[-1]["content"] == "complete answer"
        assert fake.calls == 3
    finally:
        supervisor.llm = original


def test_broken_stream_retries_are_bounded(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = BrokenStreamLLM(breaks=10)
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "broken-stream-limit")
            wait_for(client, session_id, "error")
            session = client.get(f"/api/sessions/{session_id}").json()
        assert "stream was interrupted" in session["error"]
        assert fake.calls == 3
    finally:
        supervisor.llm = original


class AlwaysCutLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        self.calls += 1
        return LLMResponse(f"partial {self.calls}", [], "length", 1, 1)


def test_third_cut_in_a_row_ends_the_run(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = AlwaysCutLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "always-cut")
            wait_for(client, session_id, "error")
            session = client.get(f"/api/sessions/{session_id}").json()
        assert "cut off by the output limit" in session["error"]
        assert fake.calls == 3
    finally:
        supervisor.llm = original


class CutBetweenStepsLLM:
    """Cut once, run a tool, cut twice more, then answer."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        self.calls += 1
        if self.calls == 2:
            call = {
                "id": "read",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
            }
            return LLMResponse(None, [call], "tool_calls", 1, 1)
        if self.calls in {1, 3, 4}:
            return LLMResponse("partial", [], "length", 1, 1)
        return LLMResponse("complete answer", [], "stop", 1, 1)


def test_a_normal_step_resets_the_cut_off_count(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    original = supervisor.llm
    fake = CutBetweenStepsLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with authed_client() as client:
            session_id = project_session(client, tmp_path, "cut-reset")
            wait_for(client, session_id, "completed")
        assert fake.calls == 5
    finally:
        supervisor.llm = original
