"""Integration tests for the HTTP message tree routes.

The router is not yet wired into ``agent.main``, so each test attaches it to the test
application fixture-wise (``app.include_router``) before the authed client is created.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent.api import supervisor
from agent.llm import LLMResponse
from agent.main import app
from agent.message_tree_api import router
from tests.integration.helpers import authed_client


def _router_installed() -> bool:
    return any(
        getattr(route, "path", "") == "/api/sessions/{session_id}/messages/{message_id}/siblings"
        for route in app.routes
    )


@contextmanager
def tree_client() -> Iterator[TestClient]:
    if not _router_installed():
        app.include_router(router)
    with authed_client() as client:
        yield client


class EchoLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, answer: str = "The answer") -> None:
        self.answer = answer
        self.calls = 0

    async def chat(
        self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(self.answer, [], "stop", 10, 5)


class SleepingLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, delay: float = 0.4) -> None:
        self.delay = delay
        self.calls = 0

    async def chat(
        self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any
    ) -> LLMResponse:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return LLMResponse("The answer", [], "stop", 10, 5)


class ToolCallingLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.calls = 0
        self.final_answer = "The final answer"

    async def chat(
        self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                None,
                [
                    {
                        "id": "call_test_1",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": "{}"},
                    }
                ],
                "tool_calls",
                10,
                5,
            )
        return LLMResponse(self.final_answer, [], "stop", 10, 5)


def wait_for(client: TestClient, session_id: str, expected: str, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{session_id}").json()["status"]
        if status == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"Session did not reach {expected}")


def setup_completed(client: TestClient, tmp_path: Path, prompt: str = "hello") -> str:
    root = tmp_path / prompt.replace(" ", "_")
    root.mkdir(parents=True, exist_ok=True)
    project = client.post("/api/projects", json={"name": prompt, "root_path": str(root)}).json()
    session = client.post(
        "/api/sessions",
        json={
            "project_id": project["id"],
            "prompt": prompt,
            "mode": "ask",
            "auto_start": True,
        },
    ).json()
    wait_for(client, session["id"], "completed")
    return str(session["id"])


def user_messages(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    return [
        m for m in client.get(f"/api/sessions/{session_id}/messages").json() if m["role"] == "user"
    ]


def active_assistant(client: TestClient, session_id: str) -> dict[str, Any]:
    return next(
        m
        for m in client.get(f"/api/sessions/{session_id}/messages").json()
        if m["role"] == "assistant" and not m["skipped"]
    )


def test_edit_creates_new_version_hides_old_and_starts_run(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = EchoLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            old_id = int(user_messages(client, session_id)[0]["id"])

            response = client.post(
                f"/api/sessions/{session_id}/messages/{old_id}/edit",
                json={"content": "edited prompt"},
            )
            assert response.status_code == 201
            new = response.json()
            assert new["content"] == "edited prompt"
            assert new["skipped"] is False
            assert new["id"] != old_id

            # The edit starts a new run that produces a fresh answer.
            wait_for(client, session_id, "completed")
            assert fake.calls == 2

            after = client.get(f"/api/sessions/{session_id}/messages").json()
            old_row = next(m for m in after if m["id"] == old_id)
            assert old_row["skipped"] is True
            new_row = next(m for m in after if m["id"] == new["id"])
            assert new_row["skipped"] is False
            active_assistant_row = next(
                m for m in after if m["role"] == "assistant" and not m["skipped"]
            )
            assert active_assistant_row["content"] == "The answer"
    finally:
        supervisor.llm = original


def test_edit_foreign_message_returns_404(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            s1 = setup_completed(client, tmp_path, "first")
            s2 = setup_completed(client, tmp_path, "second")
            other = int(user_messages(client, s2)[0]["id"])
            response = client.post(
                f"/api/sessions/{s1}/messages/{other}/edit", json={"content": "x"}
            )
            assert response.status_code == 404
    finally:
        supervisor.llm = original


def test_edit_nonexistent_message_returns_404(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            response = client.post(
                f"/api/sessions/{session_id}/messages/999999/edit",
                json={"content": "x"},
            )
            assert response.status_code == 404
    finally:
        supervisor.llm = original


def test_edit_while_run_in_progress_returns_409(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = SleepingLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path, "conflict")
            user_id = int(user_messages(client, session_id)[0]["id"])
            first = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": "first edit"},
            )
            assert first.status_code == 201
            wait_for(client, session_id, "running")
            second = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": "second edit"},
            )
            assert second.status_code == 409
            wait_for(client, session_id, "completed")
            after = client.get(f"/api/sessions/{session_id}/messages").json()
            # The rejected 409 added nothing; only the first edit's version is active.
            active_users = [m for m in after if m["role"] == "user" and not m["skipped"]]
            assert len(active_users) == 1
            assert active_users[0]["content"] == "first edit"
    finally:
        supervisor.llm = original


def test_regenerate_with_llm_profile(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = EchoLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            assistant = active_assistant(client, session_id)
            response = client.post(
                f"/api/sessions/{session_id}/messages/{assistant['id']}/regenerate",
                json={"llm_profile": "default"},
            )
            assert response.status_code == 201
            new_user = response.json()
            assert new_user["role"] == "user"
            wait_for(client, session_id, "completed")
            assert fake.calls == 2
            assert client.get(f"/api/sessions/{session_id}").json()["llm_profile"] == "default"
    finally:
        supervisor.llm = original


def test_branch_switches_and_siblings_show_versions(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            before = client.get(f"/api/sessions/{session_id}/messages").json()
            user_v1 = next(m for m in before if m["role"] == "user")
            assistant_v1 = next(m for m in before if m["role"] == "assistant")

            # Siblings before any edit: a single version is active.
            siblings = client.get(f"/api/sessions/{session_id}/messages/{user_v1['id']}/siblings")
            assert siblings.status_code == 200
            assert siblings.json() == {"ids": [user_v1["id"]], "active_index": 0}

            # Edit creates a second version of the user turn, active.
            edit = client.post(
                f"/api/sessions/{session_id}/messages/{user_v1['id']}/edit",
                json={"content": "edited"},
            )
            assert edit.status_code == 201
            user_v2 = edit.json()
            wait_for(client, session_id, "completed")

            siblings_after = client.get(
                f"/api/sessions/{session_id}/messages/{user_v1['id']}/siblings"
            ).json()
            assert siblings_after["ids"] == [user_v1["id"], user_v2["id"]]
            assert siblings_after["active_index"] == 1

            # Switch back to the old branch through its leaf.
            branch = client.post(
                f"/api/sessions/{session_id}/branch",
                json={"leaf_id": assistant_v1["id"]},
            )
            assert branch.status_code == 200
            assert [m["id"] for m in branch.json()] == [
                before[0]["id"],
                user_v1["id"],
                assistant_v1["id"],
            ]
            siblings_switched = client.get(
                f"/api/sessions/{session_id}/messages/{user_v1['id']}/siblings"
            ).json()
            assert siblings_switched["ids"] == [user_v1["id"], user_v2["id"]]
            assert siblings_switched["active_index"] == 0
    finally:
        supervisor.llm = original


def test_fork_creates_new_session_with_path_copy(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            title = client.get(f"/api/sessions/{session_id}").json()["title"]
            assistant = active_assistant(client, session_id)

            response = client.post(
                f"/api/sessions/{session_id}/messages/{assistant['id']}/fork",
                json={},
            )
            assert response.status_code == 201
            copy = response.json()
            assert copy["id"] != session_id
            assert copy["title"] == f"{title} (ветка)"
            assert copy["status"] == "completed"
            copy_messages = client.get(f"/api/sessions/{copy['id']}/messages").json()
            source_messages = client.get(f"/api/sessions/{session_id}/messages").json()
            assert [m["role"] for m in copy_messages] == [m["role"] for m in source_messages]
            assert [m["content"] for m in copy_messages] == [m["content"] for m in source_messages]

            # A custom title overrides the default.
            response2 = client.post(
                f"/api/sessions/{session_id}/messages/{assistant['id']}/fork",
                json={"title": "my copy"},
            )
            assert response2.status_code == 201
            assert response2.json()["title"] == "my copy"
    finally:
        supervisor.llm = original


def test_edit_empty_content_returns_422(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            user_id = int(user_messages(client, session_id)[0]["id"])
            response = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": ""},
            )
            assert response.status_code == 422
    finally:
        supervisor.llm = original


def test_edit_content_length_boundaries(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = EchoLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            user_id = int(user_messages(client, session_id)[0]["id"])

            # One over the limit is rejected by validation; no run starts.
            too_long = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": "x" * 2_000_001},
            )
            assert too_long.status_code == 422
            assert fake.calls == 1

            # Exactly at the limit is accepted and starts a new run.
            at_limit = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": "x" * 2_000_000},
            )
            assert at_limit.status_code == 201
            assert at_limit.json()["content"] == "x" * 2_000_000
            # The edit's run would tokenize a 2M-character message (seconds of CPU) so
            # we do not wait for it; the 201 already proves the boundary is accepted.
            assert at_limit.json()["id"] != user_id
    finally:
        supervisor.llm = original


def test_fork_title_length_boundaries(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            assistant = active_assistant(client, session_id)

            too_long = client.post(
                f"/api/sessions/{session_id}/messages/{assistant['id']}/fork",
                json={"title": "x" * 301},
            )
            assert too_long.status_code == 422

            at_limit = client.post(
                f"/api/sessions/{session_id}/messages/{assistant['id']}/fork",
                json={"title": "x" * 300},
            )
            assert at_limit.status_code == 201
            assert at_limit.json()["title"] == "x" * 300
    finally:
        supervisor.llm = original


def test_extra_fields_are_rejected(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            user_id = int(user_messages(client, session_id)[0]["id"])
            response = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": "x", "bogus": 1},
            )
            assert response.status_code == 422
    finally:
        supervisor.llm = original


def test_whitespace_content_is_rejected(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            user_id = int(user_messages(client, session_id)[0]["id"])
            response = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit",
                json={"content": "   "},
            )
            assert response.status_code == 400
    finally:
        supervisor.llm = original


def test_failed_edit_rolls_back_session_status(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = EchoLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            s1 = setup_completed(client, tmp_path, "one")
            s2 = setup_completed(client, tmp_path, "two")
            other = int(user_messages(client, s2)[0]["id"])
            assert client.get(f"/api/sessions/{s1}").json()["status"] == "completed"

            response = client.post(
                f"/api/sessions/{s1}/messages/{other}/edit", json={"content": "x"}
            )
            assert response.status_code == 404
            assert fake.calls == 2  # the two completed set-ups, no extra run from the failure
            # _claim_idle had flipped s1 to pending; the rollback restored its idle state.
            assert client.get(f"/api/sessions/{s1}").json()["status"] == "completed"
    finally:
        supervisor.llm = original


def test_regenerate_rejects_invalid_effort_and_profile(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = EchoLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            assistant = active_assistant(client, session_id)
            before = client.get(f"/api/sessions/{session_id}/messages").json()
            url = f"/api/sessions/{session_id}/messages/{assistant['id']}/regenerate"
            assert client.post(url, json={"reasoning_effort": "banana"}).status_code == 422
            assert client.post(url, json={"llm_profile": "missing"}).status_code == 422
            # Nothing changed and no run started.
            assert client.get(f"/api/sessions/{session_id}/messages").json() == before
            assert client.get(f"/api/sessions/{session_id}").json()["status"] == "completed"
            assert fake.calls == 1
    finally:
        supervisor.llm = original


def test_regenerate_hides_old_branch_and_lists_both_versions(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path)
            old_user = user_messages(client, session_id)[0]
            old_assistant = active_assistant(client, session_id)
            response = client.post(
                f"/api/sessions/{session_id}/messages/{old_assistant['id']}/regenerate",
                json={},
            )
            assert response.status_code == 201
            new_user = response.json()
            wait_for(client, session_id, "completed")
            rows = {m["id"]: m for m in client.get(f"/api/sessions/{session_id}/messages").json()}
            assert rows[old_user["id"]]["skipped"] is True
            assert rows[old_assistant["id"]]["skipped"] is True
            assert rows[new_user["id"]]["skipped"] is False
            versions = client.get(
                f"/api/sessions/{session_id}/messages/{new_user['id']}/siblings"
            ).json()
            assert versions == {"ids": [old_user["id"], new_user["id"]], "active_index": 1}
    finally:
        supervisor.llm = original


def test_tree_operations_on_busy_session_return_409(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path, "busy")
            assistant = active_assistant(client, session_id)
            user_id = int(user_messages(client, session_id)[0]["id"])
            supervisor.llm = SleepingLLM(delay=1.0)  # type: ignore[assignment]
            edit = client.post(
                f"/api/sessions/{session_id}/messages/{user_id}/edit", json={"content": "again"}
            )
            assert edit.status_code == 201
            wait_for(client, session_id, "running")
            base = f"/api/sessions/{session_id}"
            regenerate = client.post(f"{base}/messages/{assistant['id']}/regenerate", json={})
            branch = client.post(f"{base}/branch", json={"leaf_id": assistant["id"]})
            fork = client.post(f"{base}/messages/{edit.json()['id']}/fork", json={})
            assert [regenerate.status_code, branch.status_code, fork.status_code] == [
                409,
                409,
                409,
            ]
            wait_for(client, session_id, "completed")
    finally:
        supervisor.llm = original


def test_fork_foreign_message_returns_404(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            first = setup_completed(client, tmp_path, "first")
            second = setup_completed(client, tmp_path, "second")
            foreign = active_assistant(client, second)
            response = client.post(f"/api/sessions/{first}/messages/{foreign['id']}/fork", json={})
            assert response.status_code == 404
    finally:
        supervisor.llm = original


def test_fork_drops_tool_rows_and_empty_tool_call_turns(tmp_path: Path) -> None:
    original = supervisor.llm
    fake = ToolCallingLLM()
    supervisor.llm = fake  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path, "tools")
            source = client.get(f"/api/sessions/{session_id}/messages").json()
            assert any(m["role"] == "tool" for m in source)  # the run used a tool
            final = active_assistant_with_text(source)
            response = client.post(
                f"/api/sessions/{session_id}/messages/{final['id']}/fork", json={}
            )
            assert response.status_code == 201
            copy = client.get(f"/api/sessions/{response.json()['id']}/messages").json()
            assert all(m["role"] != "tool" for m in copy)
            assert all(not m.get("tool_call_id") and not m.get("tool_calls") for m in copy)
            assert all((m["content"] or "").strip() for m in copy if m["role"] == "assistant")
            assert copy[-1]["content"] == fake.final_answer
    finally:
        supervisor.llm = original


def active_assistant_with_text(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        m
        for m in reversed(messages)
        if m["role"] == "assistant" and not m["skipped"] and (m["content"] or "").strip()
    )


def test_concurrent_edits_create_one_version(tmp_path: Path) -> None:
    original = supervisor.llm
    supervisor.llm = EchoLLM()  # type: ignore[assignment]
    try:
        with tree_client() as client:
            session_id = setup_completed(client, tmp_path, "race")
            user_id = int(user_messages(client, session_id)[0]["id"])
            supervisor.llm = SleepingLLM(delay=0.5)  # type: ignore[assignment]
            url = f"/api/sessions/{session_id}/messages/{user_id}/edit"
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(client.post, url, json={"content": f"edit {n}"}) for n in (1, 2)
                ]
                codes = sorted(future.result().status_code for future in futures)
            assert codes == [201, 409]
            wait_for(client, session_id, "completed")
            after = client.get(f"/api/sessions/{session_id}/messages").json()
            active_users = [m for m in after if m["role"] == "user" and not m["skipped"]]
            assert len(active_users) == 1
    finally:
        supervisor.llm = original
