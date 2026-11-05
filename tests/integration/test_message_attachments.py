"""Text files attached to a user message reach the model and keep their record in the history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent.api import supervisor
from agent.llm import LLMResponse
from tests.integration.test_message_tree_api import (
    setup_completed,
    tree_client,
    user_messages,
    wait_for,
)


class CapturingLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self) -> None:
        self.requests: list[list[dict[str, Any]]] = []

    async def chat(
        self, messages: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any
    ) -> LLMResponse:
        self.requests.append(messages)
        return LLMResponse("Прочитал", [], "stop", 10, 5)


@pytest.fixture
def fake() -> Any:
    original = supervisor.llm
    llm = CapturingLLM()
    supervisor.llm = llm  # type: ignore[assignment]
    yield llm
    supervisor.llm = original


NOTES = {"name": "notes.md", "text": "# Заметки\nстрока"}
SCRIPT = {"name": "../secret/run.py", "text": 'print("ok")'}
COMPOSED = (
    "Посмотри файлы\n\n"
    '<attachment name="notes.md">\n# Заметки\nстрока\n</attachment>\n\n'
    '<attachment name="run.py">\nprint("ok")\n</attachment>'
)


def send(client: TestClient, session_id: str, content: str, files: list[Any]) -> Any:
    return client.post(
        f"/api/sessions/{session_id}/messages", json={"content": content, "attachments": files}
    )


def test_text_files_reach_the_model_after_the_typed_text(tmp_path: Path, fake: Any) -> None:
    with tree_client() as client:
        session_id = setup_completed(client, tmp_path)
        response = send(client, session_id, "Посмотри файлы  \n", [NOTES, SCRIPT])
        assert response.status_code == 201
        message = response.json()
        assert message["content"] == COMPOSED
        assert message["attachments"] == {
            "text_end": len("Посмотри файлы"),
            "files": [
                {"name": "notes.md", "size": len(NOTES["text"].encode()), "chars": 16},
                {"name": "run.py", "size": 11, "chars": 11},
            ],
        }
        wait_for(client, session_id, "completed")
        assert fake.requests[-1][-1] == {"role": "user", "content": COMPOSED}
        stored = user_messages(client, session_id)[-1]
        assert stored["attachments"] == message["attachments"]

        plain = send(client, session_id, "  без файлов  ", [])
        assert plain.status_code == 201
        assert plain.json()["content"] == "  без файлов  "
        assert plain.json()["attachments"] is None


def test_files_can_go_without_text_and_start_a_session(tmp_path: Path, fake: Any) -> None:
    with tree_client() as client:
        root = tmp_path / "files"
        root.mkdir()
        project = client.post("/api/projects", json={"name": "files", "root_path": str(root)})
        created = client.post(
            "/api/sessions",
            json={"project_id": project.json()["id"], "prompt": "", "attachments": [NOTES]},
        )
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert created.json()["title"] == "notes.md"
        wait_for(client, session_id, "completed")
        first = user_messages(client, session_id)[0]
        assert first["content"] == '<attachment name="notes.md">\n# Заметки\nстрока\n</attachment>'
        assert first["attachments"]["text_end"] == 0

        assert send(client, session_id, "", [SCRIPT]).status_code == 201
        assert send(client, session_id, "   ", []).status_code == 422


@pytest.mark.parametrize(
    ("files", "detail"),
    [
        ([{"name": f"{index}.txt", "text": "x"} for index in range(11)], "не больше 10"),
        ([{"name": "big.txt", "text": "я" * 530_000}], "больше 1 МБ"),
        ([{"name": "image.png", "text": "\x89PNG\x00data"}], "не текстовый"),
        ([{"name": f"{index}.txt", "text": "x" * 13_000} for index in range(10)], "четверти"),
    ],
)
def test_files_over_a_limit_are_refused_without_a_new_message(
    tmp_path: Path, fake: Any, files: list[Any], detail: str
) -> None:
    with tree_client() as client:
        session_id = setup_completed(client, tmp_path)
        before = len(user_messages(client, session_id))
        response = send(client, session_id, "посмотри", files)
        assert response.status_code == 422
        assert detail in response.json()["detail"]
        assert len(user_messages(client, session_id)) == before
        assert client.get(f"/api/sessions/{session_id}").json()["status"] == "completed"


def test_edit_regenerate_and_fork_keep_the_files(tmp_path: Path, fake: Any) -> None:
    with tree_client() as client:
        session_id = setup_completed(client, tmp_path)
        sent = send(client, session_id, "Посмотри файлы", [NOTES, SCRIPT]).json()
        wait_for(client, session_id, "completed")

        edited = client.post(
            f"/api/sessions/{session_id}/messages/{sent['id']}/edit", json={"content": "Новый"}
        )
        assert edited.status_code == 201
        assert edited.json()["content"] == COMPOSED.replace("Посмотри файлы", "Новый")
        assert edited.json()["attachments"]["text_end"] == len("Новый")
        assert edited.json()["attachments"]["files"] == sent["attachments"]["files"]
        wait_for(client, session_id, "completed")

        only_files = client.post(
            f"/api/sessions/{session_id}/messages/{edited.json()['id']}/edit", json={"content": ""}
        )
        assert only_files.status_code == 201
        assert only_files.json()["content"] == COMPOSED.removeprefix("Посмотри файлы\n\n")
        assert only_files.json()["attachments"]["text_end"] == 0
        wait_for(client, session_id, "completed")

        answer = [
            row
            for row in client.get(f"/api/sessions/{session_id}/messages").json()
            if row["role"] == "assistant" and not row["skipped"]
        ][-1]
        again = client.post(
            f"/api/sessions/{session_id}/messages/{answer['id']}/regenerate", json={}
        )
        assert again.status_code == 201
        assert again.json()["attachments"] == only_files.json()["attachments"]
        wait_for(client, session_id, "completed")

        fork = client.post(
            f"/api/sessions/{session_id}/messages/{again.json()['id']}/fork", json={}
        )
        assert fork.status_code == 201
        copied = user_messages(client, fork.json()["id"])[-1]
        assert copied["content"] == only_files.json()["content"]
        assert copied["attachments"] == only_files.json()["attachments"]
