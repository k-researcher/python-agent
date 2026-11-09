"""Speech recognition: local profiles, fallback to the next server and safe limits."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from agent import speech
from agent.database import session_factory
from agent.main import app
from agent.models import OutboundAudit
from tests.integration.helpers import authed_client

SETTINGS = "/api/settings/speech"
AUDIO = b"\x1aE\xdf\xa3recorded speech"
WEBM = {"Content-Type": "audio/webm;codecs=opus"}


async def _clean() -> None:
    async with session_factory() as db:
        await db.execute(delete(speech.SpeechProfile))
        await db.execute(delete(OutboundAudit).where(OutboundAudit.category == "speech"))
        await db.commit()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with authed_client() as test_client:
        test_client.portal.call(_clean)  # type: ignore[union-attr]
        yield test_client
        test_client.portal.call(_clean)  # type: ignore[union-attr]


def _servers(
    monkeypatch: pytest.MonkeyPatch, respond: Callable[[httpx.Request], httpx.Response]
) -> None:
    monkeypatch.setattr(
        speech, "client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )


def _add(client: TestClient, profile_id: str, port: int, **fields: Any) -> dict[str, Any]:
    response = client.put(
        f"{SETTINGS}/profiles/{profile_id}",
        json={"base_url": f"http://127.0.0.1:{port}/v1", **fields},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _audit(client: TestClient) -> list[OutboundAudit]:
    async def read() -> list[OutboundAudit]:
        async with session_factory() as db:
            rows = await db.scalars(
                select(OutboundAudit)
                .where(OutboundAudit.category == "speech")
                .order_by(OutboundAudit.id)
            )
            return list(rows)

    return client.portal.call(read)  # type: ignore[union-attr, no-any-return]


def test_transcription_without_profiles_says_it_is_not_set_up(client: TestClient) -> None:
    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)
    assert response.status_code == 503
    assert response.json()["detail"] == "Распознавание речи не настроено"


@pytest.mark.parametrize(
    "url",
    ["https://speech.example.com/v1", "http://192.168.1.5:8178/v1", "ftp://127.0.0.1/v1", "x"],
)
def test_only_a_server_on_this_computer_is_accepted(client: TestClient, url: str) -> None:
    response = client.put(f"{SETTINGS}/profiles/remote", json={"base_url": url})
    assert response.status_code == 422
    assert client.get(SETTINGS).json() == {"profiles": []}


def test_profile_id_and_language_are_checked(client: TestClient) -> None:
    local = {"base_url": "http://127.0.0.1:8178/v1"}
    assert client.put(f"{SETTINGS}/profiles/bad%20id", json=local).status_code == 422
    assert (
        client.put(f"{SETTINGS}/profiles/main", json={**local, "language": "Russian"}).status_code
        == 422
    )
    assert client.put(f"{SETTINGS}/profiles/main", json={"model": "m"}).status_code == 422


def test_text_comes_from_the_main_profile_and_the_audio_is_not_stored(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "  сделай коммит в main  "})

    _servers(monkeypatch, respond)
    _add(client, "main", 8178, language="ru", model="turbo")

    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)

    assert response.status_code == 200, response.text
    assert response.json() == {"text": "сделай коммит в main", "profile": "main"}
    assert str(seen[0].url) == "http://127.0.0.1:8178/v1/audio/transcriptions"
    body = seen[0].read()
    assert AUDIO in body
    assert b'name="language"\r\n\r\nru' in body
    assert b'name="model"\r\n\r\nturbo' in body
    assert b'filename="speech.webm"' in body
    [row] = _audit(client)
    assert (row.operation, row.status, row.detail, row.payload_bytes) == (
        "transcribe",
        "completed",
        "200",
        len(AUDIO),
    )
    assert row.session_id is None


def test_a_language_in_the_request_wins_over_the_profile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(200, json={"text": "hello"})

    _servers(monkeypatch, respond)
    _add(client, "main", 8178, language="ru")
    response = client.post("/api/speech/transcribe?language=en", content=AUDIO, headers=WEBM)
    assert response.status_code == 200
    assert b'name="language"\r\n\r\nen' in bodies[0]
    bad = client.post("/api/speech/transcribe?language=english", content=AUDIO, headers=WEBM)
    assert bad.status_code == 422


@pytest.mark.parametrize("failure", ["connect", "timeout", "500"])
def test_the_next_profile_answers_when_the_main_server_does_not(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8178:
            if failure == "connect":
                raise httpx.ConnectError("refused", request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(500)
        return httpx.Response(200, json={"text": "запасной"})

    _servers(monkeypatch, respond)
    _add(client, "main", 8178)
    _add(client, "spare", 8179)

    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)

    assert response.status_code == 200, response.text
    assert response.json() == {"text": "запасной", "profile": "spare"}
    assert [row.status for row in _audit(client)] == ["error", "completed"]


def test_a_refused_recording_is_not_sent_to_the_next_server(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ports: list[int | None] = []

    def respond(request: httpx.Request) -> httpx.Response:
        ports.append(request.url.port)
        return httpx.Response(400, json={"error": "bad audio at /internal/path"})

    _servers(monkeypatch, respond)
    _add(client, "main", 8178)
    _add(client, "spare", 8179)

    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)

    assert response.status_code == 422
    assert response.json()["detail"] == "Сервер распознавания отклонил запись: HTTP 400"
    assert ports == [8178]


def test_all_servers_down_gives_a_plain_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _servers(monkeypatch, respond)
    _add(client, "main", 8178)
    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)
    assert response.status_code == 503
    assert response.json()["detail"] == "Серверы распознавания не отвечают"


def test_a_disabled_profile_is_skipped_and_primary_changes_the_order(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ports: list[int | None] = []

    def respond(request: httpx.Request) -> httpx.Response:
        ports.append(request.url.port)
        return httpx.Response(200, json={"text": "ok"})

    _servers(monkeypatch, respond)
    _add(client, "main", 8178)
    _add(client, "spare", 8179)
    _add(client, "third", 8180)

    order = client.post(f"{SETTINGS}/profiles/third/primary").json()
    assert [item["id"] for item in order["profiles"]] == ["third", "main", "spare"]

    client.put(f"{SETTINGS}/profiles/third", json={"disabled": True})
    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)
    assert response.json()["profile"] == "main"
    assert ports == [8178]

    remaining = client.delete(f"{SETTINGS}/profiles/main").json()
    assert [item["id"] for item in remaining["profiles"]] == ["third", "spare"]
    assert client.delete(f"{SETTINGS}/profiles/main").status_code == 404


def test_format_size_and_empty_recordings_are_checked(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _servers(monkeypatch, lambda request: httpx.Response(200, json={"text": "x"}))
    _add(client, "main", 8178)
    text = {"Content-Type": "text/plain"}
    assert client.post("/api/speech/transcribe", content=AUDIO, headers=text).status_code == 415
    assert client.post("/api/speech/transcribe", content=b"", headers=WEBM).status_code == 422
    monkeypatch.setattr(speech, "MAX_AUDIO_BYTES", 8)
    response = client.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)
    assert response.status_code == 413
    assert _audit(client) == []


def test_profile_check_sends_silence_and_reports_the_answer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(200, json={"text": ""})

    _servers(monkeypatch, respond)
    _add(client, "main", 8178)
    result = client.post(f"{SETTINGS}/profiles/main/test").json()
    assert result["ok"] is True and result["status"] == 200
    assert b'filename="speech.wav"' in bodies[0] and b"RIFF" in bodies[0]
    assert client.post(f"{SETTINGS}/profiles/none/test").status_code == 404


def test_speech_routes_need_a_login() -> None:
    with TestClient(app) as anonymous:
        assert anonymous.get(SETTINGS).status_code == 401
        response = anonymous.post("/api/speech/transcribe", content=AUDIO, headers=WEBM)
        assert response.status_code == 401


@pytest.fixture
def marker(tmp_path: Path) -> Path:
    """The file that the test restart command writes."""
    return tmp_path / "restarted"


@pytest.fixture
def restart_client(marker: Path) -> Iterator[TestClient]:
    """A client with restart commands: "main" writes the marker file, the others fail."""
    from agent.config import get_settings

    settings = get_settings().model_copy(
        update={
            "speech_restart_commands": {
                # The shell expands $(echo done): the command runs through /bin/sh.
                "main": f"echo $(echo done) > '{marker}'",
                "broken": "echo 'service not loaded' >&2; exit 3",
                "missing": "no-such-restart-program-xyz",
                "slow": "sleep 5",
            }
        }
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with authed_client() as test_client:
            test_client.portal.call(_clean)  # type: ignore[union-attr]
            yield test_client
            test_client.portal.call(_clean)  # type: ignore[union-attr]
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_restart_runs_the_configured_command_and_checks_the_server(
    restart_client: TestClient, marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET" and calls.count("GET") == 1:
            raise httpx.ConnectError("starting")
        return httpx.Response(200, json={"text": ""})

    _servers(monkeypatch, respond)
    monkeypatch.setattr(speech, "RESTART_POLL_SECONDS", 0.01)
    _add(restart_client, "main", 8178)
    _add(restart_client, "spare", 8179)

    profiles = {item["id"]: item for item in restart_client.get(SETTINGS).json()["profiles"]}
    assert profiles["main"]["restart"] is True
    assert profiles["spare"]["restart"] is False
    assert "command" not in profiles["main"]

    result = restart_client.post(f"{SETTINGS}/profiles/main/restart").json()
    assert result["ok"] is True and result["restarted"] is True and result["status"] == 200
    assert marker.read_text().strip() == "done"
    assert calls == ["GET", "GET", "POST"]


def test_restart_without_a_command_or_with_a_failing_command(
    restart_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _servers(monkeypatch, lambda request: httpx.Response(200, json={"text": ""}))
    _add(restart_client, "spare", 8179)
    _add(restart_client, "broken", 8180)
    _add(restart_client, "missing", 8181)
    _add(restart_client, "slow", 8182)
    monkeypatch.setattr(speech, "RESTART_COMMAND_TIMEOUT_SECONDS", 0.2)
    assert restart_client.post(f"{SETTINGS}/profiles/spare/restart").status_code == 404
    assert restart_client.post(f"{SETTINGS}/profiles/none/restart").status_code == 404

    def message(profile_id: str) -> str:
        result = restart_client.post(f"{SETTINGS}/profiles/{profile_id}/restart").json()
        assert result["ok"] is False and result["restarted"] is False
        assert result["status"] is None and result["seconds"] is None
        return str(result["message"])

    assert message("broken") == (
        "Перезапуск не удался — команда завершилась с ошибкой, код 3: service not loaded. "
        "Проверьте AGENT_SPEECH_RESTART_COMMANDS."
    )
    assert message("missing").startswith("Перезапуск не удался — программа из команды не найдена: ")
    assert message("slow") == "Перезапуск не удался — команда не завершилась за 0.2 с."


def test_restart_reports_a_server_that_does_not_come_back(
    restart_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    _servers(monkeypatch, respond)
    monkeypatch.setattr(speech, "RESTART_POLL_SECONDS", 0.01)
    monkeypatch.setattr(speech, "RESTART_READY_TIMEOUT_SECONDS", 0.1)
    _add(restart_client, "main", 8178)
    result = restart_client.post(f"{SETTINGS}/profiles/main/restart").json()
    assert result["ok"] is False and result["restarted"] is True
    assert result["message"] == (
        "Команда выполнена, но сервер не ответил за 0.1 с. Посмотрите журнал сервера распознавания."
    )


def test_an_empty_restart_command_is_refused() -> None:
    from pydantic import ValidationError

    from agent.config import Settings

    with pytest.raises(ValidationError):
        Settings(speech_restart_commands={"main": "  "})
