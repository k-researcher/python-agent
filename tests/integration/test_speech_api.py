"""Speech recognition: local profiles, fallback to the next server and safe limits."""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
