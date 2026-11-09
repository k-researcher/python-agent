"""Turn recorded speech into text through local recognition servers.

A profile names one OpenAI-compatible transcription server. The first enabled profile
is the main one; when its server does not answer, the next profile is tried. Only
servers on this computer are accepted, so audio does not leave the machine. Audio is
kept in memory for one request and is never stored.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from sqlalchemy import Boolean, DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from agent.models import Base, OutboundAudit, utcnow

# A recording of a few minutes in a compressed browser format is far below this limit.
MAX_AUDIO_BYTES = 25 * 1024 * 1024
TRANSCRIBE_TIMEOUT_SECONDS = 120.0
TEST_TIMEOUT_SECONDS = 30.0
RESTART_COMMAND_TIMEOUT_SECONDS = 30.0
# A large model can need some time to load after a restart.
RESTART_READY_TIMEOUT_SECONDS = 60.0
RESTART_POLL_SECONDS = 0.5
RESTART_ERROR_CHARS = 200
PROFILE_ID = re.compile(r"[A-Za-z0-9._-]{1,100}\Z")
LANGUAGE = re.compile(r"[a-z]{2,3}\Z")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# The browser sends a container type such as "audio/webm;codecs=opus".
_EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav"}


class SpeechProfile(Base):
    """One transcription server; a lower priority is tried first."""

    __tablename__ = "speech_profiles"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(500))
    model: Mapped[str] = mapped_column(String(200), default="")
    # An empty language lets the server detect it.
    language: Mapped[str] = mapped_column(String(10), default="")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class SpeechError(Exception):
    """A transcription failure with a message that is safe to show."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Transcription:
    text: str
    profile: str


def is_local_url(url: str) -> bool:
    """True for an http(s) address on this computer."""
    try:
        parts = urlsplit(url)
        return parts.scheme in {"http", "https"} and parts.hostname in _LOCAL_HOSTS
    except ValueError:
        return False


def transcription_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/audio/transcriptions"


def audio_extension(content_type: str) -> str | None:
    """The file extension for a supported audio type, or None."""
    return _EXTENSIONS.get(content_type.split(";", 1)[0].strip().lower())


async def ordered_profiles(db: AsyncSession) -> list[SpeechProfile]:
    rows = await db.scalars(
        select(SpeechProfile).order_by(SpeechProfile.priority, SpeechProfile.id)
    )
    return list(rows)


def silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A short silent recording that checks the full path to a server."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def client() -> httpx.AsyncClient:
    # Local servers only: no proxy from the environment and no redirects to other hosts.
    return httpx.AsyncClient(follow_redirects=False, trust_env=False)


async def _send(
    db: AsyncSession,
    profile: SpeechProfile,
    audio: bytes,
    extension: str,
    content_type: str,
    language: str,
    timeout: float,
) -> tuple[int | None, str | None]:
    """Send audio to one server. Return the HTTP status (None: no answer) and the text."""
    url = transcription_url(profile.base_url)
    audit = OutboundAudit(
        category="speech",
        destination=url,
        operation="transcribe",
        payload_bytes=len(audio),
        payload_sha256=hashlib.sha256(audio).hexdigest(),
        status="started",
    )
    db.add(audit)
    await db.commit()
    data: dict[str, Any] = {"response_format": "json"}
    if profile.model:
        data["model"] = profile.model
    if language:
        data["language"] = language
    status: int | None = None
    text: str | None = None
    try:
        async with client() as http:
            response = await http.post(
                url,
                files={"file": (f"speech.{extension}", audio, content_type)},
                data=data,
                timeout=timeout,
            )
        status = response.status_code
        if response.is_success:
            body = response.json()
            value = body.get("text") if isinstance(body, dict) else None
            text = value.strip() if isinstance(value, str) else None
        audit.detail = "No text" if response.is_success and text is None else str(status)
    except httpx.TimeoutException:
        audit.detail = "No answer in time"
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, UnicodeError):
        audit.detail = "Server unavailable"
    audit.status = "completed" if text is not None else "error"
    await db.commit()
    return status, text


async def transcribe(
    db: AsyncSession, audio: bytes, content_type: str, language: str | None = None
) -> Transcription:
    """Try the enabled profiles in order; move on only when a server does not answer."""
    extension = audio_extension(content_type)
    if extension is None:
        raise SpeechError("Этот формат записи не поддерживается", status=415)
    if not audio:
        raise SpeechError("Пустая запись", status=422)
    if len(audio) > MAX_AUDIO_BYTES:
        raise SpeechError("Запись слишком длинная", status=413)
    profiles = [item for item in await ordered_profiles(db) if not item.disabled]
    if not profiles:
        raise SpeechError("Распознавание речи не настроено", status=503)
    for profile in profiles:
        if not is_local_url(profile.base_url):
            continue
        status, text = await _send(
            db,
            profile,
            audio,
            extension,
            content_type,
            language if language is not None else profile.language,
            TRANSCRIBE_TIMEOUT_SECONDS,
        )
        if text is not None:
            return Transcription(text=text, profile=profile.id)
        # The server answered but refused this recording: another server will not help.
        if status is not None and 400 <= status < 500:
            raise SpeechError(f"Сервер распознавания отклонил запись: HTTP {status}", status=422)
    raise SpeechError("Серверы распознавания не отвечают", status=503)


async def test_profile(db: AsyncSession, profile: SpeechProfile) -> dict[str, Any]:
    """Send one second of silence; an answer with a text field means the server works."""
    if not is_local_url(profile.base_url):
        return {"ok": False, "status": None, "seconds": None, "message": "not local"}
    started = time.monotonic()
    status, text = await _send(
        db, profile, silence_wav(), "wav", "audio/wav", profile.language, TEST_TIMEOUT_SECONDS
    )
    seconds = round(time.monotonic() - started, 2)
    return {"ok": text is not None, "status": status, "seconds": seconds if status else None}


# One restart at a time: a second click must not kill a server that is still starting.
_restart_lock = asyncio.Lock()


def _first_line(output: bytes) -> str:
    """The first non-empty line of the command output, short enough for one status line."""
    for line in output.decode(errors="replace").splitlines():
        if line.strip():
            text = line.strip()
            return text if len(text) <= RESTART_ERROR_CHARS else text[:RESTART_ERROR_CHARS] + "…"
    return ""


def _restart_failure(code: int, error: str) -> str:
    """A message that says what went wrong and what to look at."""
    if code == 127:
        reason = "программа из команды не найдена"
    elif code == 126:
        reason = "нет прав на запуск программы из команды"
    else:
        reason = f"команда завершилась с ошибкой, код {code}"
    detail = f": {error}" if error else ""
    return f"Перезапуск не удался — {reason}{detail}. Проверьте AGENT_SPEECH_RESTART_COMMANDS."


async def _run_restart_command(command: str) -> str | None:
    """Run the command with /bin/sh. Return a message for a failure, or None."""
    try:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return "Перезапуск не удался — не запускается /bin/sh."
    try:
        _, error = await asyncio.wait_for(process.communicate(), RESTART_COMMAND_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()
        seconds = f"{RESTART_COMMAND_TIMEOUT_SECONDS:g}"
        return f"Перезапуск не удался — команда не завершилась за {seconds} с."
    code = process.returncode
    return None if code == 0 else _restart_failure(code or 0, _first_line(error))


async def _wait_until_answers(base_url: str) -> bool:
    """True when the server gives any HTTP answer before the time limit."""
    deadline = time.monotonic() + RESTART_READY_TIMEOUT_SECONDS
    async with client() as http:
        while time.monotonic() < deadline:
            try:
                await http.get(base_url, timeout=RESTART_POLL_SECONDS * 4)
                return True
            except (httpx.HTTPError, httpx.InvalidURL):
                await asyncio.sleep(RESTART_POLL_SECONDS)
    return False


async def restart_profile(db: AsyncSession, profile: SpeechProfile, command: str) -> dict[str, Any]:
    """Run the restart command from the configuration, wait for the server, then check it."""
    if _restart_lock.locked():
        raise SpeechError("Перезапуск уже идёт", status=409)
    async with _restart_lock:
        failure = await _run_restart_command(command)
        if failure is not None:
            return {
                "ok": False,
                "status": None,
                "seconds": None,
                "restarted": False,
                "message": failure,
            }
        if not await _wait_until_answers(profile.base_url):
            return {
                "ok": False,
                "status": None,
                "seconds": None,
                "restarted": True,
                "message": (
                    f"Команда выполнена, но сервер не ответил за "
                    f"{RESTART_READY_TIMEOUT_SECONDS:g} с. Посмотрите журнал сервера распознавания."
                ),
            }
        return {**await test_profile(db, profile), "restarted": True}
