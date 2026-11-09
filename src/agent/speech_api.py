"""Speech recognition settings and the transcription endpoint."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import speech
from agent.config import Settings, get_settings
from agent.database import get_db

settings_router = APIRouter(prefix="/api/settings/speech", tags=["speech settings"])
router = APIRouter(prefix="/api/speech", tags=["speech"])
Database = Annotated[AsyncSession, Depends(get_db)]
Configuration = Annotated[Settings, Depends(get_settings)]


class ProfilePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    language: str | None = Field(default=None, max_length=10)
    disabled: bool | None = None


def _view(profile: speech.SpeechProfile, settings: Settings) -> dict[str, Any]:
    return {
        "id": profile.id,
        "base_url": profile.base_url,
        "model": profile.model,
        "language": profile.language,
        "disabled": profile.disabled,
        # The command itself stays on the server; the UI only learns that it exists.
        "restart": profile.id in settings.speech_restart_commands,
    }


async def _settings(db: AsyncSession, settings: Settings) -> dict[str, Any]:
    return {"profiles": [_view(item, settings) for item in await speech.ordered_profiles(db)]}


async def _profile(db: AsyncSession, profile_id: str) -> speech.SpeechProfile:
    profile = await db.get(speech.SpeechProfile, profile_id)
    if profile is None:
        raise HTTPException(404, "Speech profile not found")
    return profile


@settings_router.get("")
async def get_speech_settings(db: Database, settings: Configuration) -> dict[str, Any]:
    return await _settings(db, settings)


@settings_router.put("/profiles/{profile_id}")
async def put_profile(
    profile_id: str, patch: ProfilePatch, db: Database, settings: Configuration
) -> dict[str, Any]:
    if speech.PROFILE_ID.fullmatch(profile_id) is None:
        raise HTTPException(422, "Use letters, digits, dot, dash or underscore in the ID")
    if patch.base_url is not None and not speech.is_local_url(patch.base_url):
        raise HTTPException(422, "Only a server on this computer is allowed")
    if patch.language and speech.LANGUAGE.fullmatch(patch.language) is None:
        raise HTTPException(422, "Use a language code such as ru or en")
    profile = await db.get(speech.SpeechProfile, profile_id)
    if profile is None:
        if patch.base_url is None:
            raise HTTPException(422, "A new profile needs a server address")
        last = await db.scalar(select(func.max(speech.SpeechProfile.priority)))
        profile = speech.SpeechProfile(
            id=profile_id, base_url=patch.base_url, priority=(last or 0) + 1
        )
        db.add(profile)
    for name, value in patch.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(profile, name, value)
    await db.commit()
    return await _settings(db, settings)


@settings_router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    await db.delete(await _profile(db, profile_id))
    await db.commit()
    return await _settings(db, settings)


@settings_router.post("/profiles/{profile_id}/primary")
async def make_primary(profile_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    """Put the profile first and keep the order of the others."""
    chosen = await _profile(db, profile_id)
    others = [item for item in await speech.ordered_profiles(db) if item.id != chosen.id]
    for index, item in enumerate([chosen, *others]):
        item.priority = index
    await db.commit()
    return await _settings(db, settings)


@settings_router.post("/profiles/{profile_id}/test")
async def test_profile(profile_id: str, db: Database) -> dict[str, Any]:
    return await speech.test_profile(db, await _profile(db, profile_id))


@settings_router.post("/profiles/{profile_id}/restart")
async def restart_profile(profile_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    """Run the restart command from the server configuration, then check the server."""
    profile = await _profile(db, profile_id)
    command = settings.speech_restart_commands.get(profile.id)
    if command is None:
        raise HTTPException(404, "No restart command for this profile")
    try:
        return await speech.restart_profile(db, profile, command)
    except speech.SpeechError as error:
        raise HTTPException(error.status, str(error)) from None


@router.post("/transcribe")
async def transcribe(request: Request, db: Database, language: str | None = None) -> dict[str, str]:
    """Take the raw recording as the request body; the audio is not stored."""
    if language is not None and speech.LANGUAGE.fullmatch(language) is None:
        raise HTTPException(422, "Use a language code such as ru or en")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > speech.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Запись слишком длинная")
    audio = bytearray()
    async for chunk in request.stream():
        audio.extend(chunk)
        if len(audio) > speech.MAX_AUDIO_BYTES:
            raise HTTPException(413, "Запись слишком длинная")
    try:
        result = await speech.transcribe(
            db, bytes(audio), request.headers.get("content-type", ""), language
        )
    except speech.SpeechError as error:
        raise HTTPException(error.status, str(error)) from None
    return {"text": result.text, "profile": result.profile}
