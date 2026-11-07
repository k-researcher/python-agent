"""Shared and project skills: list, read, create, change, import, delete and switch."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent import skill_library
from agent.config import Settings, get_settings
from agent.database import get_db
from agent.events import emit_event
from agent.models import Project
from agent.skill_library import Scope, SkillCatalog, SkillConflictError, SkillState
from agent.skill_prompt import project_skills_dir
from agent.skills import MAX_SKILL_BYTES, SkillError, validate_name

router = APIRouter(prefix="/api/skills", tags=["skills"])
Database = Annotated[AsyncSession, Depends(get_db)]
Configuration = Annotated[Settings, Depends(get_settings)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class SkillCreate(_Body):
    scope: Literal["shared", "project"]
    project_id: str | None = None
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    instructions: str = Field(max_length=MAX_SKILL_BYTES)


class SkillUpdate(_Body):
    project_id: str | None = None
    description: str = Field(min_length=1, max_length=1024)
    instructions: str = Field(max_length=MAX_SKILL_BYTES)
    revision: str = Field(min_length=64, max_length=64)


class SkillSwitch(_Body):
    # Without a project the switch is the default of a shared skill for all projects.
    project_id: str | None = None
    # None removes the own choice, so the default applies again.
    enabled: bool | None


def _view(state: SkillState) -> dict[str, Any]:
    skill = state.skill
    return {
        "scope": state.scope,
        "name": skill.name,
        "description": skill.description,
        "version": skill.version,
        "enabled": state.enabled,
        "default_enabled": state.default_enabled,
        "project_choice": state.project_choice,
        "replaced": state.replaced,
        "files": len(skill.files),
        "path": str(skill.root),
    }


def _catalog_view(
    settings: Settings, project: Project | None, catalog: SkillCatalog
) -> dict[str, Any]:
    return {
        "shared_dir": str(settings.shared_skills_dir),
        "project_dir": (str(project_skills_dir(Path(project.root_path))) if project else None),
        "skills": [_view(state) for state in catalog.skills],
        "errors": [{"scope": scope, "message": message} for scope, message in catalog.errors],
    }


async def _project(db: AsyncSession, project_id: str | None) -> Project | None:
    if project_id is None:
        return None
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _scope(value: str) -> Scope:
    if value == "shared":
        return "shared"
    if value == "project":
        return "project"
    raise HTTPException(404, "Unknown skill scope")


async def _state(
    db: AsyncSession, settings: Settings, scope: str, name: str, project_id: str | None
) -> tuple[Project | None, SkillState]:
    kind = _scope(scope)
    project = await _project(db, project_id)
    if kind == "project" and project is None:
        raise HTTPException(422, "Для навыка проекта нужен проект")
    catalog = await skill_library.load_catalog(db, settings, project)
    try:
        return project, catalog.find(kind, name)
    except SkillError:
        raise HTTPException(404, "Навык не найден") from None


def _refuse(error: SkillError) -> HTTPException:
    status = 409 if isinstance(error, SkillConflictError) else 422
    return HTTPException(status, str(error))


async def _changed(
    db: AsyncSession, settings: Settings, project: Project | None, event: dict[str, Any]
) -> dict[str, Any]:
    # The event commits the switch changes of the same request.
    await emit_event(db, "skills.changed", event)
    return _catalog_view(settings, project, await skill_library.load_catalog(db, settings, project))


@router.get("")
async def list_skills(
    db: Database, settings: Configuration, project_id: str | None = None
) -> dict[str, Any]:
    project = await _project(db, project_id)
    return _catalog_view(settings, project, await skill_library.load_catalog(db, settings, project))


@router.get("/{scope}/{name}")
async def read_skill(
    scope: str, name: str, db: Database, settings: Configuration, project_id: str | None = None
) -> dict[str, Any]:
    _project_row, state = await _state(db, settings, scope, name, project_id)
    text = skill_library.skill_text(state.skill)
    return {
        **_view(state),
        "instructions": state.skill.body,
        "file_list": list(state.skill.files),
        "revision": skill_library.revision(text),
    }


@router.post("", status_code=201)
async def create_skill(
    payload: SkillCreate, db: Database, settings: Configuration
) -> dict[str, Any]:
    project = await _project(db, payload.project_id)
    catalog = await skill_library.load_catalog(db, settings, project)
    try:
        skill = skill_library.create_skill(
            settings,
            catalog,
            payload.scope,
            project,
            validate_name(payload.name),
            payload.description,
            payload.instructions,
        )
    except SkillError as error:
        raise _refuse(error) from None
    # A switch left by an old skill with this name must not decide for the new one.
    await skill_library.forget_switches(db, payload.scope, skill.name, project)
    event = {"action": "created", "scope": payload.scope, "name": skill.name}
    return await _changed(db, settings, project, event)


@router.post("/import", status_code=201)
async def import_skill(
    request: Request,
    db: Database,
    settings: Configuration,
    scope: Literal["shared", "project"],
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    project_id: str | None = None,
) -> dict[str, Any]:
    """Take a SKILL.md file or a zip archive of a skill directory as the request body."""
    project = await _project(db, project_id)
    limit = skill_library.MAX_PACKAGE_BYTES
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"Файл больше {limit // (1024 * 1024)} МБ")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, f"Файл больше {limit // (1024 * 1024)} МБ")
    catalog = await skill_library.load_catalog(db, settings, project)
    try:
        skill = await asyncio.to_thread(
            skill_library.import_skill, settings, catalog, scope, project, filename, bytes(data)
        )
    except SkillError as error:
        raise _refuse(error) from None
    await skill_library.forget_switches(db, scope, skill.name, project)
    event = {"action": "imported", "scope": scope, "name": skill.name}
    return await _changed(db, settings, project, event)


@router.put("/{scope}/{name}")
async def update_skill(
    scope: str, name: str, payload: SkillUpdate, db: Database, settings: Configuration
) -> dict[str, Any]:
    project, state = await _state(db, settings, scope, name, payload.project_id)
    try:
        skill_library.update_skill(
            settings, state, project, payload.description, payload.instructions, payload.revision
        )
    except SkillError as error:
        raise _refuse(error) from None
    event = {"action": "updated", "scope": state.scope, "name": name}
    return await _changed(db, settings, project, event)


@router.delete("/{scope}/{name}")
async def delete_skill(
    scope: str,
    name: str,
    db: Database,
    settings: Configuration,
    revision: Annotated[str, Query(min_length=64, max_length=64)],
    project_id: str | None = None,
) -> dict[str, Any]:
    project, state = await _state(db, settings, scope, name, project_id)
    try:
        await asyncio.to_thread(skill_library.delete_skill, settings, state, project, revision)
    except SkillError as error:
        raise _refuse(error) from None
    await skill_library.forget_switches(db, state.scope, name, project)
    event = {"action": "deleted", "scope": state.scope, "name": name}
    return await _changed(db, settings, project, event)


@router.put("/{scope}/{name}/switch")
async def switch_skill(
    scope: str, name: str, payload: SkillSwitch, db: Database, settings: Configuration
) -> dict[str, Any]:
    project, state = await _state(db, settings, scope, name, payload.project_id)
    try:
        await skill_library.set_switch(db, state.scope, name, project, payload.enabled)
    except SkillError as error:
        raise _refuse(error) from None
    event = {
        "action": "switched",
        "scope": state.scope,
        "name": name,
        "project_id": project.id if project else None,
        "enabled": payload.enabled,
    }
    return await _changed(db, settings, project, event)
