"""Personal engine ownership checks, shared by HTTP and dispatch boundaries."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import EngineProfileSettings, Settings
from agent.engine_adapters import configured_profiles
from agent.models import Project, Session


def require_operator(profile: EngineProfileSettings, actor_user_id: str | None) -> None:
    if not actor_user_id or actor_user_id != profile.operator_user_id:
        raise PermissionError("Engine доступен только личному оператору профиля")


def require_personal_project(project: Project) -> None:
    if project.workspace_id is not None:
        raise PermissionError("Engine недоступен для проекта с workspace_id")


async def check_engine_access(
    db: AsyncSession, settings: Settings, session: Session, actor_user_id: str | None
) -> None:
    if session.engine_profile is None:
        return
    project = await db.get(Project, session.project_id)
    if project is None:
        raise PermissionError("Engine project недоступен")
    require_personal_project(project)
    if not actor_user_id or actor_user_id != session.created_by_user_id:
        raise PermissionError("Engine доступен только создателю сессии")
    profile = configured_profiles(settings).get(session.engine_profile)
    if profile is not None:
        require_operator(profile, actor_user_id)
    elif (session.engine_state or {}).get("operator_user_id") != actor_user_id:
        raise PermissionError("Engine operator недоступен")
