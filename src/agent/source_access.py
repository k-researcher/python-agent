"""Read fresh, trusted session facts before checking access to a tool source."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.models import Project, Session, User, WorkspaceMembership
from agent.source_policy import SourceActor, SourceScope, can_use_source


async def can_use_session_source(
    db: AsyncSession,
    session_id: str,
    source: SourceScope,
) -> bool:
    """Check the session creator's current membership without writing or flushing.

    Legacy sessions without a creator or project workspace are denied. Scalar
    columns bypass cached ORM identities. This function does not authenticate
    the request or replace tool approval and source-revision checks.
    """
    row = (
        await db.execute(
            select(
                Session.mode,
                Project.id,
                Project.workspace_id,
                User.id,
                User.enabled,
                WorkspaceMembership.role,
                WorkspaceMembership.revoked_at,
            )
            .select_from(Session)
            .join(Project, Project.id == Session.project_id)
            .join(User, User.id == Session.created_by_user_id)
            .join(
                WorkspaceMembership,
                (WorkspaceMembership.user_id == User.id)
                & (WorkspaceMembership.workspace_id == Project.workspace_id),
            )
            .where(Session.id == session_id)
            .execution_options(autoflush=False)
        )
    ).first()
    if row is None:
        return False
    mode, project_id, workspace_id, user_id, enabled, role, revoked_at = row
    actor = SourceActor(
        user_id=user_id,
        workspace_id=workspace_id,
        role=role,
        user_enabled=enabled,
        membership_active=revoked_at is None,
    )
    return can_use_source(
        actor,
        source,
        project_id=project_id,
        project_workspace_id=workspace_id,
        mode=mode,
    )
