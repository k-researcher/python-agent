"""Adversarial checks that source access reads fresh committed facts only.

can_use_session_source must resolve Session.mode, Session.created_by_user_id and
Project.workspace_id at call time even when the caller still holds stale ORM
instances or a stale SourceScope. Source metadata alone is never authority:
the session creator's current membership and the project's current workspace
decide access.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import Project, Session, User, Workspace, WorkspaceMembership
from agent.session_service import create_session_record
from agent.source_access import can_use_session_source
from agent.source_policy import SourceScope


def _scope(
    workspace_id: str,
    project_id: str,
    *,
    enabled: bool = True,
    allowed_modes: frozenset[str] = frozenset({"dev", "ask"}),
) -> SourceScope:
    return SourceScope(
        source_id="adv",
        workspace_id=workspace_id,
        project_ids=frozenset({project_id}),
        enabled=enabled,
        allowed_modes=allowed_modes,
    )


async def _workspace_with_user(
    db: AsyncSession, tmp_path: Path, role: str, *, enabled: bool = True
) -> tuple[Workspace, User, Project]:
    workspace = Workspace(name="Adversarial workspace")
    user = User(display_name="Adversarial user", enabled=enabled)
    db.add_all([workspace, user])
    await db.flush()
    project = Project(
        name="Adversarial project", root_path=str(tmp_path), workspace_id=workspace.id
    )
    db.add(project)
    await db.flush()
    membership = WorkspaceMembership(workspace_id=workspace.id, user_id=user.id, role=role)
    db.add(membership)
    await db.flush()
    return workspace, user, project


async def test_creator_changed_to_other_user_with_membership_viewer_denied(
    tmp_path: Path,
) -> None:
    await init_database()
    async with session_factory() as db:
        workspace, member, project = await _workspace_with_user(db, tmp_path, "member")
        viewer = User(display_name="Other viewer", enabled=True)
        db.add(viewer)
        await db.flush()
        db.add(WorkspaceMembership(workspace_id=workspace.id, user_id=viewer.id, role="viewer"))
        session = Session(
            project_id=project.id,
            created_by_user_id=member.id,
            mode="dev",
            title="Adv session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        session_id = session.id
        viewer_id = viewer.id
        source = _scope(workspace.id, project.id)

        assert await can_use_session_source(db, session_id, source) is True

        async with session_factory() as other:
            await other.execute(
                update(Session).where(Session.id == session_id).values(created_by_user_id=viewer_id)
            )
            await other.commit()
        assert session.created_by_user_id == member.id
        assert await can_use_session_source(db, session_id, source) is False


async def test_mode_read_from_committed_not_stale_orm(tmp_path: Path) -> None:
    await init_database()
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member")
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Adv session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        session_id = session.id
        source = _scope(workspace.id, project.id, allowed_modes=frozenset({"dev"}))

        async with session_factory() as other:
            await other.execute(update(Session).where(Session.id == session_id).values(mode="ask"))
            await other.commit()

        assert await can_use_session_source(db, session_id, source) is False


async def test_creator_cleared_in_second_transaction_denied(tmp_path: Path) -> None:
    await init_database()
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member")
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Adv session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        session_id = session.id
        source = _scope(workspace.id, project.id)

        async with session_factory() as other:
            await other.execute(
                update(Session).where(Session.id == session_id).values(created_by_user_id=None)
            )
            await other.commit()

        assert await can_use_session_source(db, session_id, source) is False


async def test_project_workspace_moved_in_second_transaction_denied(tmp_path: Path) -> None:
    await init_database()
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member")
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Adv session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        session_id = session.id
        source = _scope(workspace.id, project.id)

        async with session_factory() as other:
            await other.execute(
                update(Project).where(Project.id == project.id).values(workspace_id=None)
            )
            await other.commit()

        assert project.workspace_id == workspace.id
        assert await can_use_session_source(db, session_id, source) is False


async def test_create_session_record_actor_allowed_when_matching(tmp_path: Path) -> None:
    await init_database()
    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("System prompt", encoding="utf-8")
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member")
        await db.commit()
        source = _scope(workspace.id, project.id)

        session = await create_session_record(
            db=db,
            settings=settings,
            project=project,
            prompt="Hello",
            mode="dev",
            actor_user_id=user.id,
        )
        assert await can_use_session_source(db, session.id, source) is True


async def test_create_session_record_viewer_actor_denied(tmp_path: Path) -> None:
    await init_database()
    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("System prompt", encoding="utf-8")
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member")
        viewer = User(display_name="Viewer", enabled=True)
        db.add(viewer)
        await db.flush()
        db.add(WorkspaceMembership(workspace_id=workspace.id, user_id=viewer.id, role="viewer"))
        await db.commit()
        source = _scope(workspace.id, project.id)

        session = await create_session_record(
            db=db,
            settings=settings,
            project=project,
            prompt="Hello",
            mode="dev",
            actor_user_id=viewer.id,
        )
        assert await can_use_session_source(db, session.id, source) is False


async def test_disabled_actor_denied_even_with_matching_source(tmp_path: Path) -> None:
    await init_database()
    settings = Settings(prompts_dir=tmp_path)
    (tmp_path / "dev.md").write_text("System prompt", encoding="utf-8")
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member", enabled=False)
        await db.commit()
        source = _scope(workspace.id, project.id)

        session = await create_session_record(
            db=db,
            settings=settings,
            project=project,
            prompt="Hello",
            mode="dev",
            actor_user_id=user.id,
        )
        assert await can_use_session_source(db, session.id, source) is False


async def test_source_metadata_alone_is_never_authority(tmp_path: Path) -> None:
    await init_database()
    async with session_factory() as db:
        workspace, user, project = await _workspace_with_user(db, tmp_path, "member")
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Adv session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        session_id = session.id

        bare_enabled = _scope(workspace.id, project.id)
        assert await can_use_session_source(db, session_id, bare_enabled) is True

        disabled = _scope(workspace.id, project.id, enabled=False)
        assert await can_use_session_source(db, session_id, disabled) is False

        wrong_workspace = SourceScope(
            source_id="adv",
            workspace_id="unknown",
            project_ids=frozenset({project.id}),
            enabled=True,
        )
        assert await can_use_session_source(db, session_id, wrong_workspace) is False
