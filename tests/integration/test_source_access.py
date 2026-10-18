"""Verify source access against real database rows, revocations, and ORM caches."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select, update

from agent.database import init_database, session_factory
from agent.models import Message, Project, Session, User, Workspace, WorkspaceMembership
from agent.source_access import can_use_session_source
from agent.source_policy import SourceScope


@dataclass(frozen=True)
class AccessCase:
    session: Session
    user: User
    membership: WorkspaceMembership
    source: SourceScope


@pytest.fixture
async def access_case(tmp_path: Path) -> AccessCase:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="Source workspace")
        user = User(display_name="Source user", enabled=True)
        db.add_all([workspace, user])
        await db.flush()
        membership = WorkspaceMembership(workspace_id=workspace.id, user_id=user.id, role="member")
        project = Project(name="Source project", root_path=str(tmp_path), workspace_id=workspace.id)
        db.add_all([membership, project])
        await db.flush()
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Source session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        return AccessCase(
            session,
            user,
            membership,
            SourceScope(
                source_id="source",
                workspace_id=workspace.id,
                project_ids=frozenset({project.id}),
                allowed_modes=frozenset({"dev"}),
            ),
        )


@pytest.mark.parametrize(
    "role,allowed",
    [
        ("owner", True),
        ("admin", True),
        ("member", True),
        ("viewer", False),
    ],
)
async def test_roles(access_case: AccessCase, role: str, allowed: bool) -> None:
    async with session_factory() as db:
        await db.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == access_case.source.workspace_id,
                WorkspaceMembership.user_id == access_case.user.id,
            )
            .values(role=role)
        )
        await db.commit()
        assert (
            await can_use_session_source(db, access_case.session.id, access_case.source) is allowed
        )


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "revoked",
        "missing-membership",
        "missing-creator",
        "missing-workspace",
        "wrong-user",
        "wrong-workspace",
        "ask",
        "source-disabled",
        "source-workspace",
        "source-project",
    ],
)
async def test_denied_facts(access_case: AccessCase, change: str) -> None:
    case = access_case
    source = case.source
    async with session_factory() as db:
        membership = await db.get(WorkspaceMembership, (case.source.workspace_id, case.user.id))
        session = await db.get(Session, case.session.id)
        assert membership is not None and session is not None
        if change == "disabled":
            await db.execute(update(User).where(User.id == case.user.id).values(enabled=False))
        elif change == "revoked":
            membership.revoked_at = datetime.now(UTC)
        elif change == "missing-membership":
            await db.delete(membership)
        elif change == "missing-creator":
            session.created_by_user_id = None
        elif change == "missing-workspace":
            await db.execute(
                update(Project)
                .where(Project.id == session.project_id)
                .values(
                    workspace_id=None,
                )
            )
        elif change == "wrong-user":
            other_user = User(display_name="Other")
            db.add(other_user)
            await db.flush()
            membership.user_id = other_user.id
        elif change == "wrong-workspace":
            other_workspace = Workspace(name="Other")
            db.add(other_workspace)
            await db.flush()
            membership.workspace_id = other_workspace.id
        elif change == "ask":
            session.mode = "ask"
        elif change == "source-disabled":
            source = replace(source, enabled=False)
        elif change == "source-workspace":
            source = replace(source, workspace_id="other")
        elif change == "source-project":
            source = replace(source, project_ids=frozenset({"other"}))
        await db.commit()
        before = await db.execute(
            select(Session.run_generation, Session.status).where(
                Session.id == session.id,
            )
        )
        expected_state = before.one()
        counts = tuple(
            [
                await db.scalar(select(func.count()).select_from(model))
                for model in (Session, Message)
            ]
        )
        assert await can_use_session_source(db, session.id, source) is False
        assert (
            await db.execute(
                select(Session.run_generation, Session.status).where(
                    Session.id == session.id,
                )
            )
        ).one() == expected_state
        assert (
            tuple(
                [
                    await db.scalar(select(func.count()).select_from(model))
                    for model in (Session, Message)
                ]
            )
            == counts
        )


@pytest.mark.parametrize("change", ["revoked", "disabled", "demoted"])
async def test_revoke_after_caching(access_case: AccessCase, change: str) -> None:
    case = access_case
    async with session_factory() as db:
        user = await db.get(User, case.user.id)
        membership = await db.get(WorkspaceMembership, (case.source.workspace_id, case.user.id))
        assert user is not None and membership is not None
        assert await can_use_session_source(db, case.session.id, case.source) is True
        async with session_factory() as other:
            if change == "disabled":
                await other.execute(update(User).where(User.id == user.id).values(enabled=False))
            else:
                values = (
                    {"role": "viewer"}
                    if change == "demoted"
                    else {
                        "revoked_at": datetime.now(UTC),
                    }
                )
                await other.execute(
                    update(WorkspaceMembership)
                    .where(
                        WorkspaceMembership.workspace_id == case.source.workspace_id,
                        WorkspaceMembership.user_id == user.id,
                    )
                    .values(**values)
                )
            await other.commit()
        assert user.enabled is True
        assert membership.role == "member" and membership.revoked_at is None
        assert await can_use_session_source(db, case.session.id, case.source) is False


async def test_no_autoflush_and_unknown_session(access_case: AccessCase) -> None:
    case = access_case
    async with session_factory() as db:
        session = await db.get(Session, case.session.id)
        assert session is not None
        title = session.title
        session.title = "Unflushed change"
        assert await can_use_session_source(db, session.id, case.source) is True
        assert (
            await db.scalar(
                select(Session.title)
                .where(Session.id == session.id)
                .execution_options(autoflush=False)
            )
            == title
        )
        assert db.is_modified(session)
        assert await can_use_session_source(db, "missing", case.source) is False
