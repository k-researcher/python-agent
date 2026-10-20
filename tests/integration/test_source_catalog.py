"""Integration tests for selecting session-allowed tool sources."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import select

from agent.database import init_database, session_factory
from agent.models import Project, Session, User, Workspace, WorkspaceMembership
from agent.source_catalog import select_session_sources
from agent.tool_sources import ToolSource, ToolSourceError


@dataclass(frozen=True)
class CatalogCase:
    session: Session
    user: User
    project: Project
    workspace: Workspace


class _RefusingSession:
    async def execute(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("No SQL expected")


@pytest.fixture
async def catalog_case(tmp_path: Path) -> CatalogCase:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="Catalog workspace")
        other_workspace = Workspace(name="Other workspace")
        user = User(display_name="Catalog user", enabled=True)
        db.add_all([workspace, other_workspace, user])
        await db.flush()
        membership = WorkspaceMembership(workspace_id=workspace.id, user_id=user.id, role="member")
        db.add(membership)
        await db.flush()
        project = Project(
            name="Catalog project", root_path=str(tmp_path), workspace_id=workspace.id
        )
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Catalog session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        yield CatalogCase(session, user, project, workspace)


def _source(
    catalog_case: CatalogCase,
    *,
    source_id: str = "server-1",
    workspace_id: str | None = None,
    project_ids: set[str] | None = None,
    enabled: bool = True,
) -> ToolSource:
    return ToolSource.model_validate(
        {
            "source_id": source_id,
            "workspace_id": workspace_id or catalog_case.workspace.id,
            "project_ids": list(project_ids)
            if project_ids is not None
            else [catalog_case.project.id],
            "command": ["server"],
            "enabled": enabled,
        }
    )


def _bare_source(source_id: str, workspace_id: str) -> ToolSource:
    return ToolSource.model_validate(
        {
            "source_id": source_id,
            "workspace_id": workspace_id,
            "project_ids": [],
            "command": ["server"],
            "enabled": True,
        }
    )


async def test_matching_explicit_actor_and_member_allowed(catalog_case: CatalogCase) -> None:
    source = _source(catalog_case)
    async with session_factory() as db:
        result = await select_session_sources(db, catalog_case.session.id, (source,))
    assert result == (source,)


async def test_mixed_workspaces_with_same_source_id_do_not_bleed(
    catalog_case: CatalogCase,
) -> None:
    matching = _source(catalog_case, source_id="server-x")
    foreign = _source(
        catalog_case,
        source_id="server-x",
        workspace_id="00000000-0000-0000-0000-00000000ffff",
    )
    async with session_factory() as db:
        result = await select_session_sources(db, catalog_case.session.id, (matching, foreign))
    assert result == (matching,)


async def test_disabled_source_excluded(catalog_case: CatalogCase) -> None:
    disabled = _source(catalog_case, enabled=False)
    async with session_factory() as db:
        result = await select_session_sources(db, catalog_case.session.id, (disabled,))
    assert result == ()


async def test_wrong_project_source_excluded(catalog_case: CatalogCase) -> None:
    wrong_project = _source(
        catalog_case,
        source_id="server-wrong-project",
        project_ids={"00000000-0000-0000-0000-0000000000aa"},
    )
    matching = _source(catalog_case)
    async with session_factory() as db:
        result = await select_session_sources(
            db, catalog_case.session.id, (wrong_project, matching)
        )
    assert result == (matching,)


async def test_empty_project_allowlist_allows_any_project(catalog_case: CatalogCase) -> None:
    any_project = _source(catalog_case, project_ids=set())
    async with session_factory() as db:
        result = await select_session_sources(db, catalog_case.session.id, (any_project,))
    assert result == (any_project,)


async def test_null_creator_denies(catalog_case: CatalogCase) -> None:
    source = _source(catalog_case)
    async with session_factory() as db:
        session = await db.get(Session, catalog_case.session.id)
        assert session is not None
        session.created_by_user_id = None
        await db.commit()
        result = await select_session_sources(db, session.id, (source,))
    assert result == ()


async def test_unknown_session_denies(catalog_case: CatalogCase) -> None:
    source = _source(catalog_case)
    async with session_factory() as db:
        result = await select_session_sources(db, "missing-session", (source,))
    assert result == ()


async def test_duplicates_fail_before_db_execute(catalog_case: CatalogCase) -> None:
    duplicate = _source(catalog_case, source_id="server-dup")
    with pytest.raises(ToolSourceError):
        await select_session_sources(_RefusingSession(), "any-session", (duplicate, duplicate))


async def test_duplicates_even_when_disabled(catalog_case: CatalogCase) -> None:
    first = _source(catalog_case, source_id="server-dup", enabled=False)
    second = _source(catalog_case, source_id="server-dup")
    with pytest.raises(ToolSourceError):
        await select_session_sources(_RefusingSession(), "any-session", (first, second))


async def test_disabled_duplicate_before_db_execute() -> None:
    first = _bare_source("server-dup", "ws")
    second = _bare_source("server-dup", "ws")
    with pytest.raises(ToolSourceError):
        await select_session_sources(_RefusingSession(), "any-session", (first, second))


async def test_empty_input_does_no_sql() -> None:
    result = await select_session_sources(_RefusingSession(), "any-session", ())
    assert result == ()


async def test_dirty_orm_data_not_flushed(catalog_case: CatalogCase) -> None:
    source = _source(catalog_case)
    async with session_factory() as db:
        session = await db.get(Session, catalog_case.session.id)
        assert session is not None
        original_title = session.title
        session.title = "Unflushed title"
        result = await select_session_sources(db, session.id, (source,))
        assert result == (source,)
        stored_title = await db.scalar(
            select(Session.title).where(Session.id == session.id).execution_options(autoflush=False)
        )
        assert stored_title == original_title
        assert db.is_modified(session)


async def test_output_preserves_original_order(catalog_case: CatalogCase) -> None:
    first = _source(catalog_case, source_id="server-a")
    second = _source(catalog_case, source_id="server-b")
    denied = _source(catalog_case, source_id="server-c", enabled=False)
    async with session_factory() as db:
        result = await select_session_sources(db, catalog_case.session.id, (first, denied, second))
    assert result == (first, second)
