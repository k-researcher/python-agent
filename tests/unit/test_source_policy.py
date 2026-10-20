from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agent.source_policy import SourceActor, SourceScope, can_manage_source, can_use_source


@pytest.fixture
def actor() -> SourceActor:
    return SourceActor("user", "workspace", "admin")


@pytest.fixture
def source() -> SourceScope:
    return SourceScope("source", "workspace", project_ids=frozenset({"project"}))


def decisions(actor: SourceActor, source: SourceScope) -> tuple[bool, bool]:
    return can_manage_source(actor, source), can_use_source(
        actor, source, project_id="project", project_workspace_id="workspace", mode="dev"
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("owner", (True, True)),
        ("admin", (True, True)),
        ("member", (False, True)),
        ("viewer", (False, False)),
        ("superadmin", (False, False)),
        ("", (False, False)),
        (None, (False, False)),
    ],
)
def test_role_permissions(
    actor: SourceActor,
    source: SourceScope,
    role: str | None,
    expected: tuple[bool, bool],
) -> None:
    assert decisions(replace(actor, role=role), source) == expected


@pytest.mark.parametrize("field", ["user_enabled", "membership_active"])
def test_inactive_actor_is_denied(actor: SourceActor, source: SourceScope, field: str) -> None:
    assert decisions(replace(actor, **{field: False}), source) == (False, False)


@pytest.mark.parametrize("value", ["", " ", "\t"])
@pytest.mark.parametrize("field", ["user_id", "workspace_id"])
def test_actor_ids_must_be_present(
    actor: SourceActor, source: SourceScope, field: str, value: str
) -> None:
    assert decisions(replace(actor, **{field: value}), source) == (False, False)


@pytest.mark.parametrize("value", ["", " ", "\t"])
@pytest.mark.parametrize("field", ["source_id", "workspace_id"])
def test_source_ids_must_be_present(
    actor: SourceActor, source: SourceScope, field: str, value: str
) -> None:
    assert decisions(actor, replace(source, **{field: value})) == (False, False)


@pytest.mark.parametrize("workspace_id", ["other", " workspace", "workspace "])
def test_same_source_name_cannot_cross_workspaces(
    actor: SourceActor, source: SourceScope, workspace_id: str
) -> None:
    assert decisions(actor, replace(source, workspace_id=workspace_id)) == (False, False)
    assert decisions(replace(actor, workspace_id=workspace_id), source) == (False, False)


def test_disabled_source_can_be_managed_but_not_used(
    actor: SourceActor, source: SourceScope
) -> None:
    assert decisions(actor, replace(source, enabled=False)) == (True, False)


@pytest.mark.parametrize("project_ids", [frozenset(), frozenset({"project"})])
@pytest.mark.parametrize("workspace_id", [None, "other", " workspace", "workspace "])
def test_project_workspace_is_required_even_without_a_project_allowlist(
    actor: SourceActor,
    source: SourceScope,
    project_ids: frozenset[str],
    workspace_id: str | None,
) -> None:
    assert not can_use_source(
        actor,
        replace(source, project_ids=project_ids),
        project_id="project",
        project_workspace_id=workspace_id,
        mode="dev",
    )


@pytest.mark.parametrize("project_ids", [frozenset(), frozenset({"project"})])
@pytest.mark.parametrize("project_id", ["", " ", "\t"])
def test_empty_project_id_is_denied(
    actor: SourceActor, source: SourceScope, project_ids: frozenset[str], project_id: str
) -> None:
    assert not can_use_source(
        actor,
        replace(source, project_ids=project_ids),
        project_id=project_id,
        project_workspace_id="workspace",
        mode="dev",
    )


@pytest.mark.parametrize("project_id", ["project", "other", " project", "project "])
@pytest.mark.parametrize("restricted", [True, False])
def test_project_allowlist_uses_exact_ids(
    actor: SourceActor, source: SourceScope, project_id: str, restricted: bool
) -> None:
    scoped = source if restricted else replace(source, project_ids=frozenset())
    expected = not restricted or project_id == "project"
    assert (
        can_use_source(
            actor, scoped, project_id=project_id, project_workspace_id="workspace", mode="dev"
        )
        is expected
    )


@pytest.mark.parametrize("mode", ["dev", "ask", "other", "", " dev"])
@pytest.mark.parametrize(
    "allowed_modes",
    [frozenset(), frozenset({"dev"}), frozenset({"ask"}), frozenset({"dev", "ask", "other"})],
)
def test_mode_must_be_known_and_allowed(
    actor: SourceActor, source: SourceScope, mode: str, allowed_modes: frozenset[str]
) -> None:
    assert can_use_source(
        actor,
        replace(source, allowed_modes=allowed_modes),
        project_id="project",
        project_workspace_id="workspace",
        mode=mode,
    ) is (mode in {"dev", "ask"} and mode in allowed_modes)


def test_default_scope_and_facts_are_immutable(actor: SourceActor) -> None:
    source = SourceScope("source", "workspace")
    assert source.project_ids == frozenset()
    assert source.allowed_modes == frozenset({"dev", "ask"})
    assert decisions(actor, source) == (True, True)
    for target, field in ((actor, "role"), (source, "source_id")):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, "other")
        assert not hasattr(target, "__dict__")
