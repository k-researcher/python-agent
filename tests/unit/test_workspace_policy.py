"""Deterministic tests for the pure workspace role policy module."""

from __future__ import annotations

from inspect import signature

import pytest

from agent.workspace_policy import (
    WorkspaceCapabilities,
    WorkspaceRole,
    can_assign_role,
    can_control_session,
    workspace_capabilities,
)


def _assert_capabilities(
    caps: WorkspaceCapabilities,
    *,
    view_project: bool,
    run_agent: bool,
    manage_members: bool,
    manage_workspace: bool,
    manage_models: bool,
) -> None:
    assert caps.view_project is view_project
    assert caps.run_agent is run_agent
    assert caps.manage_members is manage_members
    assert caps.manage_workspace is manage_workspace
    assert caps.manage_models is manage_models


class TestWorkspaceCapabilitiesMatrix:
    def test_owner_capabilities(self) -> None:
        _assert_capabilities(
            workspace_capabilities(WorkspaceRole.OWNER.value),
            view_project=True,
            run_agent=True,
            manage_members=True,
            manage_workspace=True,
            manage_models=False,
        )

    def test_admin_capabilities(self) -> None:
        _assert_capabilities(
            workspace_capabilities(WorkspaceRole.ADMIN.value),
            view_project=True,
            run_agent=True,
            manage_members=True,
            manage_workspace=True,
            manage_models=False,
        )

    def test_member_capabilities(self) -> None:
        _assert_capabilities(
            workspace_capabilities(WorkspaceRole.MEMBER.value),
            view_project=True,
            run_agent=True,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )

    def test_viewer_capabilities(self) -> None:
        _assert_capabilities(
            workspace_capabilities(WorkspaceRole.VIEWER.value),
            view_project=True,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )

    def test_owner_enum_accepted(self) -> None:
        caps = workspace_capabilities(WorkspaceRole.OWNER)
        _assert_capabilities(
            caps,
            view_project=True,
            run_agent=True,
            manage_members=True,
            manage_workspace=True,
            manage_models=False,
        )


class TestUnknownNoneEmptyRoleCapabilities:
    def test_none_role_denies_all(self) -> None:
        _assert_capabilities(
            workspace_capabilities(None),
            view_project=False,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )

    def test_empty_role_denies_all(self) -> None:
        _assert_capabilities(
            workspace_capabilities(""),
            view_project=False,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )

    def test_unknown_role_denies_all(self) -> None:
        _assert_capabilities(
            workspace_capabilities("superadmin"),
            view_project=False,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )

    def test_unknown_role_as_enum_value_denies_all(self) -> None:
        _assert_capabilities(
            workspace_capabilities("unknown-role"),
            view_project=False,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )


class TestDisabledMembershipCapabilities:
    @pytest.mark.parametrize("role", [role.value for role in WorkspaceRole])
    def test_inactive_membership_denies_all_roles(self, role: str) -> None:
        _assert_capabilities(
            workspace_capabilities(role, membership_active=False),
            view_project=False,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )

    def test_inactive_membership_with_none_role_denies_all(self) -> None:
        _assert_capabilities(
            workspace_capabilities(None, membership_active=False),
            view_project=False,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )


class TestCanControlSession:
    @pytest.mark.parametrize("role", [WorkspaceRole.OWNER.value, WorkspaceRole.ADMIN.value])
    def test_owner_admin_control_any_session(self, role: str) -> None:
        assert can_control_session(role, "actor-1", "some-creator") is True
        assert can_control_session(role, "actor-1", None) is True
        assert can_control_session(role, "actor-1", "") is True
        assert can_control_session(role, "actor-1", "actor-1") is True

    @pytest.mark.parametrize("creator", ["other-creator", None, "", "actor-1"])
    def test_member_control_only_own_session(self, creator: str | None) -> None:
        expected = creator is not None and creator == "actor-1"
        assert can_control_session(WorkspaceRole.MEMBER.value, "actor-1", creator) is expected

    @pytest.mark.parametrize("creator", ["other-creator", None, "", "actor-1"])
    def test_viewer_never_controls(self, creator: str | None) -> None:
        assert can_control_session(WorkspaceRole.VIEWER.value, "actor-1", creator) is False

    def test_empty_actor_denied_for_all_roles(self) -> None:
        assert can_control_session(WorkspaceRole.OWNER.value, "", "creator") is False
        assert can_control_session(WorkspaceRole.ADMIN.value, "", "creator") is False
        assert can_control_session(WorkspaceRole.MEMBER.value, "", "actor-1") is False
        assert can_control_session(WorkspaceRole.VIEWER.value, "", "creator") is False

    def test_none_actor_denied(self) -> None:
        assert can_control_session(WorkspaceRole.OWNER.value, None, "creator") is False

    def test_unknown_role_denied(self) -> None:
        assert can_control_session("superadmin", "actor-1", "creator") is False

    def test_none_role_denied(self) -> None:
        assert can_control_session(None, "actor-1", "creator") is False

    def test_empty_role_denied(self) -> None:
        assert can_control_session("", "actor-1", "creator") is False

    def test_inactive_membership_denied(self) -> None:
        assert (
            can_control_session(
                WorkspaceRole.OWNER.value, "actor-1", "creator", membership_active=False
            )
            is False
        )
        assert (
            can_control_session(
                WorkspaceRole.MEMBER.value, "actor-1", "actor-1", membership_active=False
            )
            is False
        )


class TestCanAssignRole:
    @pytest.mark.parametrize(
        "target",
        [WorkspaceRole.ADMIN.value, WorkspaceRole.MEMBER.value, WorkspaceRole.VIEWER.value],
    )
    @pytest.mark.parametrize("current", ["admin", "member", "viewer"])
    def test_owner_can_assign_non_owner_roles(self, target: str, current: str) -> None:
        assert can_assign_role("owner", target, current_role=current) is True

    def test_owner_cannot_assign_owner(self) -> None:
        assert can_assign_role("owner", "owner", current_role="member") is False

    @pytest.mark.parametrize("target", [WorkspaceRole.MEMBER.value, WorkspaceRole.VIEWER.value])
    @pytest.mark.parametrize("current", ["member", "viewer"])
    def test_admin_can_assign_member_viewer(self, target: str, current: str) -> None:
        assert can_assign_role("admin", target, current_role=current) is True

    def test_admin_cannot_assign_admin(self) -> None:
        assert can_assign_role("admin", "admin", current_role="member") is False

    def test_admin_cannot_assign_owner(self) -> None:
        assert can_assign_role("admin", "owner", current_role="member") is False

    @pytest.mark.parametrize("actor", [WorkspaceRole.MEMBER.value, WorkspaceRole.VIEWER.value])
    @pytest.mark.parametrize("target", [role.value for role in WorkspaceRole])
    def test_member_viewer_cannot_assign_any_role(self, actor: str, target: str) -> None:
        assert can_assign_role(actor, target, current_role="member") is False

    def test_unknown_actor_role_cannot_assign(self) -> None:
        assert can_assign_role("superadmin", "member", current_role="member") is False

    def test_none_actor_role_cannot_assign(self) -> None:
        assert can_assign_role(None, "member", current_role="member") is False

    def test_empty_actor_role_cannot_assign(self) -> None:
        assert can_assign_role("", "member", current_role="member") is False

    def test_owner_assign_unknown_target_rejected(self) -> None:
        assert can_assign_role("owner", "superadmin", current_role="member") is False

    def test_inactive_membership_cannot_assign(self) -> None:
        assert (
            can_assign_role("owner", "admin", current_role="member", membership_active=False)
            is False
        )

    @pytest.mark.parametrize("actor", ["owner", "admin", "member", "viewer"])
    @pytest.mark.parametrize("current", ["owner", "admin", "member", "viewer"])
    @pytest.mark.parametrize("target", ["owner", "admin", "member", "viewer"])
    def test_role_change_matrix(self, actor: str, current: str, target: str) -> None:
        allowed_changes = {
            "owner": {
                "admin": {"admin", "member", "viewer"},
                "member": {"admin", "member", "viewer"},
                "viewer": {"admin", "member", "viewer"},
            },
            "admin": {"member": {"member", "viewer"}, "viewer": {"member", "viewer"}},
        }
        expected = target in allowed_changes.get(actor, {}).get(current, set())
        assert can_assign_role(actor, target, current_role=current) is expected

    def test_admin_cannot_demote_owner(self) -> None:
        assert can_assign_role("admin", "member", current_role="owner") is False

    def test_admin_cannot_demote_admin(self) -> None:
        assert can_assign_role("admin", "viewer", current_role="admin") is False

    def test_owner_can_demote_admin(self) -> None:
        assert can_assign_role("owner", "member", current_role="admin") is True

    @pytest.mark.parametrize("current", [None, "", "superadmin"])
    @pytest.mark.parametrize("actor", ["owner", "admin"])
    def test_unknown_current_role_is_rejected(self, actor: str, current: str | None) -> None:
        assert can_assign_role(actor, "viewer", current_role=current) is False

    def test_current_role_is_required(self) -> None:
        with pytest.raises(TypeError, match="current_role"):
            signature(can_assign_role).bind("owner", "member")
