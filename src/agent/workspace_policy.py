"""Pure server-side workspace role policy.

This module encodes capabilities and control rules for the team workspace.
It has no database, HTTP, environment or UI dependencies and can be tested
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorkspaceRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


@dataclass(frozen=True, slots=True)
class WorkspaceCapabilities:
    view_project: bool
    run_agent: bool
    manage_members: bool
    manage_workspace: bool
    manage_models: bool


def _map_role(role: str | None) -> WorkspaceRole | None:
    if role is None:
        return None
    try:
        return WorkspaceRole(role)
    except ValueError:
        return None


def workspace_capabilities(
    role: str | None,
    *,
    membership_active: bool = True,
) -> WorkspaceCapabilities:
    """Return the capabilities of a role; an unknown role or an inactive member gets none."""
    mapped = _map_role(role)
    if mapped is None or not membership_active:
        return WorkspaceCapabilities(False, False, False, False, False)

    if mapped in (WorkspaceRole.OWNER, WorkspaceRole.ADMIN):
        return WorkspaceCapabilities(
            view_project=True,
            run_agent=True,
            manage_members=True,
            manage_workspace=True,
            manage_models=False,
        )
    if mapped is WorkspaceRole.MEMBER:
        return WorkspaceCapabilities(
            view_project=True,
            run_agent=True,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )
    if mapped is WorkspaceRole.VIEWER:
        return WorkspaceCapabilities(
            view_project=True,
            run_agent=False,
            manage_members=False,
            manage_workspace=False,
            manage_models=False,
        )
    return WorkspaceCapabilities(False, False, False, False, False)


def can_control_session(
    role: str | None,
    actor_id: str,
    creator_id: str | None,
    *,
    membership_active: bool = True,
) -> bool:
    """Allow owner and admin to control any session, and a member only their own session."""
    if not membership_active:
        return False
    if not actor_id:
        return False
    mapped = _map_role(role)
    if mapped is None:
        return False
    if mapped in (WorkspaceRole.OWNER, WorkspaceRole.ADMIN):
        return True
    if mapped is WorkspaceRole.MEMBER:
        return bool(creator_id) and creator_id == actor_id
    return False


def can_assign_role(
    actor_role: str | None,
    target_role: str,
    *,
    current_role: str | None,
    membership_active: bool = True,
) -> bool:
    """Decide whether the actor may change the current role of a member to ``target_role``.

    The owner may change any role except owner. An admin may change only member
    and viewer, and may assign only member or viewer. Nobody assigns or changes
    owner here: ownership transfer is a separate operation. Unknown roles are refused.
    """
    if not membership_active:
        return False
    mapped_actor = _map_role(actor_role)
    if mapped_actor is None:
        return False
    mapped_target = _map_role(target_role)
    if mapped_target is None:
        return False
    mapped_current = _map_role(current_role)
    if mapped_current is None:
        return False
    if mapped_target is WorkspaceRole.OWNER or mapped_current is WorkspaceRole.OWNER:
        return False
    if mapped_actor is WorkspaceRole.OWNER:
        return mapped_target in (WorkspaceRole.ADMIN, WorkspaceRole.MEMBER, WorkspaceRole.VIEWER)
    if mapped_actor is WorkspaceRole.ADMIN:
        return mapped_current in (WorkspaceRole.MEMBER, WorkspaceRole.VIEWER) and mapped_target in (
            WorkspaceRole.MEMBER,
            WorkspaceRole.VIEWER,
        )
    return False
