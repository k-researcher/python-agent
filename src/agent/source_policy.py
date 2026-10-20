"""Decide access to tool sources from trusted server-side facts.

This module does not authenticate requests. It does not replace tool risk checks,
approval checks, or validation that an approval still matches the source settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.workspace_policy import workspace_capabilities


@dataclass(frozen=True, slots=True)
class SourceActor:
    """Store trusted facts about a user and their workspace membership."""

    user_id: str
    workspace_id: str
    role: str | None
    user_enabled: bool = True
    membership_active: bool = True


@dataclass(frozen=True, slots=True)
class SourceScope:
    """Limit a tool source to one workspace and optional project and mode sets."""

    source_id: str
    workspace_id: str
    project_ids: frozenset[str] = frozenset()
    enabled: bool = True
    allowed_modes: frozenset[str] = frozenset({"dev", "ask"})


def _valid_id(value: str) -> bool:
    return bool(value.strip())


def _valid_actor(actor: SourceActor, source: SourceScope) -> bool:
    return (
        all(
            _valid_id(value)
            for value in (
                actor.user_id,
                actor.workspace_id,
                source.source_id,
                source.workspace_id,
            )
        )
        and actor.user_enabled
        and actor.membership_active
        and actor.workspace_id == source.workspace_id
    )


def can_manage_source(actor: SourceActor, source: SourceScope) -> bool:
    """Allow an active owner or admin to manage even a disabled source."""
    return _valid_actor(actor, source) and workspace_capabilities(actor.role).manage_workspace


def can_use_source(
    actor: SourceActor,
    source: SourceScope,
    *,
    project_id: str,
    project_workspace_id: str | None,
    mode: str,
) -> bool:
    """Allow an active runner to use an enabled source within its exact scope."""
    return (
        _valid_actor(actor, source)
        and source.enabled
        and _valid_id(project_id)
        and project_workspace_id == source.workspace_id
        and (not source.project_ids or project_id in source.project_ids)
        and workspace_capabilities(actor.role).run_agent
        and mode in {"dev", "ask"}
        and mode in source.allowed_modes
    )
