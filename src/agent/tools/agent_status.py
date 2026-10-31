"""Read bounded session facts without exposing conversation or tool payloads."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.database import session_factory
from agent.engine_access import check_engine_access
from agent.models import (
    Approval,
    Message,
    OutboundAudit,
    Project,
    Session,
    ToolCall,
    User,
    WorkspaceMembership,
)
from agent.session_budget_store import budget_limits, load_budget_usage
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.workspace_policy import workspace_capabilities

_APPROVAL_LIMIT = 10
_BUDGET_REASONS = {
    "steps_exhausted",
    "tokens_exhausted",
    "cost_exhausted",
    "deadline_exceeded",
    "token_usage_unknown",
    "cost_unknown",
    "start_time_unknown",
}


def _aware(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _stamp(moment: datetime | None) -> str | None:
    return _aware(moment).isoformat() if moment is not None else None


async def _check_access(
    db: AsyncSession, settings: Settings, current: Session, target: Session
) -> None:
    if current.project_id != target.project_id:
        raise ToolError("Session not found")
    project = await db.get(Project, current.project_id)
    if project is None:
        raise ToolError("Session not found")
    actor = current.created_by_user_id
    if actor is not None:
        user = await db.get(User, actor)
        if user is None or not user.enabled:
            raise ToolError("Session not found")
    if project.workspace_id is not None:
        membership = (
            await db.get(WorkspaceMembership, (project.workspace_id, actor)) if actor else None
        )
        if (
            membership is None
            or membership.revoked_at is not None
            or not workspace_capabilities(membership.role).view_project
        ):
            raise ToolError("Session not found")
    try:
        await check_engine_access(db, settings, current, actor)
        await check_engine_access(db, settings, target, actor)
    except PermissionError:
        raise ToolError("Session not found") from None


class AgentStatusTool(Tool):
    name = "agent_status"
    description = (
        "Read status, last activity, pending approvals and budget of a session in the same "
        "project. Reports stored facts only; silence does not prove that an agent is stuck."
    )
    risk_level = RiskLevel.read_only
    input_schema = {
        "type": "object",
        "properties": {"session_id": {"type": "string", "minLength": 1, "maxLength": 36}},
        "required": ["session_id"],
        "additionalProperties": False,
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC)
        async with session_factory() as db:
            current = await db.get(Session, context.session_id)
            target = await db.get(Session, str(arguments["session_id"]))
            if current is None or target is None:
                raise ToolError("Session not found")
            await _check_access(db, self.settings, current, target)
            limits = budget_limits(self.settings, target)
            usage = await load_budget_usage(db, target, now)
            state = target.budget_state or {}
            same_run = state.get("run_generation") == target.run_generation
            started_at = usage.started_at if same_run else None
            tool = (
                await db.execute(
                    select(
                        ToolCall.id,
                        ToolCall.name,
                        ToolCall.status,
                        ToolCall.created_at,
                        ToolCall.updated_at,
                    )
                    .where(ToolCall.session_id == target.id)
                    .order_by(ToolCall.updated_at.desc(), ToolCall.id.desc())
                    .limit(1)
                )
            ).first()
            active_tool = None
            if target.status == "running" and started_at is not None:
                active_tool = (
                    await db.execute(
                        select(ToolCall.id, ToolCall.name, ToolCall.arguments)
                        .where(
                            ToolCall.session_id == target.id,
                            ToolCall.status == "running",
                            ToolCall.created_at >= started_at,
                        )
                        .order_by(ToolCall.updated_at.desc(), ToolCall.id.desc())
                        .limit(1)
                    )
                ).first()
            inference = (
                await db.execute(
                    select(
                        OutboundAudit.id,
                        OutboundAudit.status,
                        OutboundAudit.created_at,
                        OutboundAudit.finished_at,
                    )
                    .where(OutboundAudit.session_id == target.id, OutboundAudit.category == "llm")
                    .order_by(OutboundAudit.id.desc())
                    .limit(1)
                )
            ).first()
            message_at = await db.scalar(
                select(func.max(Message.created_at)).where(Message.session_id == target.id)
            )
            approvals_query = (
                select(Approval.id, ToolCall.name)
                .join(ToolCall, ToolCall.id == Approval.tool_call_id)
                .where(
                    ToolCall.session_id == target.id,
                    ToolCall.status == "awaiting_confirmation",
                    Approval.status == "pending",
                    or_(
                        Approval.generation == target.run_generation,
                        (Approval.generation.is_(None)) & (target.run_generation == 0),
                    ),
                    or_(Approval.expires_at.is_(None), Approval.expires_at > now),
                )
            )
            pending_count = (
                await db.scalar(select(func.count()).select_from(approvals_query.subquery())) or 0
            )
            approvals = (
                await db.execute(
                    approvals_query.order_by(Approval.created_at, Approval.id).limit(
                        _APPROVAL_LIMIT
                    )
                )
            ).all()
            activity = [_aware(target.updated_at)]
            if message_at is not None:
                activity.append(_aware(message_at))
            if tool is not None:
                activity.append(_aware(tool.updated_at))
            if inference is not None:
                activity.append(_aware(inference.finished_at or inference.created_at))
            last_activity = max(activity)
            waiting_for = None
            if target.status == "awaiting_confirmation":
                waiting_for = "approval"
            elif target.status == "running" and started_at is not None:
                if active_tool is not None:
                    waiting_for = (
                        "child"
                        if active_tool.name == "run_agent"
                        and active_tool.arguments.get("wait", True) is True
                        else "tool"
                    )
                elif (
                    inference is not None
                    and inference.status == "started"
                    and _aware(inference.created_at) >= started_at
                ):
                    waiting_for = "model"
            reason = state.get("stop_reason") if same_run else None
            return {
                "session_id": target.id,
                "status": target.status,
                "run_generation": target.run_generation,
                "stop_requested": target.stop_requested,
                "observed_at": _stamp(now),
                "last_activity_at": _stamp(last_activity),
                "seconds_since_activity": max(0, int((now - last_activity).total_seconds())),
                "waiting_for": waiting_for,
                "active_tool_call": {
                    "id": active_tool.id,
                    "name": active_tool.name[:100],
                }
                if active_tool is not None
                else None,
                "latest_tool_call": {
                    "id": tool.id,
                    "name": tool.name[:100],
                    "status": tool.status[:50],
                    "updated_at": _stamp(tool.updated_at),
                }
                if tool is not None
                else None,
                "latest_inference": {
                    "id": inference.id,
                    "status": inference.status[:30],
                    "started_at": _stamp(inference.created_at),
                    "finished_at": _stamp(inference.finished_at),
                }
                if inference is not None
                else None,
                "approvals": {
                    "pending_count": pending_count,
                    "items": [{"id": row.id, "tool": row.name[:100]} for row in approvals],
                    "truncated": pending_count > _APPROVAL_LIMIT,
                },
                "budget": {
                    "max_steps": limits.max_steps,
                    "steps": usage.steps,
                    "max_tokens": limits.max_tokens,
                    "known_tokens": usage.known_tokens,
                    "max_cost": str(limits.max_cost) if limits.max_cost is not None else None,
                    "known_cost": str(usage.known_cost),
                    "has_unknown_tokens": usage.has_unknown_tokens,
                    "has_unknown_cost": usage.has_unknown_cost,
                    "deadline_seconds": limits.deadline_seconds,
                    "started_at": _stamp(started_at),
                    "stop_reason": reason if reason in _BUDGET_REASONS else None,
                }
                if target.engine_profile is None
                else None,
            }
