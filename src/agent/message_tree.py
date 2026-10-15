"""Message tree of a session: append, edit, regenerate, branch switch and fork.

Invariant: the active rows of a session (kind "normal", skipped False) form one chain from the
root to the head. A child is always created after its parent, so along any branch the ids grow,
and the id order of the active rows is the branch order. Context building, summaries and
checkpoints therefore keep working on the active rows.

Rules:
- Rows are immutable. An edit or a regeneration adds a new user row next to the old one; the old
  rows are only hidden (skipped). Summary rows are not part of the chain and have no parent.
- Tree changes run only while the session is idle (completed, stopped, error, interrupted) and no
  executor holds a lease. The idle check, the ownership checks and the tree change are one
  transaction. Edit and regenerate move the session to "pending" and start a new run generation in
  that transaction, so a stale executor cannot write and no second request can switch the branch
  before the new run starts.
- Pending tool calls and approvals of a hidden tail are cancelled, so they never run later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import get_settings
from agent.model_registry import effective_registry, select_model
from agent.models import (
    Approval,
    ApprovalStatus,
    Message,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from agent.reasoning import validate_effort
from agent.session_fence import CURRENT_FENCE, fence_check

IDLE_STATUSES = (
    SessionStatus.completed.value,
    SessionStatus.stopped.value,
    SessionStatus.error.value,
    SessionStatus.interrupted.value,
)
_UNFINISHED_CALLS = (ToolCallStatus.pending.value, ToolCallStatus.awaiting_confirmation.value)


class TreeError(ValueError):
    """A tree operation is not possible; ``status`` is the HTTP status for the API."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class Siblings:
    ids: list[int]
    active_index: int | None


async def _rowcount(db: AsyncSession, statement: sa.Executable) -> int:
    return cast(CursorResult[Any], await db.execute(statement)).rowcount


async def lock_session(db: AsyncSession, session_id: str) -> None:
    """Lock the session row until the commit; in an executor, check its fence instead."""
    fence = CURRENT_FENCE.get()
    if fence is not None and fence.session_id == session_id:
        await fence_check(db, fence)
        return
    touched = await _rowcount(
        db, sa.update(Session).where(Session.id == session_id).values(id=Session.id)
    )
    if touched != 1:
        raise TreeError("Session not found", 404)


async def current_head(db: AsyncSession, session_id: str) -> Message | None:
    """Return the last active normal row of the session (the head of the active chain)."""
    return (
        await db.execute(
            sa.select(Message)
            .where(
                Message.session_id == session_id,
                Message.kind == "normal",
                Message.skipped.is_(False),
            )
            .order_by(Message.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def append_message(db: AsyncSession, session_id: str, **fields: Any) -> Message:
    """Add a message after the current head; no commit.

    Call it once per message, in order: each call locks the session, reads the head (including
    rows added earlier in the same transaction) and flushes the new row.
    """
    if "parent_id" in fields:
        raise TreeError("append_message sets parent_id itself", 400)
    await lock_session(db, session_id)
    head = await current_head(db, session_id)
    message = Message(session_id=session_id, parent_id=head.id if head else None, **fields)
    db.add(message)
    await db.flush()
    return message


async def _claim_idle(db: AsyncSession, session_id: str, *, start_run: bool) -> None:
    """Take an idle session for a tree change; a new generation makes stale executors fail."""
    values: dict[str, Any] = {
        "run_generation": Session.run_generation + 1,
        "owner_token": None,
        "lease_until": None,
    }
    if start_run:
        values.update(status=SessionStatus.pending.value, stop_requested=False, error=None)
    claimed = await _rowcount(
        db,
        sa.update(Session)
        .where(
            Session.id == session_id,
            Session.status.in_(IDLE_STATUSES),
            # Free: no owner, or an owner whose lease is known and expired. An owner with an
            # unknown (NULL) lease is not proven gone, so the session stays busy.
            sa.or_(
                Session.owner_token.is_(None),
                sa.and_(
                    Session.lease_until.is_not(None),
                    Session.lease_until < datetime.now(UTC),
                ),
            ),
        )
        .values(**values)
        # The database evaluates the condition. The ORM must not evaluate it in Python on a
        # cached row: SQLite gives naive datetimes, and the comparison would fail.
        .execution_options(synchronize_session=False),
    )
    if claimed != 1:
        exists = await db.scalar(sa.select(Session.id).where(Session.id == session_id))
        if exists is None:
            raise TreeError("Session not found", 404)
        raise TreeError("The session is busy; wait until it finishes or stop it", 409)
    # Refresh a cached Session row: the update above did not synchronize it.
    await db.get(Session, session_id, populate_existing=True)


async def _active_message(db: AsyncSession, session_id: str, message_id: int) -> Message:
    message = await db.get(Message, message_id)
    if message is None or message.session_id != session_id:
        raise TreeError("Message not found in this session", 404)
    if message.kind != "normal" or message.skipped:
        raise TreeError("Only an active message can be changed", 409)
    return message


async def _hide_from(db: AsyncSession, session_id: str, first_id: int) -> None:
    """Hide the active tail that starts at ``first_id`` and cancel its unfinished tool calls."""
    hidden_ids = list(
        (
            await db.execute(
                sa.select(Message.id).where(
                    Message.session_id == session_id,
                    Message.kind == "normal",
                    Message.skipped.is_(False),
                    Message.id >= first_id,
                )
            )
        ).scalars()
    )
    if hidden_ids:
        await db.execute(sa.update(Message).where(Message.id.in_(hidden_ids)).values(skipped=True))
    await _cancel_unfinished_calls(db, session_id)


async def _cancel_unfinished_calls(db: AsyncSession, session_id: str) -> None:
    call_ids = list(
        (
            await db.execute(
                sa.select(ToolCall.id).where(
                    ToolCall.session_id == session_id, ToolCall.status.in_(_UNFINISHED_CALLS)
                )
            )
        ).scalars()
    )
    if not call_ids:
        return
    await db.execute(
        sa.update(ToolCall)
        .where(ToolCall.id.in_(call_ids))
        .values(status=ToolCallStatus.cancelled.value)
    )
    await db.execute(
        sa.update(Approval)
        .where(Approval.tool_call_id.in_(call_ids), Approval.status == ApprovalStatus.pending.value)
        .values(status=ApprovalStatus.cancelled.value)
    )


async def _replace_user_turn(
    db: AsyncSession, session_id: str, user: Message, content: str, reasoning_effort: str | None
) -> Message:
    """Hide ``user`` and its tail; add a new user row with the same parent. No commit."""
    await _hide_from(db, session_id, user.id)
    new_user = Message(
        session_id=session_id,
        parent_id=user.parent_id,
        role="user",
        content=content,
        reasoning_effort=reasoning_effort,
    )
    db.add(new_user)
    await db.flush()
    return new_user


async def edit_message(db: AsyncSession, session_id: str, message_id: int, content: str) -> Message:
    """Replace a user message with a new version and prepare a new run; no commit.

    After the commit, schedule the run with ``supervisor.start(session_id, new_run=False)``:
    the new generation is already set here.
    """
    if not content.strip():
        raise TreeError("The message text is empty", 400)
    await _claim_idle(db, session_id, start_run=True)
    user = await _active_message(db, session_id, message_id)
    if user.role != "user":
        raise TreeError("Only a user message can be edited", 409)
    return await _replace_user_turn(db, session_id, user, content, user.reasoning_effort)


async def regenerate(
    db: AsyncSession,
    session_id: str,
    message_id: int,
    *,
    llm_profile: str | None = None,
    reasoning_effort: str | None = None,
) -> Message:
    """Run the turn of an assistant message again; no commit.

    The turn restarts from the nearest user ancestor: a copy of that user row is added next to it,
    so the old answer stays available as a sibling branch. Overrides apply to the new run. The
    requested model profile and reasoning effort are validated the same way ``add_message`` does,
    against the session state read after the claim; an invalid choice raises ``ValueError``
    before the tree changes, and the caller rolls back the claim, so no run starts.
    """
    settings = get_settings()
    # Claim first, then validate against the state read after the claim, in the same
    # transaction: a cached Session may carry an old model profile. The caller rolls back
    # the claim when validation fails.
    await _claim_idle(db, session_id, start_run=True)
    session = await db.get(Session, session_id, populate_existing=True)
    if session is None:
        raise TreeError("Session not found", 404)
    models = await effective_registry(db, settings)
    if llm_profile is not None:
        model = select_model(models, llm_profile, strict=True)
    else:
        model = select_model(models, session.llm_profile, strict=False)
    if reasoning_effort is not None:
        validate_effort(model, reasoning_effort)
    target = await _active_message(db, session_id, message_id)
    if target.role != "assistant":
        raise TreeError("Only an assistant message can be regenerated", 409)
    user = await _nearest_user_ancestor(db, session_id, target)
    if not (user.content or "").strip():
        raise TreeError("The message text is empty", 409)
    if llm_profile is not None:
        await db.execute(
            sa.update(Session).where(Session.id == session_id).values(llm_profile=llm_profile)
        )
    effort = reasoning_effort if reasoning_effort is not None else user.reasoning_effort
    return await _replace_user_turn(db, session_id, user, user.content or "", effort)


async def _nearest_user_ancestor(db: AsyncSession, session_id: str, message: Message) -> Message:
    path = await path_to(db, session_id, message.id)
    for row in reversed(path):
        if row.role == "user":
            return row
    raise TreeError("The message has no user turn to restart", 409)


async def path_to(db: AsyncSession, session_id: str, leaf_id: int) -> list[Message]:
    """Return the rows from the root to ``leaf_id`` by parent links; reject broken links."""
    path: list[Message] = []
    current = await db.get(Message, leaf_id)
    seen: set[int] = set()
    while current is not None:
        if current.session_id != session_id or current.kind != "normal":
            raise TreeError("The branch is broken: a parent is outside this session", 409)
        if current.id in seen or (path and current.id >= path[-1].id):
            raise TreeError("The branch is broken: parents must be older rows", 409)
        seen.add(current.id)
        path.append(current)
        if current.parent_id is None:
            break
        parent = await db.get(Message, current.parent_id)
        if parent is None:
            raise TreeError("The branch is broken: a parent row is missing", 409)
        current = parent
    if not path:
        raise TreeError("Message not found in this session", 404)
    path.reverse()
    return path


async def switch_branch(db: AsyncSession, session_id: str, leaf_id: int) -> list[Message]:
    """Make the path to ``leaf_id`` the active chain; no commit. Returns the new active path."""
    await _claim_idle(db, session_id, start_run=False)
    leaf = await db.get(Message, leaf_id)
    if leaf is None or leaf.session_id != session_id or leaf.kind != "normal":
        raise TreeError("Message not found in this session", 404)
    path = await path_to(db, session_id, leaf_id)
    await db.execute(
        sa.update(Message)
        .where(Message.session_id == session_id, Message.kind == "normal")
        .values(skipped=True)
    )
    await db.execute(
        sa.update(Message).where(Message.id.in_([row.id for row in path])).values(skipped=False)
    )
    await _cancel_unfinished_calls(db, session_id)
    return path


async def fork(
    db: AsyncSession,
    session_id: str,
    message_id: int,
    *,
    title: str,
    actor_user_id: str | None = None,
) -> Session:
    """Copy the path to ``message_id`` into a new idle session; no commit.

    The source row is locked first; its state and lease are read after the lock, so the copy
    cannot interleave with an edit or a run. The source generation does not change (read only).
    The copy belongs to the requesting actor and has new ids and parent links. Checkpoints,
    tool calls and approvals are not copied.

    Tool rows and tool-call references are not copied. The provider history must not contain a
    tool message without the matching assistant call; the tool rows themselves live in the
    ``tool_calls`` table of the source session and their rows are not copied here. Skipping the
    ``tool`` rows and dropping ``tool_call_id``/``tool_calls`` from the copied assistant rows
    keeps the provider chain clean: the assistant text survives, while no orphaned references
    remain. An assistant row with no text (only tool calls) is skipped, so no empty turn remains.
    """
    await lock_session(db, session_id)
    state = (
        await db.execute(
            sa.select(Session.status, Session.owner_token, Session.lease_until).where(
                Session.id == session_id
            )
        )
    ).one()
    lease = state.lease_until
    if lease is not None and lease.tzinfo is None:
        lease = lease.replace(tzinfo=UTC)
    # An owner with an unknown (NULL) lease is not proven gone: treat it as alive.
    owner_alive = state.owner_token is not None and (lease is None or lease >= datetime.now(UTC))
    if state.status not in IDLE_STATUSES or owner_alive:
        raise TreeError("The session is busy; wait until it finishes or stop it", 409)
    source = await db.get(Session, session_id, populate_existing=True)
    if source is None:
        raise TreeError("Session not found", 404)
    leaf = await db.get(Message, message_id)
    if leaf is None or leaf.session_id != session_id:
        raise TreeError("Message not found in this session", 404)
    path = await path_to(db, session_id, message_id)
    copy = Session(
        project_id=source.project_id,
        mode=source.mode,
        llm_profile=source.llm_profile,
        title=title,
        created_by_user_id=actor_user_id,
        status=SessionStatus.completed.value,
        configuration=dict(source.configuration or {}),
    )
    db.add(copy)
    await db.flush()
    parent_id: int | None = None
    for row in path:
        if row.role == "tool":
            # A tool message has no meaning without its call; skip it entirely.
            continue
        if row.role == "assistant" and not (row.content or "").strip():
            # Only tool calls and no text: without the calls the turn is empty; skip it.
            continue
        clone = Message(
            session_id=copy.id,
            parent_id=parent_id,
            role=row.role,
            content=row.content,
            token_count=row.token_count,
            reasoning_content=row.reasoning_content,
            reasoning_effort=row.reasoning_effort,
        )
        db.add(clone)
        await db.flush()
        parent_id = clone.id
    return copy


async def siblings(db: AsyncSession, session_id: str, message_id: int) -> Siblings:
    """Return the versions of a message: rows with the same parent and role, in id order."""
    message = await db.get(Message, message_id)
    if message is None or message.session_id != session_id or message.kind != "normal":
        raise TreeError("Message not found in this session", 404)
    parent_filter = (
        Message.parent_id.is_(None)
        if message.parent_id is None
        else Message.parent_id == message.parent_id
    )
    rows = list(
        (
            await db.execute(
                sa.select(Message.id, Message.skipped)
                .where(
                    Message.session_id == session_id,
                    Message.kind == "normal",
                    Message.role == message.role,
                    parent_filter,
                )
                .order_by(Message.id)
            )
        ).all()
    )
    ids = [int(row.id) for row in rows]
    active = next((index for index, row in enumerate(rows) if not row.skipped), None)
    return Siblings(ids, active)
