from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
import uuid
from contextlib import nullcontext, suppress
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import attributes, selectinload

from agent.checkpoints import load_context_rows
from agent.config import Settings
from agent.context_inference import TRANSIENT_KINDS, ContextInference, InferenceResult
from agent.context_manager import ContextManager
from agent.database import session_factory
from agent.engine_access import check_engine_access
from agent.engine_adapters import EngineRegistry, configured_profiles
from agent.engine_journal import recover_engines
from agent.engine_runner import EngineRunner
from agent.events import (
    CURRENT_STREAMS,
    PendingEvent,
    StreamPublisher,
    add_event,
    broker,
    emit_event,
    publish,
    resume_stream_run,
)
from agent.inference_guard import (
    CURRENT_RUN_GUARD,
    GuardContext,
    LoopClock,
    bounded_cleanup,
    guard_error,
    retain,
)
from agent.llm import LLMClient, LLMError, LLMResponse
from agent.llm_errors import LLMErrorKind
from agent.mcp_audit import execute_mcp_attempt
from agent.message_tree import append_message
from agent.model_config import ModelConfigError, ResolvedModel, ResolvedModelRegistry
from agent.model_registry import effective_registry, select_model
from agent.models import (
    Approval,
    ApprovalStatus,
    Message,
    OutboundAudit,
    Project,
    Session,
    SessionInbox,
    SessionRunOutcome,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
    utcnow,
)
from agent.modes import ROLE_PROFILES
from agent.queue import RedisTaskQueue
from agent.reasoning import ReasoningDecision, decide, lower_effort
from agent.response_validation import arguments_object
from agent.run_finish import finish_run
from agent.sandbox import policy_from_settings
from agent.session_budget import check_budget
from agent.session_budget_store import (
    BudgetExceeded,
    budget_limits,
    load_budget_usage,
    mark_budget_stop,
    record_step,
)
from agent.session_fence import (
    CURRENT_FENCE,
    FenceLostError,
    RunFence,
    claim,
    fence_check,
    install_commit_guard,
    new_generation,
    release,
    renew,
)
from agent.session_inbox import (
    auto_claim,
    deliver,
    enqueue,
    has_eligible,
    lock_session,
    promote_notices,
    retryable,
)
from agent.session_service import create_session_record
from agent.skill_prompt import load_skill_set
from agent.source_execution import capture_source_revision, validate_source_execution
from agent.source_lifecycle import source_registry
from agent.tool_batch import BatchCall, BatchResult, is_parallel_safe, run_batch
from agent.tools import ToolRegistry
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.mcp_tools import McpBackedTool
from agent.tools.skill_tools import ReadSkillTool

logger = logging.getLogger(__name__)

# Lease of the run owner, renewed every third of it. A lost lease cancels the run.
FENCE_LEASE_SECONDS = 60.0
RECONCILE_SECONDS = 5.0
# Most read-only tool calls of one model reply that run at the same time.
PARALLEL_TOOL_LIMIT = 4

TERMINAL_STATUSES = {
    SessionStatus.completed.value,
    SessionStatus.stopped.value,
    SessionStatus.error.value,
    SessionStatus.interrupted.value,
}


def _batch_result(outcome: BatchResult) -> dict[str, Any]:
    """Convert a batch outcome to the result shape of a sequential tool call."""
    if outcome.ok:
        return {"success": True, "output": outcome.output}
    return {"success": False, "error": outcome.error}


# A cut-off reply or a broken stream is retried this many times in a row before the run fails.
MAX_CUT_OFF_REPLIES = 2
MAX_STREAM_RETRIES = 2
MAX_INVALID_CALL_REPLIES = 2
INVALID_CALL_NOTICE = (
    "Служебное уведомление агента: предыдущий ответ модели содержит некорректный вызов "
    "инструмента и не использован, инструменты из него не запускались. Повтори шаг: укажи "
    "имя инструмента из списка и аргументы строго по его схеме."
)
CUT_OFF_NOTICE = (
    "Служебное уведомление агента: предыдущий ответ модели обрезан лимитом вывода и не "
    "использован, инструменты из него не запускались. Сделай шаг меньше: запиши большой файл "
    "по частям (write_file, затем edit_file) или раздели правку на несколько вызовов."
)


def _cut_in_reasoning(response: LLMResponse) -> bool:
    """Tell whether the output limit cut the reply while the model was still reasoning."""
    return (
        response.finish_reason == "length"
        and not (response.content or "").strip()
        and not response.tool_calls
        and bool((response.reasoning or "").strip())
    )


def _lowered(
    decisions: dict[str, ReasoningDecision], models: ResolvedModelRegistry
) -> dict[str, ReasoningDecision]:
    """Return the decisions with each effective level one step lower where the model allows it."""
    lowered: dict[str, ReasoningDecision] = {}
    for model_id, decision in decisions.items():
        lower = lower_effort(models.models[model_id], decision.effective)
        lowered[model_id] = (
            decision
            if lower is None
            else ReasoningDecision(
                decision.requested, lower, decision.source, "lowered after a reply cut off"
            )
        )
    return lowered


def _history_safe_call(call: dict[str, Any]) -> dict[str, Any]:
    """Return the tool call with arguments that are valid JSON for the provider history."""
    function = dict(call.get("function") or {})
    try:
        parsed = json.loads(function.get("arguments") or "{}")
        valid = isinstance(parsed, dict)
    except (TypeError, ValueError):
        valid = False
    if valid:
        return call
    function["arguments"] = "{}"
    return {**call, "function": function}


class AgentSupervisor:
    def __init__(self, settings: Settings, registry: ToolRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ToolRegistry(settings=settings)
        self._run_registry: ContextVar[ToolRegistry | None] = ContextVar(
            "session_tool_registry", default=None
        )
        # Optional override for the default model's client (used by tests and fakes).
        self.llm: LLMClient | None = None
        self._llm_clients: dict[str, LLMClient] = {}
        # Replaced clients can still serve a running request; close them at shutdown only.
        self._retired_clients: list[LLMClient] = []
        self._last_models: ResolvedModelRegistry | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # The generation that each task in _tasks was scheduled for (None: not known).
        self._task_generation: dict[str, int | None] = {}
        # A waited child's continuation uses its parent's slot, including after approval.
        self._waited_children: set[str] = set()
        # Tasks of older generations that still finish, with their session; kept referenced
        # until they end, and cancelled by Stop of the session and by shutdown.
        self._finishing: dict[asyncio.Task[None], str] = {}
        self._semaphore = asyncio.Semaphore(settings.max_parallel_sessions)
        self._engine_registry: EngineRegistry | None = None
        self._engine_slots: dict[str, asyncio.Semaphore] = {}
        self._engine_sessions: dict[str, asyncio.Lock] = {}
        self._engine_tasks: set[asyncio.Task[Any]] = set()
        self._reconcile_signal = asyncio.Event()
        self.task_queue = RedisTaskQueue(settings) if settings.execution_mode == "redis" else None
        install_commit_guard()

    @property
    def engine_registry(self) -> EngineRegistry:
        if self._engine_registry is None:
            self._engine_registry = EngineRegistry(configured_profiles(self.settings))
        return self._engine_registry

    async def start(
        self,
        session_id: str,
        *,
        new_run: bool = True,
        expected_generation: int | None = None,
        actor_user_id: str | None = None,
    ) -> bool:
        """Schedule the session. A new run starts a new generation; a resume continues it.

        With ``expected_generation`` (a resume of a generation that the caller already
        claimed), the contract is explicit:
        - True: this generation is scheduled now, or it is already scheduled or running.
        - False: the session no longer is at this generation (a Stop, a newer run, a
          terminal state); nothing is scheduled.
        A task of an older generation that is still finishing does not block the new
        generation: the new run starts at once, and the fence stops the old task at its
        next write. The old task is not cancelled here.
        Without ``expected_generation`` the old behaviour stays: an alive task in this
        process makes the call return False.
        """
        if expected_generation is not None and new_run:
            # A new run makes a new generation, so it can never match the expected one.
            raise ValueError("expected_generation needs new_run=False")
        async with session_factory() as db:
            record = await db.get(Session, session_id)
            if record is not None and record.engine_profile is not None:
                # A queued generation carries the identity of the already authorized
                # creator. A direct new run must supply its actor explicitly.
                actor = actor_user_id
                if actor is None and expected_generation is not None:
                    actor = (record.engine_state or {}).get("operator_user_id")
                await check_engine_access(db, self.settings, record, actor)
        if expected_generation is not None and not await self._at_generation(
            session_id, expected_generation
        ):
            return False
        if self.task_queue is None:
            task = self._tasks.get(session_id)
            if task is not None and not task.done():
                if expected_generation is None:
                    return False
                if self._task_generation.get(session_id) == expected_generation:
                    return True
                # Keep a reference until the older task ends.
                self._finishing[task] = session_id
                task.add_done_callback(lambda done: self._finishing.pop(done, None))
        if new_run:
            async with session_factory() as db:
                await new_generation(db, session_id)
                await db.commit()
        if self.task_queue is not None:
            await self.task_queue.ensure_group()
            queued = await self.task_queue.enqueue(session_id, generation=expected_generation)
            # A duplicate of the same generation is already queued: it is not a lost run.
            return queued or expected_generation is not None
        self._tasks[session_id] = asyncio.create_task(
            self._run_guarded(session_id, expected_generation),
            name=f"agent-session-{session_id}",
        )
        self._task_generation[session_id] = expected_generation
        return True

    async def _at_generation(self, session_id: str, generation: int) -> bool:
        """Return True when the session is still at this generation and no Stop is pending."""
        async with session_factory() as db:
            row = (
                await db.execute(
                    select(Session.run_generation, Session.stop_requested, Session.status).where(
                        Session.id == session_id
                    )
                )
            ).one_or_none()
        return (
            row is not None
            and row.run_generation == generation
            and not row.stop_requested
            and row.status not in TERMINAL_STATUSES
        )

    async def run_job(self, session_id: str, generation: int | None = None) -> None:
        await self._run_guarded(session_id, generation)

    async def reconcile_pending(self) -> None:
        """Dispatch durable pending intents whose executor is absent or expired."""
        token = CURRENT_FENCE.set(None)
        try:
            async with session_factory() as db:
                recovered = await recover_engines(db)
                await db.commit()
            await publish(recovered)
            await self._reconcile_inbox()
        finally:
            CURRENT_FENCE.reset(token)
        async with session_factory() as db:
            rows = (
                await db.execute(
                    select(Session.id, Session.run_generation).where(
                        Session.status == SessionStatus.pending.value,
                        Session.run_generation > 0,
                        Session.stop_requested.is_(False),
                        or_(
                            Session.owner_token.is_(None),
                            Session.lease_until < datetime.now(UTC),
                        ),
                    )
                )
            ).all()
        for session_id, generation in rows:
            try:
                await self.start(session_id, new_run=False, expected_generation=generation)
            except Exception:
                logger.exception("Session %s: pending dispatch will be retried", session_id)

    async def _reconcile_inbox(self) -> None:
        await promote_notices(session_factory)
        async with session_factory() as db:
            targets = (
                await db.execute(
                    select(Session.id, Session.run_generation).where(
                        Session.status == "completed",
                        select(SessionInbox.id)
                        .where(
                            SessionInbox.target_session_id == Session.id,
                            SessionInbox.delivered_at.is_(None),
                            SessionInbox.rejected_at.is_(None),
                        )
                        .exists(),
                    )
                )
            ).all()
        for session_id, generation in targets:
            try:
                async with session_factory() as db:
                    claimed = await auto_claim(db, session_id, generation, self.settings)
                    events = []
                    if claimed is not None:
                        events.append(
                            await add_event(
                                db,
                                "session.status",
                                {
                                    "status": "pending",
                                    "generation": claimed,
                                },
                                session_id,
                            )
                        )
                    await db.commit()
                await publish(events)
            except DBAPIError:
                logger.exception("Session %s: inbox wake will be retried", session_id)

    def start_reconciliation(self) -> asyncio.Task[None]:
        """Start reconciliation; the API or worker lifespan owns the returned task."""
        self._reconcile_signal = asyncio.Event()
        return asyncio.create_task(self._reconcile_loop(), name="pending-reconciler")

    async def _reconcile_loop(self) -> None:
        while True:
            self._reconcile_signal.clear()
            try:
                await self.reconcile_pending()
            except Exception:
                logger.exception("Pending reconciliation will be retried")
            with suppress(TimeoutError):
                await asyncio.wait_for(self._reconcile_signal.wait(), RECONCILE_SECONDS)

    async def infrastructure_status(self) -> dict[str, Any]:
        if self.task_queue is None:
            return {"execution_mode": "embedded", "redis": "disabled", "workers": 0}
        status = await self.task_queue.health()
        return {"execution_mode": "redis", **status}

    async def stop(self, session_id: str) -> None:
        """Stop the active subtree in one transaction, then cancel local executors."""
        token = CURRENT_FENCE.set(None)
        try:
            pending: list[PendingEvent] = []
            targets: set[asyncio.Task[None]] = set()
            stopped_runs: list[tuple[str, int]] = []
            seen: set[str] = set()
            remaining: list[tuple[str, int | None]] = [(session_id, None)]
            async with session_factory() as db:
                while remaining:
                    current_id, generation = remaining.pop()
                    if current_id in seen:
                        continue
                    seen.add(current_id)
                    captured = {
                        task for task, owner in self._finishing.items() if owner == current_id
                    }
                    current = self._tasks.get(current_id)
                    if current is not None:
                        captured.add(current)
                    rows = await finish_run(
                        db,
                        current_id,
                        generation,
                        "stopped",
                        source="stop" if generation is None else "scoped_stop",
                    )
                    pending.extend(rows)
                    if generation is not None and not rows:
                        current_generation = await db.scalar(
                            select(Session.run_generation)
                            .where(Session.id == current_id)
                            .with_for_update()
                        )
                        if current_generation != generation:
                            continue
                    if rows:
                        stopped_runs.append((current_id, rows[0].payload["payload"]["generation"]))
                        targets.update(captured)
                        cancelled = await self._cancel_unfinished_tool_calls(db, current_id)
                        if cancelled:
                            pending.append(
                                await add_event(
                                    db,
                                    "approval.cancelled",
                                    {"tool_call_ids": cancelled},
                                    current_id,
                                )
                            )
                    remaining.extend(
                        (child_id, child_generation)
                        for child_id, child_generation in await db.execute(
                            select(Session.id, Session.run_generation)
                            .where(Session.parent_id == current_id)
                            .order_by(Session.id)
                            .with_for_update()
                        )
                    )
                await db.commit()
                for stopped_id, stopped_generation in stopped_runs:
                    targets.update(self._tasks_of_generation(stopped_id, stopped_generation))
                for task in targets:
                    task.cancel()
                await self._settle_stopped(targets, pending)
        finally:
            CURRENT_FENCE.reset(token)

    async def stop_scoped(self, session_id: str, expected_generation: int) -> bool:
        """Stop only the expected run generation; return True when it was stopped.

        The task of the expected generation is captured before the database write. The
        write checks the generation; on a mismatch nothing changes. After the commit the
        captured task is cancelled at once, before any event, and the method waits for it
        also when an event fails or the caller is cancelled. A task of a newer generation
        is never touched. There is no cascade to children.
        """
        token = CURRENT_FENCE.set(None)
        try:
            captured = self._tasks_of_generation(session_id, expected_generation)
            async with session_factory() as db:
                pending = await finish_run(
                    db, session_id, expected_generation, "stopped", source="scoped_stop"
                )
                if not pending:
                    return False
                cancelled = await self._cancel_unfinished_tool_calls(db, session_id)
                if cancelled:
                    pending.append(
                        await add_event(
                            db, "approval.cancelled", {"tool_call_ids": cancelled}, session_id
                        )
                    )
                await db.commit()
                targets = captured | self._tasks_of_generation(session_id, expected_generation)
                for task in targets:
                    task.cancel()
                await self._settle_stopped(targets, pending)
            return True
        finally:
            CURRENT_FENCE.reset(token)

    async def _settle_stopped(
        self, targets: set[asyncio.Task[None]], pending: list[PendingEvent]
    ) -> None:
        clock = LoopClock()
        for task in targets:
            retain(task)

        async def settle() -> None:
            try:
                await publish(pending)
            finally:
                if targets:
                    await asyncio.wait(targets)

        await bounded_cleanup(
            settle(), clock, clock.now() + (5 if targets & self._engine_tasks else 2)
        )

    def _tasks_of_generation(self, session_id: str, generation: int) -> set[asyncio.Task[None]]:
        """Return the unfinished current task of the session when it runs ``generation``."""
        task = self._tasks.get(session_id)
        if task is None or task.done() or self._task_generation.get(session_id) != generation:
            return set()
        return {task}

    @staticmethod
    async def _cancel_unfinished_tool_calls(db: Any, session_id: str) -> list[str]:
        """Withdraw queued and awaiting calls so nothing approved before Stop runs later."""
        calls = (
            await db.execute(
                select(ToolCall)
                .where(
                    ToolCall.session_id == session_id,
                    or_(
                        ToolCall.status.in_(
                            {
                                ToolCallStatus.pending.value,
                                ToolCallStatus.awaiting_confirmation.value,
                            }
                        ),
                        (ToolCall.name == "message_to_session") & (ToolCall.status == "running"),
                    ),
                )
                .options(selectinload(ToolCall.approval))
            )
        ).scalars()
        cancelled: list[str] = []
        for call in calls:
            if call.approval is not None and call.approval.status == ApprovalStatus.pending.value:
                call.approval.status = ApprovalStatus.cancelled.value
                call.approval.resolved_at = utcnow()
                call.approval.comment = "Session was stopped"
            call.status = ToolCallStatus.cancelled.value
            call.result = {"success": False, "error": "Cancelled because the session was stopped"}
            await append_message(
                db,
                session_id,
                role="tool",
                tool_call_id=call.id,
                content=json.dumps(call.result, ensure_ascii=False),
            )
            cancelled.append(call.id)
        return cancelled

    async def shutdown(self) -> None:
        clock = LoopClock()
        tasks = [task for task in [*self._tasks.values(), *self._finishing] if not task.done()]
        for task in tasks:
            retain(task)
            task.cancel()

        async def close_resources() -> None:
            if tasks:
                await asyncio.wait(tasks)
            for client in [*self._llm_clients.values(), *self._retired_clients]:
                if isinstance(client, LLMClient):
                    await client.aclose()
            if self.task_queue is not None:
                await self.task_queue.close()

        await bounded_cleanup(
            close_resources(), clock, clock.now() + (5 if set(tasks) & self._engine_tasks else 2)
        )

    async def resolve_approval(
        self,
        approval_id: str,
        decision: str,
        comment: str | None,
        *,
        actor_user_id: str | None = None,
        generation: int | None = None,
        executor: str | None = None,
    ) -> str:
        async with session_factory() as db:
            session_id = await db.scalar(
                select(ToolCall.session_id)
                .join(Approval, Approval.tool_call_id == ToolCall.id)
                .where(Approval.id == approval_id)
            )
            if session_id is None:
                raise LookupError("Approval not found")
            record = await db.get(Session, session_id)
            if record is not None and record.engine_profile is not None:
                from agent.engine_approvals import resolve_native

                return await resolve_native(
                    db,
                    self.settings,
                    record,
                    approval_id,
                    decision,
                    comment,
                    actor_user_id,
                    generation,
                    executor,
                )
            await db.execute(select(Session).where(Session.id == session_id).with_for_update())
            query = (
                select(Approval)
                .where(Approval.id == approval_id)
                .with_for_update()
                .options(
                    selectinload(Approval.tool_call)
                    .selectinload(ToolCall.session)
                    .selectinload(Session.project)
                )
            )
            approval = (await db.execute(query)).scalar_one_or_none()
            if approval is None:
                raise LookupError("Approval not found")
            if approval.status != ApprovalStatus.pending.value:
                raise ValueError("Approval is already resolved")

            call = approval.tool_call
            session = call.session
            if session.stop_requested or session.status == SessionStatus.stopped.value:
                raise ValueError("Session is stopped; start it again before approving tools")

            # Conditional writes: a concurrent Stop cancels the approval first and must win.
            approved = decision == "approve"
            claimed = cast(
                CursorResult[Any],
                await db.execute(
                    update(Approval)
                    .where(
                        Approval.id == approval.id,
                        Approval.status == ApprovalStatus.pending.value,
                    )
                    .values(
                        status=(
                            ApprovalStatus.approved.value
                            if approved
                            else ApprovalStatus.rejected.value
                        ),
                        comment=comment,
                        resolved_at=utcnow(),
                    )
                    .execution_options(synchronize_session=False)
                ),
            )
            if claimed.rowcount != 1:
                await db.rollback()
                raise ValueError("Approval is already resolved")
            result = (
                None if approved else {"success": False, "error": comment or "Rejected by user"}
            )
            await db.execute(
                update(ToolCall)
                .where(
                    ToolCall.id == call.id,
                    ToolCall.status == ToolCallStatus.awaiting_confirmation.value,
                )
                .values(
                    status=(
                        ToolCallStatus.pending.value if approved else ToolCallStatus.rejected.value
                    ),
                    result=result,
                )
                .execution_options(synchronize_session=False)
            )
            if result is not None:
                await append_message(
                    db,
                    session.id,
                    role="tool",
                    tool_call_id=call.id,
                    content=json.dumps(result, ensure_ascii=False),
                )
            await db.commit()

            pending = (
                await db.execute(
                    select(Approval.id)
                    .join(ToolCall)
                    .where(
                        ToolCall.session_id == session.id,
                        Approval.status == ApprovalStatus.pending.value,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if pending is not None:
                await emit_event(
                    db,
                    "approval.resolved",
                    {"approval_id": approval.id, "decision": decision},
                    session.id,
                )
                return session.id

            resumed = cast(
                CursorResult[Any],
                await db.execute(
                    update(Session)
                    .where(
                        Session.id == session.id,
                        Session.stop_requested.is_(False),
                        Session.status.not_in(TERMINAL_STATUSES),
                    )
                    .values(status=SessionStatus.pending.value)
                    .execution_options(synchronize_session=False)
                ),
            )
            await db.commit()
            if resumed.rowcount == 1:
                await emit_event(
                    db, "session.status", {"status": SessionStatus.pending.value}, session.id
                )
                await self.start(session.id, new_run=False)
            return session.id

    async def _run_guarded(
        self,
        session_id: str,
        generation: int | None = None,
        *,
        use_slot: bool = True,
        owned_fence: RunFence | None = None,
    ) -> None:
        token = CURRENT_FENCE.set(None)
        try:
            async with (
                self._semaphore
                if use_slot and session_id not in self._waited_children
                else nullcontext()
            ):
                await self._execute_run(session_id, generation, owned_fence)
        finally:
            CURRENT_FENCE.reset(token)

    async def _execute_run(
        self, session_id: str, generation: int | None, fence: RunFence | None = None
    ) -> None:
        if fence is None:
            async with session_factory() as db:
                fence = await claim(
                    db,
                    session_id,
                    lease_seconds=FENCE_LEASE_SECONDS,
                    expected_generation=generation,
                )
        if fence is None:
            return  # terminal, stopped, or another executor owns the run
        if fence.session_id != session_id or (
            generation is not None and fence.generation != generation
        ):
            raise FenceLostError("The fence does not belong to this run")
        async with session_factory() as db:
            stream_run = await resume_stream_run(db, session_id, fence.generation)
        run_guard_token = CURRENT_RUN_GUARD.set(None)
        token = CURRENT_FENCE.set(fence)
        stream_token = CURRENT_STREAMS.set(stream_run)
        run = asyncio.current_task()
        keeper = asyncio.create_task(self._keep_lease(fence, run), name=f"lease-{session_id}")
        try:
            await self._run(session_id)
        except asyncio.CancelledError as error:
            # The completed database frames can still own a SQLite cursor after
            # connection close. Release it before lease cleanup opens a writer.
            # The task's result is not retrieved until after this finally block.
            traceback.clear_frames(error.__traceback__)
            raise
        except FenceLostError:
            logger.warning("Session %s: the run lost its fence and stops", session_id)
        except BudgetExceeded as error:
            # Belt and braces: a budget boundary that escaped the run loop must stop
            # without session.error, never become a tool error or a retry.
            current_fence = CURRENT_FENCE.get()
            async with session_factory() as db:
                await self._stop_session_by_budget(
                    db,
                    session_id,
                    current_fence.generation if current_fence is not None else 0,
                    error.reason,
                    current_fence,
                )
        except Exception as exc:
            try:
                await self._mark_error(session_id, exc)
            except FenceLostError:
                logger.warning("Session %s: error not recorded, the fence is lost", session_id)
        finally:
            keeper.cancel()
            cleanup_guard = CURRENT_RUN_GUARD.get()
            CURRENT_RUN_GUARD.reset(run_guard_token)
            CURRENT_STREAMS.reset(stream_token)
            CURRENT_FENCE.reset(token)

            async def release_resources() -> None:
                with suppress(asyncio.CancelledError):
                    await keeper
                async with session_factory() as db:
                    await release(db, fence)

            try:
                if cleanup_guard is not None and cleanup_guard.closed:
                    await cleanup_guard.cleanup(release_resources())
                else:
                    clock = LoopClock()
                    await bounded_cleanup(release_resources(), clock, clock.now() + 2)
            except Exception:
                logger.exception("Session %s: run resource cleanup failed", session_id)

    async def _keep_lease(self, fence: RunFence, run: asyncio.Task[Any] | None) -> None:
        """Renew the lease; cancel the run when the lease is lost (fail closed)."""
        while True:
            await asyncio.sleep(FENCE_LEASE_SECONDS / 3)
            try:
                async with session_factory() as db:
                    db.sync_session.info["fence_exempt"] = True
                    renewed = await renew(db, fence, lease_seconds=FENCE_LEASE_SECONDS)
            except Exception:
                logger.exception("Session %s: lease renewal failed", fence.session_id)
                renewed = False
            if not renewed:
                if run is not None:
                    run.cancel()
                return

    async def _mark_error(self, session_id: str, exc: Exception) -> None:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            if session is not None:
                fence = CURRENT_FENCE.get()
                generation = fence.generation if fence is not None else session.run_generation
                pending = await finish_run(
                    db,
                    session_id,
                    generation,
                    "error",
                    error=str(exc),
                    reason=exc.kind.value if isinstance(exc, LLMError) else None,
                )
                if pending:
                    await db.commit()
                    await publish(pending)

    async def _stop_session_by_budget(
        self,
        db: Any,
        session_id: str,
        generation: int,
        reason: str,
        fence: RunFence | None = None,
        pending_events: list[PendingEvent] | None = None,
    ) -> None:
        """Stop the session for a budget reason: durable reason, conditionally stopped status.

        The fence of the current run guards every write here. Committed even when a stopped
        status was already recorded (a repeated stop just keeps the earlier status). No
        ``stop_generation`` call: it would advance the generation and hide the stop reason
        from the API.
        """
        if fence is not None:
            await fence_check(db, fence)
        pending = await finish_run(db, session_id, generation, "stopped", reason=reason)
        if not pending:
            return
        pending = [*(pending_events or []), *pending]
        await mark_budget_stop(db, session_id, generation, reason)
        cancelled = await self._cancel_unfinished_tool_calls(db, session_id)
        if cancelled:
            pending.append(
                await add_event(db, "approval.cancelled", {"tool_call_ids": cancelled}, session_id)
            )
        await db.commit()
        await publish(pending)

    async def _enforce_budget_before_step(self, db: Any, session_id: str, generation: int) -> bool:
        """Allow or deny the next inference step; deny stops the session. Return False on stop."""
        session = await db.get(Session, session_id)
        if session is None:
            return False
        now = datetime.now(UTC)
        usage = await load_budget_usage(db, session, now)
        decision = check_budget(budget_limits(self.settings, session), usage, now)
        if not decision.allowed:
            await self._stop_session_by_budget(
                db, session_id, generation, decision.reason or "budget_exceeded"
            )
            return False
        await record_step(db, session_id, generation)
        return True

    async def _check_tool_budget(self, db: Any, session_id: str) -> None:
        """Refuse a tool call when the session-level ceilings leave no room.

        Tool execution is not an inference step, so the step counter is not consulted
        (``steps=0``): tools of the last allowed step still run, while the next inference
        boundary is denied by the step check.
        """
        session = await db.get(Session, session_id)
        if session is None:
            raise BudgetExceeded("session_missing")
        now = datetime.now(UTC)
        usage = await load_budget_usage(db, session, now)
        decision = check_budget(
            budget_limits(self.settings, session),
            replace(usage, steps=0),
            now,
        )
        if not decision.allowed:
            raise BudgetExceeded(decision.reason or "budget_exceeded")

    async def _models(self, db: Any) -> ResolvedModelRegistry:
        """Read the catalogue; keep the last valid one if the new one is invalid."""
        try:
            self._last_models = await effective_registry(db, self.settings)
        except (ModelConfigError, ValidationError, ValueError):
            if self._last_models is None:
                raise
            logger.exception("Model configuration is invalid; the last valid one stays in use")
        return self._last_models

    @property
    def _tools(self) -> ToolRegistry:
        """Return the tools bound to this run, or the base registry outside a run."""
        return self._run_registry.get() or self.registry

    async def _run(self, session_id: str) -> None:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            if session is not None and session.engine_profile is not None:
                try:
                    await check_engine_access(
                        db, self.settings, session, session.created_by_user_id
                    )
                except PermissionError as exc:
                    raise RuntimeError(f"engine недоступен: {exc}") from exc
                profile_id = session.engine_profile
                profiles = configured_profiles(self.settings)
                profile = profiles.get(profile_id)
                self.engine_registry.profiles = profiles
                # Release the read transaction before the runner opens a writer.
                await db.rollback()
                fence = CURRENT_FENCE.get()
                assert fence is not None
                lock = self._engine_sessions.setdefault(session_id, asyncio.Lock())
                if profile is None or not self.engine_registry.available(profile_id):
                    raise RuntimeError("engine недоступен")
                slots = self._engine_slots.setdefault(
                    profile_id, asyncio.Semaphore(profile.process_limit)
                )
                task = asyncio.current_task()
                assert task is not None
                self._engine_tasks.add(task)
                try:
                    async with lock, slots:
                        async with session_factory() as check_db:
                            await fence_check(check_db, fence)
                        engine = self.engine_registry.get(profile_id)
                        if engine is None:
                            raise RuntimeError("engine недоступен")
                        await EngineRunner(session_factory, engine, profile, fence).run()
                finally:
                    self._engine_tasks.discard(task)
                return
            project = await db.scalar(select(Project).join(Session).where(Session.id == session_id))
        registry = self.registry
        if project is not None:
            skills = load_skill_set(Path(project.root_path), None)
            if skills.skills:
                registry = registry.with_tools((ReadSkillTool(skills),))
        if project is None:
            return
        async with source_registry(
            self.settings, session_id, Path(project.root_path).resolve(strict=True), registry
        ) as registry:
            token = self._run_registry.set(registry)
            try:
                await self._run_session(session_id)
            finally:
                self._run_registry.reset(token)

    async def _deliver_inbox(
        self, db: Any, session: Session, *, pending: list[PendingEvent]
    ) -> bool | None:
        fence = CURRENT_FENCE.get()
        if fence is None:
            return False
        if fence.session_id != session.id:
            raise FenceLostError("Automatic delivery requires the target run fence")
        await fence_check(db, fence)
        fresh = await lock_session(db, session.id)
        if fresh is None:
            return False
        if await has_eligible(db, fresh, self.settings):
            now = datetime.now(UTC)
            usage = await load_budget_usage(db, fresh, now)
            decision = check_budget(budget_limits(self.settings, fresh), usage, now)
            if not decision.allowed:
                await self._stop_session_by_budget(
                    db,
                    fresh.id,
                    fence.generation,
                    decision.reason or "budget_exceeded",
                    pending_events=pending,
                )
                return None
        delivered, events = await deliver(db, fresh.id, settings=self.settings)
        pending.extend(events)
        if delivered:
            await db.commit()
            await publish(pending)
        return delivered

    async def _reset_inference_stream(self, publisher: StreamPublisher | None) -> None:
        if publisher is not None:
            await publisher.reset()

    def _invalid_call(
        self, calls: list[dict[str, Any]], mode: str, model_ids: list[str]
    ) -> str | None:
        """Return the reason of the first invalid tool call, for the model to correct it."""
        for call in calls:
            fn = call["function"]
            try:
                self._tools.validate(
                    fn["name"], mode, arguments_object(fn["arguments"]), model_ids=model_ids
                )
            except Exception as error:
                return f"{fn['name']}: {error}"[:500]
        return None

    async def _infer_logical(
        self,
        session_id: str,
        model: ResolvedModel,
        models: ResolvedModelRegistry,
        definitions: list[dict[str, Any]],
        decisions: dict[str, ReasoningDecision],
    ) -> InferenceResult | None:
        # The worker owns this resource, including cancellation of its pending operations.
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            if session is None:
                return None
            return await self._infer_logical_work(
                db, session, model, models, definitions, decisions
            )

    async def _infer_logical_work(
        self,
        db: Any,
        session: Session,
        model: ResolvedModel,
        models: ResolvedModelRegistry,
        definitions: list[dict[str, Any]],
        decisions: dict[str, ReasoningDecision],
    ) -> InferenceResult | None:
        cut_offs = 0
        invalid_replies = 0
        session_id = session.id
        original_decisions = decisions
        while True:
            decisions = original_decisions
            retried = False
            stream_retries = 0
            while True:

                def reasoning_for(
                    candidate: ResolvedModel,
                    snapshot: dict[str, ReasoningDecision] = decisions,
                ) -> ReasoningDecision:
                    return snapshot[candidate.id]

                inference = ContextInference(
                    session.id,
                    self.settings.context_reserved_tokens,
                    definitions,
                    models,
                    partial(self._llm_for, models),
                    reasoning_for,
                    settings=self.settings,
                )
                try:
                    outcome = await inference.run(model)
                except BudgetExceeded as error:
                    await self._stop_session_by_budget(
                        db, session_id, session.run_generation or 0, error.reason
                    )
                    return None
                except LLMError as error:
                    # The stream broke after the first fragment: no tool ran and nothing
                    # was stored, so drop the partial reply and send the step again.
                    if (
                        not error.started
                        or error.kind not in TRANSIENT_KINDS
                        or stream_retries >= MAX_STREAM_RETRIES
                    ):
                        raise
                    stream_retries += 1
                    await self._reset_inference_stream(inference.publisher)
                    await emit_event(
                        db,
                        "inference.retry",
                        {
                            "reason": "stream_interrupted",
                            "model_id": model.id,
                            "retry": stream_retries,
                        },
                        session.id,
                    )
                    continue
                response = outcome.response
                reasoning_effort = outcome.reasoning.effective
                await db.refresh(session, attribute_names=["status", "stop_requested"])
                if session.stop_requested or session.status == SessionStatus.stopped.value:
                    return None
                lower = lower_effort(outcome.model, reasoning_effort)
                if retried or lower is None or not _cut_in_reasoning(response):
                    break
                # The output limit cut the reply during reasoning, before any text: keep the
                # cut reply for diagnosis and try once more with a lower reasoning level.
                retried = True
                db.add(
                    Message(
                        session_id=session.id,
                        role="assistant",
                        kind="interrupted",
                        content=response.content,
                        token_count=response.completion_tokens or None,
                        reasoning_content=response.reasoning,
                        reasoning_effort=reasoning_effort,
                    )
                )
                await db.commit()
                await self._reset_inference_stream(outcome.publisher)
                await emit_event(
                    db,
                    "inference.retry",
                    {
                        "reason": "output_limit_in_reasoning",
                        "model_id": outcome.model.id,
                        "from": reasoning_effort,
                        "to": lower,
                    },
                    session.id,
                )
                decisions = _lowered(decisions, models)

            # Our own globally unique IDs: providers may omit or reuse theirs, and the
            # step counter restarts whenever the loop resumes after an approval.
            normalized_calls = [
                {**dict(raw_call), "id": f"call_{uuid.uuid4().hex}"}
                for raw_call in (response.interrupted_tool_calls or response.tool_calls)
            ]
            if response.finish_reason == "length":
                # The output limit cut the reply. It is not a result: keep the partial text
                # for diagnosis (kind "interrupted" never reaches the model), run no tool
                # call from it, and stop with advice. No hidden second request.
                db.add(
                    Message(
                        session_id=session.id,
                        role="assistant",
                        kind="interrupted",
                        content=response.content,
                        tool_calls=[_history_safe_call(c) for c in normalized_calls] or None,
                        token_count=response.completion_tokens or None,
                        reasoning_content=response.reasoning,
                        reasoning_effort=reasoning_effort,
                    )
                )
                cut_offs += 1
                if cut_offs > MAX_CUT_OFF_REPLIES:
                    await db.commit()
                    raise ToolError(
                        "The model reply was cut off by the output limit, so it was not "
                        "used and no tool ran. Ask for smaller steps, for example write a "
                        "large file in parts."
                    )
                # Tell the model and let it take a smaller step in the next inference.
                await append_message(db, session.id, role="user", content=CUT_OFF_NOTICE)
                await db.commit()
                await self._reset_inference_stream(outcome.publisher)
                await emit_event(
                    db,
                    "inference.retry",
                    {"reason": "output_limit", "model_id": outcome.model.id, "retry": cut_offs},
                    session.id,
                )
                allowed = await self._enforce_budget_before_step(
                    db, session.id, session.run_generation or 0
                )
                await db.commit()
                if not allowed:
                    return None
                continue
            problem = self._invalid_call(response.tool_calls, session.mode, models.ids())
            if problem is not None:
                # Like a cut-off reply: keep it for diagnosis, run nothing from it, and let
                # the model correct the call in the next inference.
                db.add(
                    Message(
                        session_id=session.id,
                        role="assistant",
                        kind="interrupted",
                        content=response.content,
                        tool_calls=[_history_safe_call(c) for c in normalized_calls] or None,
                        token_count=response.completion_tokens or None,
                        reasoning_content=response.reasoning,
                        reasoning_effort=reasoning_effort,
                    )
                )
                invalid_replies += 1
                if invalid_replies > MAX_INVALID_CALL_REPLIES:
                    await db.commit()
                    raise guard_error(
                        LLMErrorKind.invalid_response, "Некорректный вызов инструмента"
                    )
                await append_message(
                    db, session.id, role="user", content=f"{INVALID_CALL_NOTICE} Причина: {problem}"
                )
                await db.commit()
                await self._reset_inference_stream(outcome.publisher)
                await emit_event(
                    db,
                    "inference.retry",
                    {
                        "reason": "invalid_tool_call",
                        "model_id": outcome.model.id,
                        "retry": invalid_replies,
                    },
                    session.id,
                )
                allowed = await self._enforce_budget_before_step(
                    db, session.id, session.run_generation or 0
                )
                await db.commit()
                if not allowed:
                    return None
                continue
            return outcome

    async def _run_session(self, session_id: str) -> None:
        # One catalogue snapshot for the whole execution: a YAML or UI change applies to the
        # next execution, not between two steps of a running one.
        models: ResolvedModelRegistry | None = None
        while True:
            async with session_factory() as db:
                fence = CURRENT_FENCE.get()
                if fence is not None:
                    for attempt in range(3):
                        try:
                            await fence_check(db, fence)
                            break
                        except DBAPIError as error:
                            await db.rollback()
                            if not retryable(error) or attempt == 2:
                                raise
                query = (
                    select(Session)
                    .where(Session.id == session_id)
                    .options(
                        selectinload(Session.project),
                    )
                )
                session = (await db.execute(query)).scalar_one_or_none()
                if session is None or session.status in TERMINAL_STATUSES:
                    return
                rows = await load_context_rows(db, session_id)
                attributes.set_committed_value(session, "messages", rows)
                unfinished = list(
                    await db.scalars(
                        select(ToolCall)
                        .where(
                            ToolCall.session_id == session_id,
                            ToolCall.status.in_({"pending", "running", "awaiting_confirmation"}),
                        )
                        .options(selectinload(ToolCall.approval))
                        .order_by(ToolCall.created_at, ToolCall.id)
                    )
                )
                attributes.set_committed_value(session, "tool_calls", unfinished)
                inbox_pending: list[PendingEvent] = []
                if not unfinished:
                    delivered = await self._deliver_inbox(db, session, pending=inbox_pending)
                    if delivered is None:
                        return
                    if delivered:
                        continue
                visible = [
                    message
                    for message in session.messages
                    if message.kind == "normal" and not message.skipped
                ]
                if (
                    session.status == SessionStatus.running.value
                    and visible
                    and visible[-1].role == "assistant"
                    and not visible[-1].tool_calls
                    and not unfinished
                ):
                    pending = await finish_run(db, session_id, session.run_generation, "completed")
                    await db.commit()
                    await publish([*inbox_pending, *pending])
                    return
                if session.status == SessionStatus.awaiting_confirmation.value:
                    pending_approval = any(
                        call.approval is not None
                        and call.approval.status == ApprovalStatus.pending.value
                        for call in session.tool_calls
                    )
                    if pending_approval:
                        return

                try:
                    resume_state = await self._resume_tool_calls(db, session)
                except BudgetExceeded as error:
                    await self._stop_session_by_budget(
                        db, session_id, session.run_generation or 0, error.reason
                    )
                    return
                if resume_state in {"waiting", "stopped"}:
                    return
                if resume_state == "processed":
                    continue

                # Conditional write: a Stop that landed after the SELECT above must win.
                claimed = cast(
                    CursorResult[Any],
                    await db.execute(
                        update(Session)
                        .where(Session.id == session.id, Session.status.not_in(TERMINAL_STATUSES))
                        .values(status=SessionStatus.running.value, error=None)
                    ),
                )
                await db.commit()
                await publish(inbox_pending)
                if claimed.rowcount == 0:
                    return
                await emit_event(db, "session.status", {"status": session.status}, session.id)

                # The generation of the current run: inference step accounting lives per
                # run, while tokens and money stay session-wide.
                generation = session.run_generation or 0
                try:
                    allowed = await self._enforce_budget_before_step(db, session_id, generation)
                except BudgetExceeded as error:
                    await self._stop_session_by_budget(db, session_id, generation, error.reason)
                    return
                if not allowed:
                    return
                await db.commit()

                if models is None:
                    models = await self._models(db)
                model = select_model(models, session.llm_profile, strict=False)
                definitions = self._tools.definitions(session.mode, model_ids=models.ids())
                decisions = {
                    candidate.id: self._reasoning_for(session, candidate)
                    for candidate in models.models.values()
                }
                await db.commit()

                limits = budget_limits(self.settings, session)
                usage = await load_budget_usage(db, session, datetime.now(UTC))
                session_deadline = None
                if limits.deadline_seconds is not None and usage.started_at is not None:
                    remaining = (
                        limits.deadline_seconds
                        - (datetime.now(UTC) - usage.started_at).total_seconds()
                    )
                    session_deadline = asyncio.get_running_loop().time() + max(0, remaining)
                await db.commit()
                guard = GuardContext(model, session_deadline=session_deadline)
                run_guard_token = CURRENT_RUN_GUARD.set(guard)
                try:
                    outcome = await guard.run(
                        self._infer_logical(
                            session.id,
                            model,
                            models,
                            definitions,
                            decisions,
                        )
                    )
                except LLMError as error:
                    guard.close(error)
                    try:
                        await guard.cleanup(self._mark_error(session_id, error))
                    except Exception:
                        logger.exception("Session %s: terminal outcome cleanup failed", session_id)
                    return
                except BudgetExceeded as error:
                    await guard.cleanup(
                        self._stop_session_by_budget(
                            db,
                            session_id,
                            session.run_generation or 0,
                            error.reason,
                        )
                    )
                    return
                guard.close()
                CURRENT_RUN_GUARD.reset(run_guard_token)
                if outcome is None:
                    return
                response = outcome.response
                reasoning_effort = outcome.reasoning.effective
                normalized_calls = [
                    {**dict(raw_call), "id": f"call_{uuid.uuid4().hex}"}
                    for raw_call in response.tool_calls
                ]

                assistant = await append_message(
                    db,
                    session.id,
                    role="assistant",
                    content=response.content,
                    # The provider rejects a history with invalid JSON arguments, so the stored
                    # copy carries "{}" for them; the tool result reports the real problem.
                    tool_calls=[_history_safe_call(call) for call in normalized_calls] or None,
                    token_count=response.completion_tokens or None,
                    reasoning_content=response.reasoning,
                    reasoning_effort=reasoning_effort,
                )

                if not normalized_calls:
                    pending = [
                        await add_event(
                            db,
                            "message.created",
                            {
                                "role": "assistant",
                                "message_id": assistant.id,
                                **(
                                    outcome.publisher.identity
                                    if outcome.publisher is not None
                                    else {}
                                ),
                            },
                            session.id,
                        )
                    ]
                    delivered = await self._deliver_inbox(db, session, pending=pending)
                    if delivered is None:
                        return
                    if delivered:
                        continue
                    pending.extend(
                        await finish_run(db, session_id, session.run_generation, "completed")
                    )
                    await db.commit()
                    await publish(pending)
                    return

                requires_approval = False
                truncated = response.finish_reason == "length"
                # The project root lets a tool prove that a local command stays inside it.
                try:
                    project_root: Path | None = Path(session.project.root_path).resolve(strict=True)
                except (OSError, RuntimeError):
                    project_root = None
                for raw_call in normalized_calls:
                    function = raw_call.get("function", {})
                    name = str(function.get("name", ""))
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError
                    except (TypeError, ValueError, json.JSONDecodeError):
                        arguments = {"_invalid_arguments": function.get("arguments")}

                    call_id = str(raw_call["id"])
                    try:
                        if "_invalid_arguments" in arguments:
                            raise ToolError(
                                "Tool arguments are not valid JSON"
                                + (" (the reply was cut off)" if truncated else "")
                            )
                        self._tools.validate(name, session.mode, arguments, model_ids=models.ids())
                        # One decision: the stored risk and the approval must never disagree.
                        risk = self._tools.risk(name, session.mode, arguments, project_root)
                        confirmation = risk is not RiskLevel.read_only
                        revision = capture_source_revision(self._tools.get(name, session.mode))
                    except Exception as exc:
                        call = ToolCall(
                            id=call_id,
                            session_id=session.id,
                            name=name,
                            arguments=arguments,
                            risk_level="unavailable",
                            status=ToolCallStatus.error.value,
                            result={"success": False, "error": str(exc)},
                        )
                        db.add(call)
                        await append_message(
                            db,
                            session.id,
                            role="tool",
                            tool_call_id=call_id,
                            content=json.dumps(call.result, ensure_ascii=False),
                        )
                        continue

                    call = ToolCall(
                        id=call_id,
                        session_id=session.id,
                        name=name,
                        arguments=arguments,
                        risk_level=risk.value,
                        source_revision=revision,
                        status=(
                            ToolCallStatus.awaiting_confirmation.value
                            if confirmation
                            else ToolCallStatus.pending.value
                        ),
                    )
                    db.add(call)
                    await db.flush()

                    if confirmation:
                        db.add(Approval(tool_call_id=call.id))
                        requires_approval = True

                await db.commit()
                await emit_event(
                    db,
                    "message.created",
                    {
                        "role": "assistant",
                        "message_id": assistant.id,
                        **(outcome.publisher.identity if outcome.publisher is not None else {}),
                    },
                    session.id,
                )
                if requires_approval:
                    session.status = SessionStatus.awaiting_confirmation.value
                    await db.commit()
                    await emit_event(
                        db, "approval.required", {"status": session.status}, session.id
                    )
                    return

    async def _resume_tool_calls(self, db: Any, session: Session) -> str:
        processed = False
        waiting = False
        unfinished = [
            call
            for call in session.tool_calls
            if call.status
            in {
                ToolCallStatus.pending.value,
                ToolCallStatus.running.value,
                ToolCallStatus.awaiting_confirmation.value,
            }
        ]
        index = 0
        while index < len(unfinished):
            call = unfinished[index]
            index += 1
            await db.refresh(session, attribute_names=["status", "stop_requested"])
            if session.stop_requested or session.status == SessionStatus.stopped.value:
                return "stopped"
            if call.status == ToolCallStatus.running.value:
                if call.name == "message_to_session":
                    fence = CURRENT_FENCE.get()
                    if fence is None or fence.session_id != session.id:
                        raise FenceLostError("Message recovery requires the sender run fence")
                    await fence_check(db, fence)
                    await db.refresh(call)
                    if call.status != "running":
                        continue
                    receipt_id = await db.scalar(
                        select(SessionInbox.id).where(SessionInbox.dedupe_key == f"tool:{call.id}")
                    )
                    if receipt_id is not None:
                        await self._store_tool_result(
                            db,
                            session,
                            call,
                            {
                                "success": True,
                                "output": {"inbox_id": receipt_id, "status": "queued"},
                            },
                        )
                        processed = True
                        continue
                call.status = ToolCallStatus.error.value
                call.result = {
                    "success": False,
                    "error": (
                        "Worker stopped while this tool was running; it was not retried "
                        "to avoid repeating a possible side effect"
                    ),
                }
                await append_message(
                    db,
                    session.id,
                    role="tool",
                    tool_call_id=call.id,
                    content=json.dumps(call.result, ensure_ascii=False),
                )
                await db.commit()
                processed = True
                continue

            approval = call.approval
            if approval is not None and approval.status == ApprovalStatus.pending.value:
                waiting = True
                continue
            if approval is None or approval.status == ApprovalStatus.approved.value:
                group = [call]
                if self._may_run_in_parallel(session, call):
                    group += self._parallel_followers(session, unfinished[index:])
                if len(group) > 1:
                    index += len(group) - 1
                    if not await self._execute_parallel(db, session, group):
                        return "stopped"
                elif not await self._execute_tool_call(db, session, call):
                    return "stopped"
                processed = True

        if waiting:
            session.status = SessionStatus.awaiting_confirmation.value
            await db.commit()
            return "waiting"
        return "processed" if processed else "ready"

    async def _claim_tool_call(self, db: Any, call: ToolCall) -> bool:
        """Mark a queued call as running unless Stop cancelled it or its session."""
        stopped = (
            select(Session.id)
            .where(
                Session.id == call.session_id,
                or_(
                    Session.stop_requested.is_(True),
                    Session.status == SessionStatus.stopped.value,
                ),
            )
            .exists()
        )
        claimed = cast(
            CursorResult[Any],
            await db.execute(
                update(ToolCall)
                .where(
                    ToolCall.id == call.id,
                    ToolCall.status == ToolCallStatus.pending.value,
                    ~stopped,
                )
                .values(status=ToolCallStatus.running.value)
                .execution_options(synchronize_session=False)
            ),
        )
        await db.commit()
        await db.refresh(call, attribute_names=["status"])
        return claimed.rowcount == 1

    def _parallel_followers(self, session: Session, rest: list[ToolCall]) -> list[ToolCall]:
        """Return the calls right after a parallel-safe call that may join its group.

        A follower is a queued call without an approval whose tool is parallel safe.
        The group stops at the first call that is not.
        """
        followers: list[ToolCall] = []
        for call in rest:
            if not self._may_run_in_parallel(session, call):
                break
            followers.append(call)
        return followers

    def _may_run_in_parallel(self, session: Session, call: ToolCall) -> bool:
        if (
            call.status != ToolCallStatus.pending.value
            or call.approval is not None
            or call.source_revision is not None
        ):
            return False
        try:
            tool = self._tools.get(call.name, session.mode)
            return capture_source_revision(tool) is None and is_parallel_safe(tool)
        except Exception:
            return False

    def _tool_context(self, session: Session) -> ToolContext:
        return ToolContext(
            session_id=session.id,
            project_root=Path(session.project.root_path).resolve(strict=True),
            depth=int((session.configuration or {}).get("depth", 0)),
            audit_egress=partial(self._audit_egress, session.id),
            spawn_child=partial(self._spawn_child, session),
            sandbox=policy_from_settings(self.settings),
        )

    async def _execute_parallel(self, db: Any, session: Session, group: list[ToolCall]) -> bool:
        """Run a group of parallel-safe calls at the same time; store results in call order.

        Only the tool bodies run concurrently. Claims and result rows use the one database
        session in sequence. Returns False when Stop prevented a claim; the claimed calls
        still run to the end and keep their results.
        """
        claimed: list[ToolCall] = []
        for call in group:
            # Every member is checked before it is claimed; a refusal aborts the whole
            # group before any tool body runs.
            await self._check_tool_budget(db, session.id)
            if not await self._claim_tool_call(db, call):
                break
            claimed.append(call)
        if claimed:
            context = self._tool_context(session)
            executable: list[ToolCall] = []
            tools: dict[str, Tool] = {}
            for call in claimed:
                try:
                    tool = self._tools.get(call.name, session.mode)
                    if capture_source_revision(tool) is not None:
                        raise ToolError("Source-backed tools require sequential execution")
                    await validate_source_execution(db, session.id, tool, call.source_revision)
                except Exception as exc:
                    await self._store_tool_result(
                        db, session, call, {"success": False, "error": str(exc)}
                    )
                else:
                    executable.append(call)
                    tools[call.id] = tool

            async def execute(item: BatchCall) -> Any:
                return await tools[item.call_id].execute(context, item.arguments)

            results = await run_batch(
                [BatchCall(call.id, call.name, dict(call.arguments)) for call in executable],
                execute,
                max_concurrency=PARALLEL_TOOL_LIMIT,
                timeout_seconds=None,
            )
            for call, outcome in zip(executable, results, strict=True):
                await self._store_tool_result(db, session, call, _batch_result(outcome))
        return len(claimed) == len(group)

    async def _source_execution_ready(
        self, db: Any, session: Session, call: ToolCall, tool: Tool
    ) -> bool:
        try:
            await validate_source_execution(db, session.id, tool, call.source_revision)
            fence = CURRENT_FENCE.get()
            if fence is not None:
                await fence_check(db, fence)
                await db.commit()
            # A waited child runs under the fence of its parent, so check its own Stop too.
            if fence is None or fence.session_id != session.id:
                await db.refresh(session, attribute_names=["status", "stop_requested"])
                if session.stop_requested or session.status == SessionStatus.stopped.value:
                    return False
                if isinstance(tool, McpBackedTool):
                    await db.commit()
        except FenceLostError:
            await db.rollback()
            raise
        return True

    async def _enqueue_message(
        self, db: Any, session: Session, call_id: str, target_id: str, content: str
    ) -> dict[str, Any]:
        receipt = await enqueue(db, session.id, target_id, content, f"tool:{call_id}")
        status = await db.scalar(select(ToolCall.status).where(ToolCall.id == call_id))
        if status != "running":
            raise FenceLostError("Message call is no longer running")
        return {"inbox_id": receipt.id, "status": "queued"}

    async def _execute_message_call(self, db: Any, session: Session, call: ToolCall) -> bool:
        session_id, call_id = session.id, call.id
        for attempt in range(3):
            try:
                fence = CURRENT_FENCE.get()
                if fence is None or fence.session_id != session_id:
                    raise FenceLostError("Message posting requires the sender run fence")
                await fence_check(db, fence)
                session = await db.scalar(
                    select(Session)
                    .where(Session.id == session_id)
                    .options(selectinload(Session.project))
                    .execution_options(populate_existing=True)
                )
                call = await db.scalar(
                    select(ToolCall)
                    .where(ToolCall.id == call_id)
                    .options(selectinload(ToolCall.approval))
                    .execution_options(populate_existing=True)
                )
                if call.status != "running":
                    return False
                tool = self._tools.get(call.name, session.mode)
                context = replace(
                    self._tool_context(session),
                    post_message=partial(
                        self._enqueue_message,
                        db,
                        session,
                        call.id,
                    ),
                )
                try:
                    output = await tool.execute(context, dict(call.arguments))
                    result = {"success": True, "output": output}
                except (ToolError, ValueError) as error:
                    result = {"success": False, "error": str(error)}
                await self._store_tool_result(db, session, call, result)
                self._reconcile_signal.set()
                return True
            except DBAPIError as error:
                await db.rollback()
                if not retryable(error):
                    raise
                if attempt == 2:
                    # Rollback expires ORM attributes. Reload under the sender fence before
                    # recording the failure; the locked target is no longer needed.
                    assert fence is not None
                    await fence_check(db, fence)
                    session = await db.get(Session, session_id)
                    call = await db.get(ToolCall, call_id)
                    await self._store_tool_result(
                        db,
                        session,
                        call,
                        {"success": False, "error": "Inbox temporarily busy; retry the message"},
                    )
                    return True
                await asyncio.sleep(0.02 * (attempt + 1))
        return False

    async def _execute_tool_call(self, db: Any, session: Session, call: ToolCall) -> bool:
        await self._check_tool_budget(db, session.id)
        if not await self._claim_tool_call(db, call):
            return False
        if call.name == "message_to_session":
            return await self._execute_message_call(db, session, call)
        try:
            tool = self._tools.get(call.name, session.mode)
            if not await self._source_execution_ready(db, session, call, tool):
                return False
            context = self._tool_context(session)
            if isinstance(tool, McpBackedTool):

                async def before_send() -> None:
                    if not await self._source_execution_ready(db, session, call, tool):
                        raise ToolError("Session stopped before MCP egress")

                output = await execute_mcp_attempt(
                    tool,
                    context,
                    dict(call.arguments),
                    tool_call_id=call.id,
                    before_send=before_send,
                )
            else:
                output = await tool.execute(context, dict(call.arguments))
            result = {"success": True, "output": output}
        except FenceLostError:
            raise
        except BudgetExceeded:
            raise
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        await self._store_tool_result(db, session, call, result)
        return True

    async def _store_tool_result(
        self, db: Any, session: Session, call: ToolCall, result: dict[str, Any]
    ) -> None:
        if call.name == "message_to_session":
            fence = CURRENT_FENCE.get()
            if fence is None or fence.session_id != session.id:
                raise FenceLostError("Message results require the sender run fence")
            await fence_check(db, fence)
            status = await db.scalar(select(ToolCall.status).where(ToolCall.id == call.id))
            if status != "running":
                raise FenceLostError("Message call is no longer running")
        call.status = (
            ToolCallStatus.completed.value if result["success"] else ToolCallStatus.error.value
        )
        call.result = result
        await append_message(
            db,
            session.id,
            role="tool",
            tool_call_id=call.id,
            content=json.dumps(result, ensure_ascii=False),
        )
        event = await add_event(
            db,
            "tool.completed",
            {"tool_call_id": call.id, "name": call.name, "status": call.status},
            session.id,
        )
        await db.commit()
        await publish([event])

    @staticmethod
    def _reasoning_for(session: Session, model: ResolvedModel) -> ReasoningDecision:
        """Message override, then the session level, then "auto" or the model default."""
        messages = [
            message
            for message in session.messages
            if message.kind == "normal" and not message.skipped
        ]
        users = [message for message in messages if message.role == "user"]
        recent_errors = 0
        for message in reversed(messages):
            if message.role != "tool":
                if message.role == "assistant" and message.tool_calls:
                    continue
                break
            if '"success": false' in (message.content or ""):
                recent_errors += 1
            else:
                break
        return decide(
            model,
            message_effort=users[-1].reasoning_effort if users else None,
            session_effort=(session.configuration or {}).get("reasoning_effort"),
            mode=session.mode,
            task_chars=len(users[0].content or "") if users else 0,
            recent_errors=recent_errors,
        )

    def context_for(self, model: ResolvedModel | None = None) -> ContextManager:
        return ContextManager(self.settings.context_reserved_tokens)

    def _llm_for(self, models: ResolvedModelRegistry, model: ResolvedModel) -> LLMClient:
        if model.id == models.default_id and self.llm is not None:
            return self.llm
        client = self._llm_clients.get(model.id)
        if isinstance(client, LLMClient) and client.model != model:
            # The catalogue changed (YAML edit or UI override): use a new client.
            self._retired_clients.append(client)
            client = None
        if client is None:
            client = LLMClient(model)
            self._llm_clients[model.id] = client
        return client

    async def _audit_egress(
        self,
        session_id: str,
        category: str,
        destination: str,
        operation: str,
        payload: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        encoded = payload.encode("utf-8", errors="replace")
        async with session_factory() as db:
            db.add(
                OutboundAudit(
                    session_id=session_id,
                    category=category,
                    destination=destination,
                    operation=operation,
                    payload_bytes=len(encoded),
                    payload_sha256=hashlib.sha256(encoded).hexdigest(),
                    status=status,
                    detail=detail[:2000] if detail else None,
                )
            )
            await db.commit()

    async def _spawn_child(
        self,
        parent: Session,
        mode: str,
        prompt: str,
        wait: bool,
        llm_profile: str | None,
    ) -> dict[str, Any]:
        depth = int((parent.configuration or {}).get("depth", 0))
        if depth >= self.settings.max_child_depth:
            raise RuntimeError(f"Maximum child depth is {self.settings.max_child_depth}")
        if mode not in ROLE_PROFILES:
            raise ValueError(f"Unsupported child mode: {mode}")
        async with session_factory() as db:
            parent_fence = CURRENT_FENCE.get()
            if parent_fence is not None:
                if parent_fence.session_id != parent.id:
                    raise FenceLostError("Child creation requires the parent run fence")
                await fence_check(db, parent_fence)
            else:
                locked = await db.execute(
                    update(Session)
                    .where(
                        Session.id == parent.id,
                        Session.run_generation == parent.run_generation,
                        Session.status.not_in(TERMINAL_STATUSES),
                        Session.stop_requested.is_(False),
                    )
                    .values(run_generation=Session.run_generation)
                    .returning(Session.id)
                    .execution_options(synchronize_session=False)
                )
                if locked.one_or_none() is None:
                    raise FenceLostError("The parent run is no longer active")
            selected_profile = select_model(
                await effective_registry(db, self.settings), llm_profile or parent.llm_profile
            ).id
            project = await db.get(Project, parent.project_id)
            if project is None:
                raise LookupError("Parent project not found")
            child = await create_session_record(
                db,
                self.settings,
                project,
                prompt,
                mode,
                llm_profile=selected_profile,
                title=f"Child: {prompt.strip().replace(chr(10), ' ')[:110]}",
                parent_id=parent.id,
                actor_user_id=parent.created_by_user_id,
                configuration={"depth": depth + 1},
                run_generation=1,
                commit=False,
            )
            child_id = child.id
            fresh_parent = await db.get(Session, parent.id, populate_existing=True)
            assert fresh_parent is not None
            child.inbox_hop = fresh_parent.inbox_hop
            child.notify_generation = None if wait else 1
            events: list[PendingEvent] = []
            if wait and fresh_parent.status != SessionStatus.running.value:
                fresh_parent.status = SessionStatus.running.value
                events.append(
                    await add_event(db, "session.status", {"status": "running"}, parent.id)
                )
            created_event = await add_event(
                db,
                "session.child_created",
                {"child_id": child_id, "parent_id": parent.id, "wait": wait},
                parent.id,
            )
            child_fence = None
            if wait:
                child_fence = await claim(
                    db, child_id, lease_seconds=FENCE_LEASE_SECONDS, expected_generation=1
                )
                if child_fence is None:
                    raise FenceLostError("The new child run could not be claimed")
            else:
                await db.commit()
            await publish([*events, created_event])

        if wait:
            self._waited_children.add(child_id)
            # Register the waited run, so that a Stop of the child cancels it.
            task = asyncio.create_task(
                self._run_guarded(child_id, 1, use_slot=False, owned_fence=child_fence),
                name=f"agent-child-{child_id}",
            )
            self._tasks[child_id] = task
            self._task_generation[child_id] = 1
            try:
                await self._wait_child_terminal(parent.id, child_id, task)
            except BudgetExceeded:
                await self.stop_scoped(child_id, 1)
                raise
            except asyncio.CancelledError:
                await self.stop_scoped(child_id, 1)
                raise
            finally:
                self._waited_children.discard(child_id)
                if self._tasks.get(child_id) is task and task.done():
                    del self._tasks[child_id]
        else:
            await self.start(child_id, new_run=False, expected_generation=1)

        async with session_factory() as db:
            child_record = await db.get(Session, child_id)
            answers = (
                await db.scalars(
                    select(Message)
                    .where(
                        Message.session_id == child_id,
                        Message.role == "assistant",
                        Message.run_generation == 1,
                        Message.kind == "normal",
                        Message.skipped.is_(False),
                    )
                    .order_by(Message.id.desc())
                )
            ).all()
            final_message = next(
                (row for row in answers if not row.tool_calls and (row.content or "").strip()),
                None,
            )
            status = child_record.status if child_record else "missing"
            outcome = await db.get(SessionRunOutcome, (child_id, 1))
            if outcome is not None:
                status = outcome.status
            return {
                "session_id": child_id,
                "status": status,
                "llm_profile": child_record.llm_profile if child_record else selected_profile,
                "response": final_message.content if final_message else None,
            }

    async def _wait_child_terminal(
        self, parent_id: str, child_id: str, initial: asyncio.Task[None]
    ) -> None:
        """Wait across approvals without returning an unfinished tool call as an answer."""
        loop = asyncio.get_running_loop()
        remaining = float(self.settings.max_child_wait_seconds)
        checked_at = loop.time()
        paused = False
        async with broker.subscribe() as updates:
            while True:
                async with session_factory() as db:
                    child = await db.get(Session, child_id)
                    outcome = await db.get(SessionRunOutcome, (child_id, 1))
                    await self._check_tool_budget(db, parent_id)
                if outcome is not None:
                    return
                if child is None or child.run_generation != 1:
                    raise RuntimeError("The waited child run is no longer available")
                if child.status in TERMINAL_STATUSES:
                    return
                current = self._tasks.get(child_id, initial)
                if current.done() and not current.cancelled():
                    failure = current.exception()
                    if failure is not None:
                        raise failure
                now = loop.time()
                awaiting_approval = child.status == SessionStatus.awaiting_confirmation.value
                if not paused and not awaiting_approval:
                    remaining -= now - checked_at
                checked_at = now
                paused = awaiting_approval
                if not paused and remaining <= 0:
                    await self.stop_scoped(child_id, 1)
                    raise RuntimeError("Child agent exceeded its wait timeout")
                # Events wake the waiter promptly. A slow fallback also covers a lost
                # publish and another worker, without polling human approvals in a loop.
                delay = 1.0 if paused else min(0.5, max(0, remaining))
                try:
                    async with asyncio.timeout(delay):
                        while True:
                            event = await updates.get()
                            if event.get("session_id") in {child_id, parent_id} and event.get(
                                "type"
                            ) in {
                                "session.status",
                                "session.error",
                                "approval.required",
                                "approval.resolved",
                                "approval.cancelled",
                            }:
                                break
                except TimeoutError:
                    pass
