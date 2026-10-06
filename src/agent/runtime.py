from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import selectinload

from agent.config import Settings
from agent.context import ContextManager
from agent.database import session_factory
from agent.events import StreamPublisher, emit_event
from agent.llm import LLMClient, LLMResponse, LLMTransientError
from agent.model_config import ModelConfigError, ResolvedModel, ResolvedModelRegistry
from agent.model_registry import effective_registry, select_model
from agent.models import (
    Approval,
    ApprovalStatus,
    Message,
    OutboundAudit,
    Project,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
    utcnow,
)
from agent.queue import RedisTaskQueue
from agent.reasoning import ReasoningDecision, decide
from agent.session_service import create_session_record
from agent.tools import ToolRegistry
from agent.tools.base import ToolContext, ToolError

logger = logging.getLogger(__name__)

TRUNCATION_RETRIES = 2
CONTEXT_SAFETY_TOKENS = 512
# Raw DeepSeek tool-call markup that shows up in the text when the output budget runs out.
LEAKED_TOOL_MARKUP = ("<｜DSML｜", "< | DSML |", "<｜tool▁calls▁begin｜>")

TERMINAL_STATUSES = {
    SessionStatus.completed.value,
    SessionStatus.stopped.value,
    SessionStatus.error.value,
}


class AgentSupervisor:
    def __init__(self, settings: Settings, registry: ToolRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ToolRegistry(settings=settings)
        # Optional override for the default model's client (used by tests and fakes).
        self.llm: LLMClient | None = None
        self._llm_clients: dict[str, LLMClient] = {}
        # Replaced clients can still serve a running request; close them at shutdown only.
        self._retired_clients: list[LLMClient] = []
        self._last_models: ResolvedModelRegistry | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore = asyncio.Semaphore(settings.max_parallel_sessions)
        self.task_queue = RedisTaskQueue(settings) if settings.execution_mode == "redis" else None

    async def start(self, session_id: str) -> bool:
        if self.task_queue is not None:
            await self.task_queue.ensure_group()
            return await self.task_queue.enqueue(session_id)
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            return False
        self._tasks[session_id] = asyncio.create_task(
            self._run_guarded(session_id), name=f"agent-session-{session_id}"
        )
        return True

    async def run_job(self, session_id: str) -> None:
        await self._run_guarded(session_id)

    async def infrastructure_status(self) -> dict[str, Any]:
        if self.task_queue is None:
            return {"execution_mode": "embedded", "redis": "disabled", "workers": 0}
        status = await self.task_queue.health()
        return {"execution_mode": "redis", **status}

    async def stop(self, session_id: str) -> None:
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        async with session_factory() as db:
            children = list(
                (
                    await db.execute(
                        select(Session.id).where(
                            Session.parent_id == session_id,
                            Session.status.not_in(TERMINAL_STATUSES),
                        )
                    )
                ).scalars()
            )
            session = await db.get(Session, session_id)
            if session is not None:
                session.status = SessionStatus.stopped.value
                session.stop_requested = True
                cancelled = await self._cancel_unfinished_tool_calls(db, session_id)
                await db.commit()
                await emit_event(db, "session.status", {"status": session.status}, session_id)
                if cancelled:
                    await emit_event(
                        db, "approval.cancelled", {"tool_call_ids": cancelled}, session_id
                    )

        for child_id in children:
            await self.stop(child_id)

    @staticmethod
    async def _cancel_unfinished_tool_calls(db: Any, session_id: str) -> list[str]:
        """Withdraw queued and awaiting calls so nothing approved before Stop runs later."""
        calls = (
            await db.execute(
                select(ToolCall)
                .where(
                    ToolCall.session_id == session_id,
                    ToolCall.status.in_(
                        {
                            ToolCallStatus.pending.value,
                            ToolCallStatus.awaiting_confirmation.value,
                        }
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
            db.add(
                Message(
                    session_id=session_id,
                    role="tool",
                    tool_call_id=call.id,
                    content=json.dumps(call.result, ensure_ascii=False),
                )
            )
            cancelled.append(call.id)
        return cancelled

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for client in [*self._llm_clients.values(), *self._retired_clients]:
            if isinstance(client, LLMClient):
                await client.aclose()
        if self.task_queue is not None:
            await self.task_queue.close()

    async def resolve_approval(
        self, approval_id: str, decision: str, comment: str | None
    ) -> str:
        async with session_factory() as db:
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
                db.add(
                    Message(
                        session_id=session.id,
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(result, ensure_ascii=False),
                    )
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
                await self.start(session.id)
            return session.id

    async def _run_guarded(self, session_id: str) -> None:
        async with self._semaphore:
            try:
                await self._run(session_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._mark_error(session_id, exc)

    async def _mark_error(self, session_id: str, exc: Exception) -> None:
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            if session is not None:
                session.status = SessionStatus.error.value
                session.error = str(exc)
                await db.commit()
                await emit_event(
                    db,
                    "session.error",
                    {"status": session.status, "error": str(exc)},
                    session_id,
                )

    async def _models(self, db: Any) -> ResolvedModelRegistry:
        """Read the catalogue; keep the last valid one if the new one is invalid."""
        try:
            self._last_models = await effective_registry(db, self.settings)
        except (ModelConfigError, ValidationError, ValueError):
            if self._last_models is None:
                raise
            logger.exception("Model configuration is invalid; the last valid one stays in use")
        return self._last_models

    async def _run(self, session_id: str) -> None:
        # One catalogue snapshot for the whole execution: a YAML or UI change applies to the
        # next execution, not between two steps of a running one.
        models: ResolvedModelRegistry | None = None
        for _step in range(self.settings.max_inference_steps):
            async with session_factory() as db:
                query = (
                    select(Session)
                    .where(Session.id == session_id)
                    .options(
                        selectinload(Session.project),
                        selectinload(Session.messages),
                        selectinload(Session.tool_calls).selectinload(ToolCall.approval),
                    )
                )
                session = (await db.execute(query)).scalar_one_or_none()
                if session is None or session.status in TERMINAL_STATUSES:
                    return
                if (
                    session.status == SessionStatus.running.value
                    and session.messages
                    and session.messages[-1].role == "assistant"
                    and not session.messages[-1].tool_calls
                ):
                    session.status = SessionStatus.completed.value
                    await db.commit()
                    await emit_event(db, "session.status", {"status": session.status}, session.id)
                    return
                if session.status == SessionStatus.awaiting_confirmation.value:
                    pending_approval = any(
                        call.approval is not None
                        and call.approval.status == ApprovalStatus.pending.value
                        for call in session.tool_calls
                    )
                    if pending_approval:
                        return

                resume_state = await self._resume_tool_calls(db, session)
                if resume_state in {"waiting", "stopped"}:
                    return
                if resume_state == "processed":
                    continue

                # Conditional write: a Stop that landed after the SELECT above must win.
                claimed = cast(CursorResult[Any], await db.execute(
                    update(Session)
                    .where(Session.id == session.id, Session.status.not_in(TERMINAL_STATUSES))
                    .values(status=SessionStatus.running.value, error=None, stop_requested=False)
                ))
                await db.commit()
                if claimed.rowcount == 0:
                    return
                await emit_event(db, "session.status", {"status": session.status}, session.id)

                if models is None:
                    models = await self._models(db)
                model = select_model(models, session.llm_profile, strict=False)
                prepared = self.context_for(model).prepare(
                    session.messages, reasoning_history=model.reasoning.history
                )
                if prepared.omitted_messages or prepared.truncated_tool_results:
                    await emit_event(
                        db,
                        "context.compacted",
                        {
                            "approximate_tokens": prepared.approximate_tokens,
                            "omitted_messages": prepared.omitted_messages,
                            "truncated_tool_results": prepared.truncated_tool_results,
                        },
                        session.id,
                    )
                definitions = self.registry.definitions(session.mode, model_ids=models.ids())
                audit_payload = json.dumps(
                    {"messages": prepared.messages, "tools": definitions},
                    ensure_ascii=False,
                    default=str,
                )
                llm_client = self._llm_for(models, model)
                decision = self._reasoning_for(session, model)
                reasoning_effort = decision.effective
                try:
                    response = await self._complete(
                        session.id,
                        f"chat:{model.id}",
                        llm_client,
                        model,
                        prepared.messages,
                        definitions,
                        audit_payload,
                        reasoning_effort,
                        prepared.approximate_tokens,
                    )
                except LLMTransientError as primary_error:
                    response = await self._complete_with_fallback(
                        db, session, models, model, definitions, primary_error
                    )
                await db.refresh(session, attribute_names=["status", "stop_requested"])
                if session.stop_requested or session.status == SessionStatus.stopped.value:
                    return

                # Our own globally unique IDs: providers may omit or reuse theirs, and the
                # step counter restarts whenever the loop resumes after an approval.
                normalized_calls = [
                    {**dict(raw_call), "id": f"call_{uuid.uuid4().hex}"}
                    for raw_call in response.tool_calls
                ]

                assistant = Message(
                    session_id=session.id,
                    role="assistant",
                    content=response.content,
                    tool_calls=normalized_calls or None,
                    token_count=response.completion_tokens or None,
                    reasoning_content=response.reasoning,
                    reasoning_effort=reasoning_effort,
                )
                db.add(assistant)
                await db.flush()

                if not normalized_calls:
                    await db.commit()
                    await emit_event(
                        db,
                        "message.created",
                        {"role": "assistant", "message_id": assistant.id},
                        session.id,
                    )
                    session.status = SessionStatus.completed.value
                    await db.commit()
                    await emit_event(db, "session.status", {"status": session.status}, session.id)
                    return

                requires_approval = False
                truncated = response.finish_reason == "length"
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
                        self.registry.validate(
                            name, session.mode, arguments, model_ids=models.ids()
                        )
                        risk = self.registry.risk(name, session.mode)
                        confirmation = self.registry.requires_confirmation(name, session.mode)
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
                        db.add(
                            Message(
                                session_id=session.id,
                                role="tool",
                                tool_call_id=call_id,
                                content=json.dumps(call.result, ensure_ascii=False),
                            )
                        )
                        continue

                    call = ToolCall(
                        id=call_id,
                        session_id=session.id,
                        name=name,
                        arguments=arguments,
                        risk_level=risk.value,
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
                    {"role": "assistant", "message_id": assistant.id},
                    session.id,
                )
                if requires_approval:
                    session.status = SessionStatus.awaiting_confirmation.value
                    await db.commit()
                    await emit_event(
                        db, "approval.required", {"status": session.status}, session.id
                    )
                    return

        raise RuntimeError(
            f"Agent exceeded the maximum of {self.settings.max_inference_steps} inference steps"
        )

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
        for call in unfinished:
            await db.refresh(session, attribute_names=["status", "stop_requested"])
            if session.stop_requested or session.status == SessionStatus.stopped.value:
                return "stopped"
            if call.status == ToolCallStatus.running.value:
                call.status = ToolCallStatus.error.value
                call.result = {
                    "success": False,
                    "error": (
                        "Worker stopped while this tool was running; it was not retried "
                        "to avoid repeating a possible side effect"
                    ),
                }
                db.add(
                    Message(
                        session_id=session.id,
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(call.result, ensure_ascii=False),
                    )
                )
                await db.commit()
                processed = True
                continue

            approval = call.approval
            if approval is not None and approval.status == ApprovalStatus.pending.value:
                waiting = True
                continue
            if approval is None or approval.status == ApprovalStatus.approved.value:
                if not await self._execute_tool_call(db, session, call):
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

    async def _execute_tool_call(self, db: Any, session: Session, call: ToolCall) -> bool:
        if not await self._claim_tool_call(db, call):
            return False
        depth = int((session.configuration or {}).get("depth", 0))
        context = ToolContext(
            session_id=session.id,
            project_root=Path(session.project.root_path).resolve(strict=True),
            depth=depth,
            audit_egress=partial(self._audit_egress, session.id),
            spawn_child=partial(self._spawn_child, session),
        )
        try:
            output = await self.registry.execute(
                call.name, session.mode, context, dict(call.arguments)
            )
            result = {"success": True, "output": output}
            call.status = ToolCallStatus.completed.value
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
            call.status = ToolCallStatus.error.value
        call.result = result
        db.add(
            Message(
                session_id=session.id,
                role="tool",
                tool_call_id=call.id,
                content=json.dumps(result, ensure_ascii=False),
            )
        )
        await db.commit()
        await emit_event(
            db,
            "tool.completed",
            {"tool_call_id": call.id, "name": call.name, "status": call.status},
            session.id,
        )
        return True

    async def _complete(
        self,
        session_id: str,
        operation: str,
        llm_client: LLMClient,
        model: ResolvedModel,
        messages: list[dict[str, Any]],
        definitions: list[dict[str, Any]],
        audit_payload: str,
        reasoning_effort: str | None,
        input_tokens: int,
    ) -> LLMResponse:
        """Call the model, retrying with a larger output budget when the reply was cut off.

        A reply is cut off when ``finish_reason`` is ``length`` or when DeepSeek leaks its raw
        tool-call markup into the text. The budget grows by half per retry. It stays below a
        quarter of the context window and below the space that the input leaves free. After
        the last retry, the method returns the reply as it is.
        """
        max_tokens = model.max_tokens
        free = model.context_window - input_tokens - CONTEXT_SAFETY_TOKENS
        ceiling = max(model.max_tokens, min(model.context_window // 4, free))
        destination = llm_client.endpoint
        for attempt in range(TRUNCATION_RETRIES + 1):
            await self._audit_egress(
                session_id, "llm", destination, operation, audit_payload, "started"
            )
            publisher = StreamPublisher(session_id)
            try:
                response = await llm_client.chat(
                    messages,
                    definitions,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens,
                    on_delta=publisher.add,
                )
                await publisher.flush()
            except Exception as exc:
                await publisher.reset()
                await self._audit_egress(
                    session_id, "llm", destination, operation, audit_payload, "error", str(exc)
                )
                raise
            await self._audit_egress(
                session_id, "llm", destination, operation, audit_payload, "completed"
            )
            cut_off = response.finish_reason == "length" or any(
                marker in (response.content or "") for marker in LEAKED_TOOL_MARKUP
            )
            if cut_off and attempt < TRUNCATION_RETRIES and max_tokens < ceiling:
                await publisher.reset()
            if not cut_off or attempt == TRUNCATION_RETRIES or max_tokens >= ceiling:
                if cut_off and response.finish_reason != "length":
                    response.finish_reason = "length"
                return response
            max_tokens = min(ceiling, int(max_tokens * 1.5))
        raise AssertionError("unreachable")

    async def _complete_with_fallback(
        self,
        db: Any,
        session: Session,
        models: ResolvedModelRegistry,
        primary: ResolvedModel,
        definitions: list[dict[str, Any]],
        primary_error: LLMTransientError,
    ) -> LLMResponse:
        """Send the request to the routing fallback models when the primary one is down."""
        for fallback_id in models.fallback:
            fallback = models.models.get(fallback_id)
            if fallback is None or fallback.id == primary.id or not fallback.configured:
                continue
            prepared = self.context_for(fallback).prepare(session.messages)
            payload = json.dumps(
                {"messages": prepared.messages, "tools": definitions},
                ensure_ascii=False,
                default=str,
            )
            await emit_event(
                db,
                "llm.fallback",
                {"from": primary.id, "to": fallback.id, "reason": str(primary_error)[:500]},
                session.id,
            )
            try:
                return await self._complete(
                    session.id,
                    f"chat:{fallback.id}",
                    self._llm_for(models, fallback),
                    fallback,
                    prepared.messages,
                    definitions,
                    payload,
                    None,
                    prepared.approximate_tokens,
                )
            except LLMTransientError as exc:
                primary_error = exc
        raise primary_error

    @staticmethod
    def _reasoning_for(session: Session, model: ResolvedModel) -> ReasoningDecision:
        """Message override, then the session level, then "auto" or the model default."""
        users = [message for message in session.messages if message.role == "user"]
        recent_errors = 0
        for message in reversed(session.messages):
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

    def context_for(self, model: ResolvedModel) -> ContextManager:
        """Context budget of the session's own model, keeping room for its reply."""
        return ContextManager(
            model.context_window,
            max(self.settings.context_reserved_tokens, model.max_tokens),
            self.settings.max_tool_result_chars,
        )

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
        if mode not in {"dev", "ask"}:
            raise ValueError(f"Unsupported child mode: {mode}")
        async with session_factory() as db:
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
                configuration={"depth": depth + 1},
            )
            child_id = child.id
            await emit_event(
                db,
                "session.child_created",
                {"child_id": child_id, "parent_id": parent.id, "wait": wait},
                parent.id,
            )

        if wait:
            try:
                await asyncio.wait_for(
                    self._run(child_id), timeout=self.settings.max_child_wait_seconds
                )
            except TimeoutError as exc:
                await self.stop(child_id)
                raise RuntimeError("Child agent exceeded its wait timeout") from exc
            except Exception as exc:
                await self._mark_error(child_id, exc)
        else:
            await self.start(child_id)

        async with session_factory() as db:
            child_record = await db.get(Session, child_id)
            final_message = (
                await db.execute(
                    select(Message)
                    .where(Message.session_id == child_id, Message.role == "assistant")
                    .order_by(Message.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            return {
                "session_id": child_id,
                "status": child_record.status if child_record else "missing",
                "llm_profile": child_record.llm_profile if child_record else selected_profile,
                "response": final_message.content if final_message else None,
            }
