from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from agent.config import Settings
from agent.context import ContextManager
from agent.database import session_factory
from agent.events import emit_event
from agent.llm import LLMClient
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
from agent.session_service import create_session_record
from agent.tools import ToolRegistry
from agent.tools.base import ToolContext

TERMINAL_STATUSES = {
    SessionStatus.completed.value,
    SessionStatus.stopped.value,
    SessionStatus.error.value,
}


class AgentSupervisor:
    def __init__(self, settings: Settings, registry: ToolRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ToolRegistry(settings=settings)
        self.llm = LLMClient(settings, settings.default_llm_profile)
        self._llm_clients: dict[str, LLMClient] = {}
        self.context = ContextManager(
            settings.context_window,
            settings.context_reserved_tokens,
            settings.max_tool_result_chars,
        )
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
                await db.commit()
                await emit_event(db, "session.status", {"status": session.status}, session_id)

        for child_id in children:
            await self.stop(child_id)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
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
            approval.comment = comment
            approval.resolved_at = utcnow()

            if decision == "approve":
                approval.status = ApprovalStatus.approved.value
                call.status = ToolCallStatus.pending.value
            else:
                approval.status = ApprovalStatus.rejected.value
                call.status = ToolCallStatus.rejected.value
                call.result = {"success": False, "error": comment or "Rejected by user"}
                db.add(
                    Message(
                        session_id=session.id,
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(call.result, ensure_ascii=False),
                    )
                )
            await db.commit()

            pending = (
                await db.execute(
                    select(Approval)
                    .join(ToolCall)
                    .where(
                        ToolCall.session_id == session.id,
                        Approval.status == ApprovalStatus.pending.value,
                    )
                )
            ).scalars().first()
            if pending is None:
                session.status = SessionStatus.pending.value
                await db.commit()
                await emit_event(db, "session.status", {"status": session.status}, session.id)
                await self.start(session.id)
            else:
                await emit_event(
                    db,
                    "approval.resolved",
                    {"approval_id": approval.id, "decision": decision},
                    session.id,
                )
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

    async def _run(self, session_id: str) -> None:
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
                if resume_state == "waiting":
                    return
                if resume_state == "processed":
                    continue

                session.status = SessionStatus.running.value
                session.error = None
                session.stop_requested = False
                await db.commit()
                await emit_event(db, "session.status", {"status": session.status}, session.id)

                prepared = self.context.prepare(session.messages)
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
                definitions = self.registry.definitions(session.mode)
                audit_payload = json.dumps(
                    {"messages": prepared.messages, "tools": definitions},
                    ensure_ascii=False,
                    default=str,
                )
                llm_client = self._llm_for(session.llm_profile)
                destination = llm_client.endpoint
                await self._audit_egress(
                    session.id,
                    "llm",
                    destination,
                    f"chat:{session.llm_profile}",
                    audit_payload,
                    "started",
                )
                try:
                    response = await llm_client.chat(prepared.messages, definitions)
                except Exception as exc:
                    await self._audit_egress(
                        session.id,
                        "llm",
                        destination,
                        f"chat:{session.llm_profile}",
                        audit_payload,
                        "error",
                        str(exc),
                    )
                    raise
                await self._audit_egress(
                    session.id,
                    "llm",
                    destination,
                    f"chat:{session.llm_profile}",
                    audit_payload,
                    "completed",
                )
                await db.refresh(session, attribute_names=["status", "stop_requested"])
                if session.stop_requested or session.status == SessionStatus.stopped.value:
                    return

                normalized_calls: list[dict[str, Any]] = []
                for call_index, raw_call in enumerate(response.tool_calls):
                    normalized = dict(raw_call)
                    external_id = str(raw_call.get("id") or f"generated-{call_index}")
                    suffix = hashlib.sha256(external_id.encode()).hexdigest()[:16]
                    normalized["id"] = f"{session.id}:{_step}:{call_index}:{suffix}"
                    normalized_calls.append(normalized)

                assistant = Message(
                    session_id=session.id,
                    role="assistant",
                    content=response.content,
                    tool_calls=normalized_calls or None,
                    token_count=response.completion_tokens or None,
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
                for raw_call in normalized_calls:
                    function = raw_call.get("function", {})
                    name = str(function.get("name", ""))
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError
                    except (TypeError, ValueError, json.JSONDecodeError):
                        arguments = {"_invalid_arguments": function.get("arguments")}

                    call_id = str(raw_call.get("id") or f"call-{assistant.id}-{name}")
                    try:
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
                await self._execute_tool_call(db, session, call)
                processed = True

        if waiting:
            session.status = SessionStatus.awaiting_confirmation.value
            await db.commit()
            return "waiting"
        return "processed" if processed else "ready"

    async def _execute_tool_call(self, db: Any, session: Session, call: ToolCall) -> None:
        call.status = ToolCallStatus.running.value
        await db.commit()
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

    def _llm_for(self, profile_name: str) -> LLMClient:
        if profile_name == self.settings.default_llm_profile:
            return self.llm
        client = self._llm_clients.get(profile_name)
        if client is None:
            client = LLMClient(self.settings, profile_name)
            self._llm_clients[profile_name] = client
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
        selected_profile = llm_profile or parent.llm_profile
        self.settings.resolve_llm_profile(selected_profile)

        async with session_factory() as db:
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
