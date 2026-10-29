"""External session loop. Only the Engine port can perform work here."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import traceback
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent.config import EngineProfileSettings
from agent.engine import (
    ApprovalCancellation,
    ApprovalEvent,
    DiffEvent,
    DoneEvent,
    Engine,
    EngineEvent,
    EngineSpec,
    ErrorEvent,
    LimitsEvent,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UserInputEvent,
)
from agent.engine_journal import UNKNOWN_RESULT
from agent.events import PendingEvent, StreamPublisher, add_event, publish
from agent.inference_guard import Clock, LoopClock, bounded_cleanup, retain
from agent.message_tree import append_message
from agent.models import Approval, EngineTurn, Message, Project, Session, ToolCall, new_id
from agent.modes import tool_profile
from agent.run_finish import finish_run
from agent.session_fence import FenceLostError, RunFence, fence_check

_MAX_EVENT_BYTES = 1024 * 1024
_EVENT_TYPES = (
    TextEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    DiffEvent,
    UsageEvent,
    LimitsEvent,
    ErrorEvent,
    DoneEvent,
    ApprovalEvent,
    UserInputEvent,
)


def validate_event(event: EngineEvent) -> None:
    """Fail closed on incompatible required payloads, without logging raw packets."""
    if type(event) not in _EVENT_TYPES:
        raise ValueError("Unknown required engine event")
    if len(json.dumps(asdict(event), default=str).encode()) > _MAX_EVENT_BYTES:
        raise ValueError("Engine event exceeds 1 MiB")
    for identity in (event.item_id, event.thread_id, event.turn_id, event.submission_id):
        if identity is not None and (
            not isinstance(identity, str) or not 1 <= len(identity) <= 200
        ):
            raise ValueError("Invalid engine event identity")
    if event.seq is not None and (type(event.seq) is not int or event.seq < 0):
        raise ValueError("Invalid engine event sequence")
    if isinstance(event, (TextEvent, ReasoningEvent)):
        if not isinstance(event.text, str) or event.phase not in ("delta", "final"):
            raise ValueError("Invalid text engine event")
    elif isinstance(event, DoneEvent):
        if event.status not in ("completed", "stopped", "error", "interrupted") or (
            event.reason is not None and not isinstance(event.reason, str)
        ):
            raise ValueError("Invalid done engine event")
    elif isinstance(event, ToolCallEvent):
        if (
            not event.item_id
            or not isinstance(event.name, str)
            or not event.name
            or not isinstance(event.input, dict)
            or event.status not in ("pending", "running", "completed", "failed", "cancelled")
        ):
            raise ValueError("Invalid tool engine event")
    elif isinstance(event, ToolResultEvent):
        if (
            not event.related_item_id
            or not isinstance(event.output, str)
            or type(event.success) is not bool
        ):
            raise ValueError("Invalid tool result engine event")
    elif isinstance(event, DiffEvent):
        if (
            not isinstance(event.patch, str)
            or not isinstance(event.paths, list)
            or not all(isinstance(path, str) for path in event.paths)
            or event.status not in ("proposed", "applied")
        ):
            raise ValueError("Invalid diff engine event")
    elif isinstance(event, UsageEvent):
        if (
            any(
                value is not None and (type(value) is not int or value < 0)
                for value in (event.input_tokens, event.output_tokens, event.cache_tokens)
            )
            or type(event.estimated_cost) is not bool
            or (
                event.reported_cost is not None
                and (
                    not isinstance(event.reported_cost, Decimal)
                    or not event.reported_cost.is_finite()
                    or event.reported_cost < 0
                )
            )
        ):
            raise ValueError("Invalid usage engine event")
    elif isinstance(event, ErrorEvent) and (
        not isinstance(event.code, str)
        or not isinstance(event.message, str)
        or type(event.retryable) is not bool
    ):
        raise ValueError("Invalid error engine event")
    elif isinstance(event, UserInputEvent):
        if not isinstance(event.request_id, str) or not 1 <= len(event.request_id) <= 200:
            raise ValueError("Invalid user input engine event")
    elif isinstance(event, ApprovalEvent) and (
        not event.request_id
        or not isinstance(event.request_id, str)
        or len(event.request_id) > 200
        or not isinstance(event.action, str)
        or not event.action
        or not event.item_id
        or not isinstance(event.parameters, dict)
        or event.scope not in ("turn", "request")
    ):
        raise ValueError("Invalid approval engine event")


class EngineRunner:
    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        engine: Engine,
        profile: EngineProfileSettings,
        fence: RunFence,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.factory, self.engine, self.profile, self.fence = factory, engine, profile, fence
        self.clock = clock or LoopClock()
        self.turn_id: str | None = None
        self.pending_requests: set[str] = set()
        self.seen_requests: set[str] = set()
        self.approval_ids: dict[str, str] = {}
        self.publisher = StreamPublisher(fence.session_id)
        self.turn_key = ""
        self.reply_id: int | None = None
        self.thread_id: str | None = None
        self.submission_id = ""
        self.finished = False
        self.needs_interrupt = True

    async def run(self) -> None:
        try:
            prepared = await self.prepare()
            if prepared is None:
                return
            spec, input_text, resume_id = prepared
            async with asyncio.timeout(self.profile.timeouts.get("open", 30)):
                health = await self.engine.health()
                async with self.factory() as db:
                    await fence_check(db, self.fence)
                    session = await db.get(Session, self.fence.session_id)
                    assert session is not None
                    health_state = {"status": health.status, "diagnostic": health.diagnostic}
                    session.engine_state = {**(session.engine_state or {}), "health": health_state}
                    health_event = await add_event(db, "engine.health", health_state, session.id)
                    await db.commit()
                await publish([health_event])
                if health.status != "ready" or not health.supported or not health.isolation_capable:
                    raise RuntimeError(
                        "engine недоступен: обязательная защита или протокол недоступны"
                    )
                thread = await self.engine.open(spec, resume_id)
            if not isinstance(thread.native_id, str) or not 1 <= len(thread.native_id) <= 200:
                raise ValueError("Engine returned an invalid thread identity")
            self.thread_id = thread.native_id
            # This COMMIT precedes invoking turn(), including its first iterator step.
            async with self.factory() as db:
                await fence_check(db, self.fence)
                turn = await db.get(EngineTurn, self.turn_key)
                session = await db.get(Session, self.fence.session_id)
                assert turn is not None and session is not None
                if turn.state != "prepared":
                    raise RuntimeError(UNKNOWN_RESULT)
                turn.state = "dispatched"
                session.engine_state = {**(session.engine_state or {}), "thread_id": self.thread_id}
                session.status = "running"
                pending = [await add_event(db, "session.status", {"status": "running"}, session.id)]
                await db.commit()
            await publish(pending)
            async with asyncio.timeout(self.profile.timeouts.get("turn", 3600)):
                async for event in self.engine.turn(thread, input_text, self.submission_id):
                    if (
                        isinstance(event, ApprovalEvent)
                        and isinstance(event.request_id, str)
                        and event.request_id
                    ):
                        if event.request_id in self.seen_requests:
                            raise ValueError("Duplicate native approval request")
                        self.pending_requests.add(event.request_id)
                    validate_event(event)
                    await self.consume(event)
                    if self.finished:
                        return
            raise RuntimeError("Engine EOF without done")
        except asyncio.CancelledError as error:
            # Release SQLite cursors held by completed frames before cleanup opens
            # a new writer, just as the supervisor does for model runs.
            traceback.clear_frames(error.__traceback__)
            # Close the stream synchronously before any cancellation cleanup can yield.
            self.publisher.close()
            await self.cleanup(interrupted=True)
            raise
        except FenceLostError:
            self.publisher.close()
            await self.cleanup(interrupted=False)
            raise
        except Exception as error:
            self.publisher.close()
            await self.finish("error", str(error), error=str(error))
        finally:
            self.publisher.close()
            await self.cleanup(interrupted=False)

    async def prepare(self) -> tuple[EngineSpec, str, str | None] | None:
        async with self.factory() as db:
            await fence_check(db, self.fence)
            session = await db.get(Session, self.fence.session_id)
            assert session is not None
            project = await db.get(Project, session.project_id)
            assert project is not None
            state = session.engine_state or {}
            root = Path(project.root_path).resolve(strict=True)
            revision = hashlib.sha256(
                json.dumps(self.profile.model_dump(), sort_keys=True).encode()
            ).hexdigest()
            if (
                state.get("project_root") != str(root)
                or state.get("host") != socket.gethostname()
                or state.get("os_uid") != os.getuid()
                or state.get("profile_revision") != revision
            ):
                raise RuntimeError("engine недоступен: привязка сессии изменилась")
            input_row = await db.scalar(
                select(Message)
                .where(
                    Message.session_id == session.id,
                    Message.role == "user",
                    Message.kind == "normal",
                    Message.skipped.is_(False),
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
            if input_row is None:
                raise ValueError("Engine input message is missing")
            turn = await db.scalar(
                select(EngineTurn).where(
                    EngineTurn.session_id == session.id,
                    EngineTurn.generation == self.fence.generation,
                    EngineTurn.input_message_id == input_row.id,
                )
            )
            if turn is None:
                turn = EngineTurn(
                    session_id=session.id,
                    generation=self.fence.generation,
                    input_message_id=input_row.id,
                    state="prepared",
                )
                db.add(turn)
                await db.flush()
            self.turn_key, self.submission_id = turn.id, turn.submission_id
            if turn.state != "prepared":
                pending = await finish_run(
                    db, session.id, self.fence.generation, "interrupted", reason=UNKNOWN_RESULT
                )
                await db.commit()
                await publish(pending)
                self.finished = True
                return None
            instructions = "\n\n".join(
                await db.scalars(
                    select(Message.content)
                    .where(
                        Message.session_id == session.id,
                        Message.role == "system",
                    )
                    .order_by(Message.id)
                )
            )
            spec = EngineSpec(
                root,
                "read" if tool_profile(session.mode) == "ask" else "dev",
                instructions + "\n\nИспользуй только собственные инструменты внешнего исполнителя. "
                "Названия инструментов в инструкциях роли описывают назначение действий. "
                "Инструменты приложения, MCP и дочерние модели в этом ходе недоступны. "
                "Полные инструкции skills читай собственными инструментами из файлов проекта.",
                {
                    "profile": self.profile.model_dump(),
                    "session": session.configuration,
                    "session_id": session.id,
                },
            )
            await db.commit()  # prepared survives a crash during health/open
            return spec, input_row.content or "", state.get("thread_id")

    async def consume(self, event: EngineEvent) -> None:
        if event.submission_id is not None and event.submission_id != self.submission_id:
            raise ValueError("Engine submission identity changed")
        if event.thread_id is not None and event.thread_id != self.thread_id:
            raise ValueError("Engine thread identity changed")
        if self.turn_id is not None and event.turn_id is not None and event.turn_id != self.turn_id:
            raise ValueError("Engine turn identity changed")
        self.turn_id = event.turn_id or self.turn_id
        if isinstance(event, ApprovalEvent):
            await self.approve(event)
            return
        if isinstance(event, UserInputEvent):
            async with self.factory() as db:
                await fence_check(db, self.fence)
                async with asyncio.timeout(1):
                    await self.engine.answer_user_input(event.request_id)
                await db.commit()
            await self.finish(
                "interrupted", "Интерактивный ввод пользователя пока не поддерживается"
            )
            return
        if isinstance(event, ErrorEvent):
            raise RuntimeError(f"{event.code}: {event.message}")
        if isinstance(event, DoneEvent):
            await self.finish(
                event.status, event.reason, error=event.reason if event.status == "error" else None
            )
            return
        pending: list[PendingEvent] = []
        delta: tuple[str, str] | None = None
        close_stream = False
        async with self.factory() as db:
            await fence_check(db, self.fence)
            turn = await db.get(EngineTurn, self.turn_key)
            session = await db.get(Session, self.fence.session_id)
            assert turn is not None and session is not None
            cursor = dict(turn.last_item_ids or {})
            key = f"{event.kind}:{event.item_id or '_'}"
            previous = cursor.get(key, {})
            if previous.get("final") or (
                event.seq is not None and event.seq <= previous.get("seq", -1)
            ):
                return
            if isinstance(event, (TextEvent, ReasoningEvent)):
                if self.publisher.closed:
                    self.publisher = StreamPublisher(session.id)
                message_id = previous.get("message_id", self.reply_id)
                message = await db.get(Message, message_id) if message_id else None
                if message is None:
                    message = await append_message(db, session.id, role="assistant", content="")
                    self.reply_id = message.id
                field = "content" if isinstance(event, TextEvent) else "reasoning_content"
                whole = getattr(message, field) or ""
                start, length = previous.get("start", len(whole)), previous.get("length", 0)
                old = whole[start : start + length]
                text = old + event.text if event.phase == "delta" else event.text
                setattr(message, field, whole[:start] + text + whole[start + length :])
                # Items may finish out of order. Move later item offsets, rather
                # than replacing the suffix of an unrelated item.
                for other_key, other in tuple(cursor.items()):
                    if (
                        other_key != key
                        and other_key.startswith(event.kind + ":")
                        and other.get("message_id") == message.id
                        and other.get("start", -1) >= start + length
                    ):
                        cursor[other_key] = {**other, "start": other["start"] + len(text) - length}
                delta = (
                    "text" if field == "content" else "reasoning",
                    event.text
                    if event.phase == "delta"
                    else text[len(old) :]
                    if text.startswith(old)
                    else "",
                )
                cursor[key] = {
                    "seq": event.seq if event.seq is not None else -1,
                    "final": event.phase == "final",
                    "start": start,
                    "length": len(text),
                    "message_id": message.id,
                }
                close_stream = event.phase == "final"
                cursor["_stream"] = {
                    **self.publisher.identity,
                    "message_id": message.id,
                    "closed": close_stream,
                }
                if close_stream:
                    pending.append(
                        await add_event(
                            db,
                            "message.created",
                            {
                                "role": "assistant",
                                "message_id": message.id,
                                **self.publisher.identity,
                            },
                            session.id,
                        )
                    )
            elif isinstance(event, ToolCallEvent):
                if self.reply_id and not self.publisher.closed:
                    pending.append(
                        await add_event(
                            db,
                            "message.created",
                            {
                                "role": "assistant",
                                "message_id": self.reply_id,
                                **self.publisher.identity,
                            },
                            session.id,
                        )
                    )
                    cursor["_stream"] = {**cursor.get("_stream", {}), "closed": True}
                    close_stream = True
                call_id = previous.get("call_id", new_id())
                call = await db.get(ToolCall, call_id)
                if call is None:
                    call = ToolCall(
                        id=call_id,
                        session_id=session.id,
                        name=event.name,
                        arguments=event.input,
                        risk_level="observed",
                        status="running",
                        executor="engine",
                        native_item_id=event.item_id,
                    )
                    db.add(call)
                    message = await append_message(
                        db,
                        session.id,
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": event.name,
                                    "arguments": json.dumps(event.input),
                                },
                            }
                        ],
                    )
                    pending.append(
                        await add_event(
                            db,
                            "message.created",
                            {"role": "assistant", "message_id": message.id},
                            session.id,
                        )
                    )
                cursor[key] = {
                    "seq": event.seq if event.seq is not None else -1,
                    "call_id": call_id,
                }
                self.reply_id = None
            elif isinstance(event, ToolResultEvent):
                call_id = cursor.get(f"tool_call:{event.related_item_id}", {}).get("call_id")
                call = await db.get(ToolCall, call_id) if call_id else None
                if call is None:
                    raise ValueError("Engine result has no observed tool call")
                if call.result is None:
                    call.status = "completed" if event.success else "failed"
                    call.result = {"success": event.success, "output": event.output}
                    await self.tool_result(db, call, pending)
                cursor[key] = {"final": True}
            elif isinstance(event, DiffEvent):
                call = ToolCall(
                    id=new_id(),
                    session_id=session.id,
                    name="engine.diff",
                    arguments={"paths": event.paths},
                    risk_level="observed",
                    executor="engine",
                    native_item_id=event.item_id or "diff",
                    status="completed",
                    result={
                        "success": True,
                        "patch": event.patch,
                        "paths": event.paths,
                        "status": event.status,
                    },
                )
                db.add(call)
                message = await append_message(
                    db,
                    session.id,
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                )
                pending.append(
                    await add_event(
                        db,
                        "message.created",
                        {"role": "assistant", "message_id": message.id},
                        session.id,
                    )
                )
                await self.tool_result(db, call, pending)
                pending.append(
                    await add_event(
                        db,
                        "engine.diff",
                        {"tool_call_id": call.id, **(call.result or {})},
                        session.id,
                    )
                )
                cursor[key] = {"seq": event.seq if event.seq is not None else -1}
                self.reply_id = None
            elif isinstance(event, UsageEvent):
                usage = asdict(event)
                usage["reported_cost"] = (
                    str(event.reported_cost) if event.reported_cost is not None else None
                )
                session.engine_state = {**(session.engine_state or {}), "usage": usage}
                pending.append(await add_event(db, "engine.usage", usage, session.id))
                cursor[key] = {"seq": event.seq if event.seq is not None else -1}
            elif isinstance(event, LimitsEvent):
                # Optional limit notifications do not affect execution.
                return
            turn.native_turn_id = self.turn_id
            turn.last_item_ids = cursor
            session.engine_state = {
                **(session.engine_state or {}),
                "history_cursor": {"item_id": event.item_id, "seq": event.seq},
            }
            await db.commit()
        # Check the fence again before ephemeral effects; no DB transaction spans port IO.
        async with self.factory() as db:
            await fence_check(db, self.fence)
            if delta is not None and delta[1]:
                await self.publisher.add(*delta)
                await self.publisher.flush()
        await publish(pending)
        if close_stream:
            self.publisher.close()

    async def approve(self, event: ApprovalEvent) -> None:
        self.seen_requests.add(event.request_id)
        ttl = self.profile.timeouts.get("approval", 300)
        deadline = self.clock.now() + ttl
        pending: list[PendingEvent] = []
        close_stream = False
        async with self.factory() as db:
            await fence_check(db, self.fence)
            session = await db.get(Session, self.fence.session_id)
            turn = await db.get(EngineTurn, self.turn_key)
            assert session is not None and turn is not None
            duplicate = await db.scalar(
                select(ToolCall.id).where(
                    ToolCall.session_id == session.id,
                    ToolCall.executor == "engine",
                    ToolCall.native_request_id == event.request_id,
                    ToolCall.id.in_(
                        select(Approval.tool_call_id).where(
                            Approval.generation == self.fence.generation
                        )
                    ),
                )
            )
            if duplicate:
                raise ValueError("Duplicate native approval request")
            cursor = dict(turn.last_item_ids or {})
            if self.reply_id and not self.publisher.closed:
                pending.append(
                    await add_event(
                        db,
                        "message.created",
                        {
                            "role": "assistant",
                            "message_id": self.reply_id,
                            **self.publisher.identity,
                        },
                        session.id,
                    )
                )
                cursor["_stream"] = {**cursor.get("_stream", {}), "closed": True}
                close_stream = True
            call_id = cursor.get(f"tool_call:{event.item_id}", {}).get("call_id")
            call = await db.get(ToolCall, call_id) if call_id else None
            if call is None:
                call = ToolCall(
                    id=new_id(),
                    session_id=session.id,
                    name=event.action,
                    arguments=event.parameters,
                    risk_level="observed",
                    executor="engine",
                    native_item_id=event.item_id,
                    native_request_id=event.request_id,
                    status="awaiting_confirmation",
                )
                db.add(call)
                message = await append_message(
                    db,
                    session.id,
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(event.parameters),
                            },
                        }
                    ],
                )
                pending.append(
                    await add_event(
                        db,
                        "message.created",
                        {"role": "assistant", "message_id": message.id},
                        session.id,
                    )
                )
            else:
                if call.native_request_id or call.result is not None:
                    raise ValueError("Native tool already has a decision or result")
                call.arguments = event.parameters
                call.native_request_id = event.request_id
                call.status = "awaiting_confirmation"
            await db.flush()
            approval = Approval(
                id=new_id(),
                tool_call_id=call.id,
                generation=self.fence.generation,
                executor="engine",
                scope=event.scope,
                expires_at=datetime.now(UTC) + timedelta(seconds=ttl),
            )
            db.add(approval)
            await db.flush()
            self.approval_ids[event.request_id] = approval.id
            cursor[f"tool_call:{event.item_id}"] = {"call_id": call.id}
            turn.last_item_ids, turn.native_turn_id = cursor, self.turn_id
            self.reply_id = None
            session.status = "awaiting_confirmation"
            pending.append(
                await add_event(
                    db,
                    "approval.required",
                    {
                        "approval_id": approval.id,
                        "tool_call_id": call.id,
                        "request_id": event.request_id,
                        "generation": self.fence.generation,
                        "executor": "engine",
                        "scope": event.scope,
                        "expires_at": approval.expires_at.isoformat()
                        if approval.expires_at
                        else None,
                    },
                    session.id,
                )
            )
            pending.append(
                await add_event(
                    db, "session.status", {"status": "awaiting_confirmation"}, session.id
                )
            )
            await db.commit()
        if close_stream:
            self.publisher.close()
        await publish(pending)
        while True:
            async with self.factory() as db:
                await fence_check(db, self.fence)
                record = await db.get(Approval, self.approval_ids[event.request_id])
                assert record is not None
                status = record.status
                await db.commit()
            if self.clock.now() >= deadline or status not in {"pending", "approved", "rejected"}:
                await self.answer_request(event.request_id, "deny", cancelled=True)
                await self.finish(
                    "interrupted", "Срок ожидания разрешения истёк или запрос отменён"
                )
                return
            if status != "pending":
                delivered = await self.answer_request(
                    event.request_id, "allow" if status == "approved" else "deny", deadline=deadline
                )
                if not delivered:
                    await self.finish("interrupted", "Срок ожидания разрешения истёк")
                    return
                async with self.factory() as db:
                    await fence_check(db, self.fence)
                    session = await db.get(Session, self.fence.session_id)
                    assert session is not None
                    session.status = "running"
                    event_row = await add_event(
                        db, "session.status", {"status": "running"}, session.id
                    )
                    await db.commit()
                await publish([event_row])
                return
            await self.clock.until(min(deadline, self.clock.now() + 0.05))

    async def answer_request(
        self,
        request_id: str,
        decision: Literal["allow", "deny"],
        *,
        cancelled: bool = False,
        deadline: float | None = None,
    ) -> bool:
        # A deny during teardown is safe after lease loss; an allow requires the live fence.
        answered = False
        try:
            async with self.factory() as db:
                if cancelled:
                    db.sync_session.info["fence_exempt"] = True
                else:
                    await fence_check(db, self.fence)
                approval_id = self.approval_ids.get(request_id)
                approval = (
                    await db.scalar(
                        select(Approval).where(Approval.id == approval_id).with_for_update()
                    )
                    if approval_id
                    else None
                )
                if approval is not None:
                    if approval.delivered_at is not None:
                        answered = True
                        return False
                    if not cancelled and (
                        (deadline is not None and self.clock.now() >= deadline)
                        or approval.expires_at is None
                        or approval.expires_at.replace(tzinfo=UTC) <= datetime.now(UTC)
                    ):
                        cancelled, decision = True, "deny"
                    if cancelled:
                        approval.status, approval.resolved_at = "cancelled", datetime.now(UTC)
                    elif approval.status not in {"approved", "rejected"}:
                        raise FenceLostError("Native approval was withdrawn")
                elif not cancelled:
                    raise ValueError("Native approval record is missing")
                await db.flush()
                # Keep the row lock across bounded port IO so Stop cannot win before allow.
                try:
                    async with asyncio.timeout(1):
                        if cancelled and isinstance(self.engine, ApprovalCancellation):
                            await self.engine.cancel_approval(request_id)
                        else:
                            await self.engine.answer(request_id, decision)
                    answered = True
                except Exception:
                    answered = True
                    if approval is not None:
                        approval.status, approval.resolved_at = "interrupted", datetime.now(UTC)
                    await db.commit()
                    raise
                if approval is not None:
                    approval.delivered_at = datetime.now(UTC)
                    call = await db.get(ToolCall, approval.tool_call_id)
                    if (
                        not cancelled
                        and call is not None
                        and call.status == "awaiting_confirmation"
                    ):
                        call.status = "running"
                await db.commit()
                return not cancelled
        finally:
            if answered:
                self.pending_requests.discard(request_id)

    async def tool_result(
        self, db: AsyncSession, call: ToolCall, pending: list[PendingEvent]
    ) -> None:
        message = await append_message(
            db,
            self.fence.session_id,
            role="tool",
            tool_call_id=call.id,
            content=json.dumps(call.result, ensure_ascii=False),
        )
        pending.append(
            await add_event(
                db,
                "message.created",
                {"role": "tool", "message_id": message.id},
                self.fence.session_id,
            )
        )
        pending.append(
            await add_event(
                db,
                "tool.completed",
                {"tool_call_id": call.id, "status": call.status},
                self.fence.session_id,
            )
        )

    async def finish(self, status: str, reason: str | None, *, error: str | None = None) -> None:
        async with self.factory() as db:
            await fence_check(db, self.fence)
            if self.turn_key:
                turn = await db.get(EngineTurn, self.turn_key)
                assert turn is not None
                turn.native_turn_id = self.turn_id or turn.native_turn_id
            pending = await finish_run(
                db, self.fence.session_id, self.fence.generation, status, reason=reason, error=error
            )
            await db.commit()
        self.finished = True
        self.needs_interrupt = status != "completed"
        await publish(pending)

    async def cleanup(self, *, interrupted: bool) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        deadline = self.clock.now() + 5

        async def cancel_requests() -> None:
            for request_id in tuple(self.pending_requests):
                with suppress(Exception):
                    await self.answer_request(request_id, "deny", cancelled=True)
            self.pending_requests.clear()
            await self.engine.interrupt(self.turn_id or self.submission_id)

        async def close() -> None:
            try:
                if self.needs_interrupt or interrupted:
                    await bounded_cleanup(cancel_requests(), self.clock, deadline - 3)
            finally:
                await bounded_cleanup(self.engine.close(), self.clock, deadline - 0.5)
            if interrupted:
                with suppress(FenceLostError):
                    await self.finish("interrupted", "Исполнение прервано; " + UNKNOWN_RESULT)

        cleanup = retain(asyncio.create_task(bounded_cleanup(close(), self.clock, deadline)))
        # A repeated cancellation must not cancel process-group teardown.
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
