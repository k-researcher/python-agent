from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.models import Event
from agent.session_fence import CURRENT_FENCE

logger = logging.getLogger(__name__)


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._redis: Redis | None = None
        self._channel = ""
        self._listener: asyncio.Task[None] | None = None
        self._origin = secrets.token_hex(12)

    async def start(self, settings: Settings, *, listen: bool) -> None:
        if settings.execution_mode != "redis" or self._redis is not None:
            return
        self._redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=15,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        self._channel = f"{settings.redis_namespace}:events"
        await self._redis.ping()
        if listen:
            self._listener = asyncio.create_task(self._listen(), name="redis-event-listener")

    async def stop(self) -> None:
        if self._listener is not None:
            self._listener.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener
            self._listener = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def _listen(self) -> None:
        if self._redis is None:
            return
        while True:
            try:
                async with self._redis.pubsub() as pubsub:
                    await pubsub.subscribe(self._channel)
                    while True:
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=1
                        )
                        if not message:
                            await asyncio.sleep(0.05)
                            continue
                        try:
                            envelope = json.loads(str(message["data"]))
                            if envelope.get("origin") != self._origin:
                                await self._broadcast_local(dict(envelope["event"]))
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            logger.warning("Ignored malformed Redis event")
            except RedisError:
                logger.exception("Redis event listener disconnected; retrying")
                await asyncio.sleep(2)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    async def broadcast(self, event: dict[str, Any]) -> None:
        await self._broadcast_local(event)
        if self._redis is not None:
            envelope = {"origin": self._origin, "event": event}
            try:
                await self._redis.publish(self._channel, json.dumps(envelope, ensure_ascii=False))
            except RedisError:
                logger.exception("Could not publish Redis event; database event remains durable")

    async def _broadcast_local(self, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)


broker = EventBroker()


@dataclass(frozen=True)
class PendingEvent:
    event: Event
    payload: dict[str, Any]


async def add_event(
    db: AsyncSession, event_type: str, payload: dict[str, Any], session_id: str | None = None
) -> PendingEvent:
    fence = CURRENT_FENCE.get()
    if fence is not None and session_id == fence.session_id:
        payload = {"generation": fence.generation, **payload}
    event = Event(session_id=session_id, type=event_type, payload=payload)
    db.add(event)
    await db.flush()
    return PendingEvent(
        event,
        {
            "id": event.id,
            "session_id": event.session_id,
            "type": event.type,
            "payload": deepcopy(event.payload),
            "created_at": event.created_at.isoformat(),
        },
    )


async def publish(events: list[PendingEvent]) -> None:
    """Broadcast committed event snapshots without database access."""
    for pending in events:
        try:
            await broker.broadcast(pending.payload)
        except Exception:
            logger.exception("Event %s: broadcast failed", pending.payload["id"])


async def emit_event(
    db: AsyncSession, event_type: str, payload: dict[str, Any], session_id: str | None = None
) -> Event:
    pending = await add_event(db, event_type, payload, session_id)
    await db.commit()
    await publish([pending])
    return pending.event


@dataclass
class StreamRun:
    generation: int
    number: int = 0


CURRENT_STREAMS: ContextVar[StreamRun | None] = ContextVar("stream_run", default=None)


async def resume_stream_run(db: AsyncSession, session_id: str, generation: int) -> StreamRun:
    """Continue the stream numbers of a run that paused, for example for an approval.

    A client drops a stream whose number it already saw with another stream id, so the
    numbers of one generation must not start again.
    """
    payloads = await db.scalars(
        select(Event.payload).where(Event.session_id == session_id, Event.type == "message.created")
    )
    last = max(
        (
            payload["stream_no"]
            for payload in payloads
            if type(payload.get("generation")) is int
            and payload["generation"] == generation
            and type(payload.get("stream_no")) is int
        ),
        default=0,
    )
    return StreamRun(generation, last)


class StreamPublisher:
    """Send streamed text to WebSocket clients as temporary events.

    The publisher joins fragments for up to 50 ms or 8 KiB. It does not write rows to the
    events table: a client that misses deltas reads the saved message after the step.
    """

    FLUSH_SECONDS = 0.05
    FLUSH_BYTES = 8192

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.stream_id = secrets.token_hex(8)
        run = CURRENT_STREAMS.get()
        fence = CURRENT_FENCE.get()
        if run is None:
            run = StreamRun(fence.generation if fence is not None else 0)
            CURRENT_STREAMS.set(run)
        run.number += 1
        self.generation = run.generation
        self.stream_no = run.number
        self.closed = False
        self._seq = 0
        self._pending: dict[str, list[str]] = {}
        self._size = 0
        self._last_flush = asyncio.get_running_loop().time()

    def close(self) -> None:
        self.closed = True
        self._pending, self._size = {}, 0

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "stream_no": self.stream_no,
            "stream_id": self.stream_id,
        }

    async def add(self, kind: str, text: str) -> None:
        if self.closed:
            return
        self._pending.setdefault(kind, []).append(text)
        self._size += len(text)
        now = asyncio.get_running_loop().time()
        if self._size >= self.FLUSH_BYTES or now - self._last_flush >= self.FLUSH_SECONDS:
            await self.flush()

    async def flush(self) -> None:
        if self.closed:
            return
        self._last_flush = asyncio.get_running_loop().time()
        pending, self._pending, self._size = self._pending, {}, 0
        for kind, parts in pending.items():
            if self.closed:
                return
            self._seq += 1
            await broker.broadcast(
                {
                    "type": f"{'reasoning' if kind == 'reasoning' else 'message'}.delta",
                    "session_id": self.session_id,
                    "ephemeral": True,
                    "payload": {
                        **self.identity,
                        "seq": self._seq,
                        "delta": "".join(parts),
                    },
                }
            )

    async def reset(self) -> None:
        """Tell clients to drop the text of this stream (a retry or a fallback starts)."""
        self.close()
        self._seq += 1
        await broker.broadcast(
            {
                "type": "stream.reset",
                "session_id": self.session_id,
                "ephemeral": True,
                "payload": {**self.identity, "seq": self._seq},
            }
        )
