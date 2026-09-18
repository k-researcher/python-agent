from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.models import Event

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
                await self._redis.publish(
                    self._channel, json.dumps(envelope, ensure_ascii=False)
                )
            except RedisError:
                logger.exception("Could not publish Redis event; database event remains durable")

    async def _broadcast_local(self, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)


broker = EventBroker()


async def emit_event(
    db: AsyncSession, event_type: str, payload: dict[str, Any], session_id: str | None = None
) -> Event:
    event = Event(session_id=session_id, type=event_type, payload=payload)
    db.add(event)
    await db.commit()
    await db.refresh(event)
    await broker.broadcast(
        {
            "id": event.id,
            "session_id": event.session_id,
            "type": event.type,
            "payload": event.payload,
            "created_at": event.created_at.isoformat(),
        }
    )
    return event
