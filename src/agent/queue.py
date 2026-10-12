from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from agent.config import Settings


@dataclass(frozen=True, slots=True)
class QueueMessage:
    message_id: str
    session_id: str


class RedisTaskQueue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        namespace = settings.redis_namespace
        self.stream = f"{namespace}:sessions"
        self.group = f"{namespace}:workers"
        self.events_channel = f"{namespace}:events"
        self.client: Redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=15,
            health_check_interval=30,
            retry_on_timeout=True,
        )

    async def ensure_group(self) -> None:
        try:
            await self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue(self, session_id: str) -> bool:
        dedupe_key = f"{self.stream}:queued:{session_id}"
        inserted = await self.client.set(dedupe_key, "1", ex=300, nx=True)
        if not inserted:
            return False
        try:
            await self.client.xadd(
                self.stream,
                {"session_id": session_id},
                maxlen=100_000,
                approximate=True,
            )
        except Exception:
            await self.client.delete(dedupe_key)
            raise
        return True

    async def read(self, consumer: str, count: int, block_ms: int = 5000) -> list[QueueMessage]:
        rows = await self.client.xreadgroup(
            self.group,
            consumer,
            {self.stream: ">"},
            count=count,
            block=block_ms,
        )
        return self._decode(rows)

    async def claim_stale(self, consumer: str, count: int) -> list[QueueMessage]:
        minimum_idle = self.settings.worker_visibility_timeout_seconds * 1000
        result = await self.client.xautoclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=minimum_idle,
            start_id="0-0",
            count=count,
        )
        messages = result[1] if len(result) > 1 else []
        return self._decode([(self.stream, messages)])

    @staticmethod
    def _decode(rows: Any) -> list[QueueMessage]:
        messages: list[QueueMessage] = []
        for _stream, entries in rows or []:
            for message_id, fields in entries:
                session_id = fields.get("session_id")
                if session_id:
                    messages.append(QueueMessage(str(message_id), str(session_id)))
        return messages

    async def acknowledge(self, message_id: str, session_id: str) -> None:
        await self.client.xack(self.stream, self.group, message_id)
        await self.client.xdel(self.stream, message_id)
        await self.client.delete(f"{self.stream}:queued:{session_id}")

    async def acquire_session_lock(self, session_id: str) -> str | None:
        token = secrets.token_hex(16)
        acquired = await self.client.set(
            f"{self.stream}:lock:{session_id}",
            token,
            ex=self.settings.worker_lock_ttl_seconds,
            nx=True,
        )
        return token if acquired else None

    async def refresh_session_lock(self, session_id: str, token: str) -> bool:
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('expire', KEYS[1], ARGV[2])
        end
        return 0
        """
        result = await self.client.eval(
            script,
            1,
            f"{self.stream}:lock:{session_id}",
            token,
            self.settings.worker_lock_ttl_seconds,
        )
        return bool(result)

    async def release_session_lock(self, session_id: str, token: str) -> None:
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
          return redis.call('del', KEYS[1])
        end
        return 0
        """
        await self.client.eval(script, 1, f"{self.stream}:lock:{session_id}", token)

    async def heartbeat(self, worker: str) -> None:
        await self.client.set(f"{self.stream}:worker:{worker}", "1", ex=30)

    async def remove_heartbeat(self, worker: str) -> None:
        await self.client.delete(f"{self.stream}:worker:{worker}")

    async def health(self) -> dict[str, Any]:
        await self.client.ping()
        workers = 0
        async for _key in self.client.scan_iter(match=f"{self.stream}:worker:*"):
            workers += 1
        return {"redis": "ok", "workers": workers}

    async def close(self) -> None:
        await self.client.aclose()
