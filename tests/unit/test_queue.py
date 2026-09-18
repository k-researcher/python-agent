import fakeredis.aioredis

from agent.config import Settings
from agent.queue import RedisTaskQueue


async def test_redis_stream_queue_deduplicates_and_acknowledges() -> None:
    settings = Settings(
        execution_mode="redis",
        database_url="postgresql+asyncpg://agent:secret@db/agent",
        redis_namespace="test-agent",
    )
    queue = RedisTaskQueue(settings)
    await queue.client.aclose()
    queue.client = fakeredis.aioredis.FakeRedis(decode_responses=True)  # type: ignore[assignment]
    try:
        await queue.ensure_group()
        assert await queue.enqueue("session-1") is True
        assert await queue.enqueue("session-1") is False

        messages = await queue.read("worker-1", count=1, block_ms=1)
        assert len(messages) == 1
        assert messages[0].session_id == "session-1"

        await queue.acknowledge(messages[0].message_id, messages[0].session_id)
        assert await queue.enqueue("session-1") is True
    finally:
        await queue.close()


async def test_worker_heartbeat_is_visible_in_health() -> None:
    settings = Settings(
        execution_mode="redis",
        database_url="postgresql+asyncpg://agent:secret@db/agent",
        redis_namespace="test-agent-health",
    )
    queue = RedisTaskQueue(settings)
    await queue.client.aclose()
    queue.client = fakeredis.aioredis.FakeRedis(decode_responses=True)  # type: ignore[assignment]
    try:
        await queue.heartbeat("worker-1")
        assert await queue.health() == {"redis": "ok", "workers": 1}
        await queue.remove_heartbeat("worker-1")
        assert await queue.health() == {"redis": "ok", "workers": 0}
    finally:
        await queue.close()
