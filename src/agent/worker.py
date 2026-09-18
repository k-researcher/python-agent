from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from contextlib import suppress

from redis.exceptions import RedisError

from agent.config import get_settings
from agent.database import init_database
from agent.events import broker
from agent.queue import QueueMessage, RedisTaskQueue
from agent.runtime import AgentSupervisor

logger = logging.getLogger(__name__)


class AgentWorker:
    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.execution_mode != "redis":
            raise RuntimeError("agent-worker requires AGENT_EXECUTION_MODE=redis")
        self.settings.validate_runtime_security()
        self.supervisor = AgentSupervisor(self.settings)
        if self.supervisor.task_queue is None:
            raise RuntimeError("Redis task queue is unavailable")
        self.queue: RedisTaskQueue = self.supervisor.task_queue
        default_name = f"{socket.gethostname()}-{os.getpid()}"
        self.name = self.settings.worker_name or default_name
        self._tasks: set[asyncio.Task[None]] = set()

    async def serve(self) -> None:
        await init_database(recover_running=False)
        await self.queue.ensure_group()
        await broker.start(self.settings, listen=False)
        heartbeat = asyncio.create_task(self._heartbeat(), name="worker-heartbeat")
        try:
            while True:
                self._tasks = {task for task in self._tasks if not task.done()}
                capacity = self.settings.max_parallel_sessions - len(self._tasks)
                if capacity <= 0:
                    await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
                    continue

                try:
                    messages = await self.queue.claim_stale(self.name, capacity)
                    if not messages:
                        messages = await self.queue.read(self.name, capacity)
                except RedisError:
                    logger.exception("Redis queue is temporarily unavailable")
                    await asyncio.sleep(2)
                    continue
                for message in messages:
                    task = asyncio.create_task(
                        self._process(message),
                        name=f"worker-session-{message.session_id}",
                    )
                    self._tasks.add(task)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.queue.remove_heartbeat(self.name)
            await broker.stop()
            await self.supervisor.shutdown()

    async def _heartbeat(self) -> None:
        while True:
            await self.queue.heartbeat(self.name)
            await asyncio.sleep(10)

    async def _refresh_lock(self, session_id: str, token: str) -> None:
        interval = max(10, self.settings.worker_lock_ttl_seconds // 3)
        while True:
            await asyncio.sleep(interval)
            if not await self.queue.refresh_session_lock(session_id, token):
                raise RuntimeError(f"Lost distributed lock for session {session_id}")

    async def _process(self, message: QueueMessage) -> None:
        token = await self.queue.acquire_session_lock(message.session_id)
        if token is None:
            return
        lock_heartbeat = asyncio.create_task(
            self._refresh_lock(message.session_id, token),
            name=f"session-lock-{message.session_id}",
        )
        run_task = asyncio.create_task(
            self.supervisor.run_job(message.session_id),
            name=f"execute-session-{message.session_id}",
        )
        try:
            done, _pending = await asyncio.wait(
                {run_task, lock_heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if lock_heartbeat in done:
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task
                lock_heartbeat.result()
            await run_task
            await self.queue.acknowledge(message.message_id, message.session_id)
        except asyncio.CancelledError:
            run_task.cancel()
            raise
        except Exception:
            logger.exception("Worker failed session %s; job remains pending", message.session_id)
        finally:
            lock_heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await lock_heartbeat
            await self.queue.release_session_lock(message.session_id, token)


async def main() -> None:
    worker_task = asyncio.create_task(AgentWorker().serve(), name="agent-worker")
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signal_name, worker_task.cancel)
    with suppress(asyncio.CancelledError):
        await worker_task


def run() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(main())


if __name__ == "__main__":
    run()
