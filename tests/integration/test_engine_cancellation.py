"""Runner cancellation reaches the optional neutral port without becoming rejection."""

import asyncio

import pytest

from agent.events import broker
from tests.integration.test_engine_approvals import approval_rows, request, required
from tests.integration.test_engine_runner import FakeEngine, setup


@pytest.mark.parametrize("cause", ["stop", "shutdown"])
async def test_runner_uses_native_request_cancellation(tmp_path, monkeypatch, cause):
    class CancellableEngine(FakeEngine):
        def __init__(self):
            super().__init__([request()])
            self.cancellations = []

        async def cancel_approval(self, request_id):
            self.cancellations.append(request_id)

    async with asyncio.timeout(4):
        fake = CancellableEngine()
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        async with broker.subscribe() as queue:
            await worker.start(sid, new_run=False, expected_generation=1)
            task = worker._tasks[sid]
            try:
                await required(queue, task)
                if cause == "stop":
                    await worker.stop(sid)
                else:
                    await worker.shutdown()
                assert fake.cancellations == ["one"]
                assert fake.answers == []
                assert (await approval_rows(sid))[0].status == "cancelled"
                assert fake.closed.is_set()
            finally:
                await worker.shutdown()
