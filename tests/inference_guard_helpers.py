"""Bound task ownership for the inference safety specifications."""

import asyncio

import pytest


@pytest.fixture
async def guard_tasks():
    before = asyncio.all_tasks()
    yield
    async with asyncio.timeout(1):
        pending = asyncio.all_tasks() - before - {asyncio.current_task()}
        # Allow already released workers to finish, then cancel any stalled test work.
        if pending:
            _, pending = await asyncio.wait(pending, timeout=0.05)
        for task in pending:
            task.cancel()
        if pending:
            _, pending = await asyncio.wait(pending, timeout=0.1)
        for task in pending:
            task.cancel()
        if pending:
            _, pending = await asyncio.wait(pending, timeout=0.1)
        assert not pending, "The test left work that suppresses cancellation"
