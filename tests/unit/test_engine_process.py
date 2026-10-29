"""Guard specifications with a fake stdlib CLI; no native CLI or network."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest


def test_protocol_lines_and_environment(monkeypatch):
    from agent.engine_process import clean_env, decode_line

    monkeypatch.setenv("PRIVATE_TOKEN", "secret")
    assert "PRIVATE_TOKEN" not in clean_env()
    assert decode_line(b'{"optional":true}') == {"optional": True}
    for raw in (b"{", b"[]", b"x" * (1024 * 1024 + 1)):
        with pytest.raises(ValueError):
            decode_line(raw)


@pytest.mark.parametrize("stop", ["close", "eof"])
async def test_guard_control_eof_and_term_resistant_group(tmp_path, stop):
    from agent.engine_process import GuardedProcess

    async with asyncio.timeout(5):
        script = tmp_path / "fake.py"
        script.write_text("""import os, signal, subprocess, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child_code = ('import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
              'print("child-ready", flush=True); signal.pause()')
child = subprocess.Popen([sys.executable, '-c', child_code],
                         stdout=subprocess.PIPE, text=True)
assert child.stdout.readline().strip() == 'child-ready'
print(child.pid, flush=True)
signal.pause()
""")
        process = await GuardedProcess.start(
            [sys.executable, str(script)], cwd=tmp_path, lock_path=tmp_path / "session.lock"
        )
        try:
            child = int(await process.stdout.readline())
            if stop == "eof":
                process.disconnect()  # exactly the private-pipe EOF produced by worker death
                await process.wait()
            else:
                await process.close()
            assert process.returncode is not None
            # Group identity is never reconstructed from an unverified persisted PID.
            with pytest.raises(ProcessLookupError):
                os.killpg(process.group_id, 0)
            assert child > 0
        finally:
            await process.close()


async def test_new_group_waits_for_previous_session_guard(tmp_path):
    from agent.engine_process import GuardedProcess

    async with asyncio.timeout(5):
        lock = tmp_path / "session.lock"
        command = [
            sys.executable,
            "-c",
            "import signal; print('ready', flush=True); signal.pause()",
        ]
        first = await GuardedProcess.start(command, cwd=tmp_path, lock_path=lock)
        second = None
        task = None
        try:
            assert await first.stdout.readline() == b"ready\n"
            task = asyncio.create_task(GuardedProcess.start(command, cwd=tmp_path, lock_path=lock))
            # Explicit guard status barrier, rather than a timing assertion.
            await first.close()
            second = await task
            assert await second.stdout.readline() == b"ready\n"
            assert first.returncode is not None
        finally:
            await first.close()
            if second is not None:
                await second.close()
            elif task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


def test_required_events_reject_unknown_and_malformed():
    from dataclasses import dataclass

    from agent.engine import ApprovalEvent, DoneEvent, TextEvent, ToolCallEvent, UsageEvent
    from agent.engine_runner import validate_event

    @dataclass
    class Unknown:
        kind: str = "unknown-request"

    for event in (
        Unknown(),
        TextEvent(text=3, phase="final"),
        TextEvent(text="x", phase="invalid"),
        DoneEvent(status="invalid", reason=None),
        ToolCallEvent(name="shell", input={}, status="running"),
        UsageEvent(
            input_tokens=-1,
            output_tokens=0,
            cache_tokens=None,
            reported_cost=None,
            estimated_cost=False,
        ),
        ApprovalEvent(request_id="", action="write", parameters={}, scope="request"),
    ):
        with pytest.raises(ValueError):
            validate_event(event)


async def test_teardown_uses_one_deadline_with_controllable_clock():
    from agent.engine_runner import EngineRunner
    from agent.session_fence import RunFence
    from tests.unit.test_engine import profile

    class Clock:
        def __init__(self):
            self.value = 0
            self.timers = []

        def now(self):
            return self.value

        async def until(self, deadline):
            waiter = asyncio.get_running_loop().create_future()
            self.timers.append((deadline, waiter))
            if deadline <= self.value:
                waiter.set_result(None)
            await waiter

        def advance(self, value):
            self.value = value
            for deadline, waiter in self.timers:
                if deadline <= value and not waiter.done():
                    waiter.set_result(None)

    class Engine:
        async def interrupt(self, turn):
            interrupt_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                interrupt_cancelled.set()

        async def close(self):
            close_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                close_cancelled.set()

    async with asyncio.timeout(3):
        interrupt_started, interrupt_cancelled = asyncio.Event(), asyncio.Event()
        close_started, close_cancelled = asyncio.Event(), asyncio.Event()
        clock = Clock()
        runner = EngineRunner(None, Engine(), profile(), RunFence("s", 1, 1, "owner"), clock=clock)
        cleanup = asyncio.create_task(runner.cleanup(interrupted=False))
        await interrupt_started.wait()
        clock.advance(2)
        await close_started.wait()
        await interrupt_cancelled.wait()
        clock.advance(4.5)
        await cleanup
        await close_cancelled.wait()
        assert clock.now() <= 5
