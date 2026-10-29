"""Neutral CLI plumbing: restricted environment, bounded JSON and guarded groups."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

_ALLOWED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
MAX_LINE_BYTES = 1024 * 1024


def clean_env() -> dict[str, str]:
    """Omit secret environment variables. HOME does not hide credential files.

    Filesystem isolation and authentication checks are the adapter's obligation.
    """
    return {name: os.environ[name] for name in _ALLOWED_ENV if name in os.environ}


def decode_line(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError("Engine line exceeds 1 MiB")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Malformed engine JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Engine payload must be an object")
    return value


async def terminate_group(process: asyncio.subprocess.Process, *, grace: float = 2) -> None:
    """Stop an owned process group, with bounded waits even after its leader exited."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(grace):
            await process.wait()
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    async with asyncio.timeout(0.5):
        await process.wait()


class GuardedProcess:
    """A live CLI whose guard is the only process allowed to signal its group."""

    def __init__(self, process: asyncio.subprocess.Process, control: int) -> None:
        self.process, self._control = process, control
        assert process.stdout is not None
        self.stdout = process.stdout
        self.stdin = process.stdin
        self.group_id = 0
        self.stderr_tail = b""
        self._stderr_task = asyncio.create_task(self._stderr())

    @classmethod
    async def start(cls, command: list[str], *, cwd: Path, lock_path: Path) -> GuardedProcess:
        control_r, control_w = os.pipe()
        status_r, status_w = os.pipe()
        result = None
        try:
            process = await asyncio.create_subprocess_exec(
                str(Path(sys.executable).resolve()),
                str(Path(__file__).with_name("engine_guard.py")),
                str(control_r),
                str(status_w),
                str(lock_path),
                *command,
                cwd=cwd,
                env=clean_env(),
                pass_fds=(control_r, status_w),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=MAX_LINE_BYTES + 1,
            )
            os.close(control_r)
            control_r = -1
            os.close(status_w)
            status_w = -1
            result = cls(process, control_w)
            control_w = -1
            # The status descriptor contains only the guard's group handshake.
            ready = await asyncio.to_thread(os.read, status_r, 64)
            if not ready:
                raise RuntimeError("Engine guard failed before start")
            result.group_id = int(ready)
            return result
        except BaseException:
            if result is not None:
                await result.close()
            raise
        finally:
            for fd in (control_r, control_w, status_r, status_w):
                if fd >= 0:
                    os.close(fd)

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def disconnect(self) -> None:
        if self._control >= 0:
            os.close(self._control)
            self._control = -1

    async def _stderr(self) -> None:
        assert self.process.stderr is not None
        while chunk := await self.process.stderr.read(8192):
            self.stderr_tail = (self.stderr_tail + chunk)[-2000:]

    async def wait(self) -> int:
        async with asyncio.timeout(3):
            result = await self.process.wait()
            await self._stderr_task
            return result

    async def close(self) -> None:
        self.disconnect()
        if self.stdin is not None:
            self.stdin.close()
        # Do not kill a guard on timeout: it retains the session lock while an
        # unkillable group exists, preventing a second group from overlapping.
        await self.wait()
