"""Stdlib-only watchdog, executed as a separate interpreter by engine_process.

The worker alone owns the control writer. CLI/tool children never inherit either
control or status descriptors. A per-session flock remains held until the entire
CLI process group is gone; subsequent guards cannot start a replacement early.
"""

from __future__ import annotations

import fcntl
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import suppress


def group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


def stop_group(process: subprocess.Popen[bytes], exited: list[bool]) -> bool:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while not exited[0] and time.monotonic() < deadline:
        select.select([], [], [], min(0.01, max(0, deadline - time.monotonic())))
    # Keep the direct child unreaped until the LAST group signal. Its PID cannot
    # be recycled into an unrelated group between TERM and KILL.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + 0.4
    try:
        process.wait(timeout=0.4)
    except subprocess.TimeoutExpired:
        return False
    while group_exists(process.pid) and time.monotonic() < deadline:
        select.select([], [], [], min(0.01, max(0, deadline - time.monotonic())))
    return not group_exists(process.pid)


def main() -> int:
    control, status = int(sys.argv[1]), int(sys.argv[2])
    os.set_inheritable(control, False)
    os.set_inheritable(status, False)
    with open(sys.argv[3], "a+b") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if select.select([control], [], [], 0.01)[0] and not os.read(control, 1):
                    return 0
        if select.select([control], [], [], 0)[0] and not os.read(control, 1):
            return 0
        exited = [False]
        signal.signal(signal.SIGCHLD, lambda *_: exited.__setitem__(0, True))
        process = subprocess.Popen(sys.argv[4:], start_new_session=True, close_fds=True)
        os.write(status, f"{process.pid}\n".encode())
        os.close(status)
        while not exited[0]:
            if select.select([control], [], [], 0.02)[0] and not os.read(control, 1):
                break
        gone = stop_group(process, exited)
        if not gone:
            # Fail closed: retain the flock until absence can be confirmed. The
            # worker's wait remains bounded, and a replacement cannot overlap.
            while group_exists(process.pid):
                process.poll()  # No more signals: reaping cannot cause a foreign kill.
                select.select([], [], [], 0.02)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
