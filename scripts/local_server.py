"""Manage one local SQLite server without nesting its child sandboxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

TABLES = (
    "projects",
    "sessions",
    "messages",
    "llm_provider_overrides",
    "llm_model_overrides",
    "speech_profiles",
    "mcp_servers",
    "shared_skill_defaults",
    "project_skill_settings",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def quote_path(path: Path) -> str:
    text = str(path)
    require(not any(ord(char) < 32 or ord(char) == 127 for char in text), "Invalid path")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_private(path: Path, value: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def read_private(path: Path) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as source:
        info = os.fstat(source.fileno())
        require(
            info.st_uid == os.getuid() and info.st_mode & 0o777 == 0o600,
            "Control file must be private and owned by this user",
        )
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError("Invalid control file")
    return value


def git(checkout: Path, *args: str) -> str:
    return (
        subprocess.check_output(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(checkout),
                *args,
            ],
            timeout=15,
        )
        .decode()
        .strip()
    )


@dataclass(frozen=True)
class Configuration:
    checkout: Path
    data_dir: Path
    database: Path
    port: int
    env_file: Path | None = None
    models_file: Path | None = None

    def validate(self) -> None:
        require(1 <= self.port <= 65535, "Invalid port")
        for path in (self.checkout, self.data_dir, self.database):
            require(path.is_absolute() and path == path.resolve(), "Paths must be canonical")
            quote_path(path)
        require(self.checkout.is_dir() and self.data_dir.is_dir(), "Directories must exist")
        info = self.data_dir.stat()
        require(
            info.st_uid == os.getuid() and not info.st_mode & 0o022,
            "Data directory must be owned by this user and not writable by others",
        )
        require(
            self.database.is_file() and self.database.is_relative_to(self.data_dir),
            "An existing SQLite database inside the data directory is required",
        )
        require(
            not self.data_dir.is_relative_to(self.checkout)
            and not self.checkout.is_relative_to(self.data_dir),
            "Keep the checkout and data directory separate",
        )
        for configuration_file in (self.env_file, self.models_file):
            require(
                configuration_file is None or configuration_file.is_file(),
                "Configuration file is missing",
            )

    def payload(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Configuration:
        values = dict(payload)
        for key in ("checkout", "data_dir", "database", "env_file", "models_file"):
            if values[key] is not None:
                values[key] = Path(values[key])
        config = cls(**values)
        config.validate()
        return config


class LocalServer:
    def __init__(self, config: Configuration):
        self.config = config

    def preflight(self) -> None:
        require(sys.platform == "darwin", "This controller requires the macOS sandbox backend")
        result = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
        require(
            result.returncode == 0,
            "Cannot create child sandboxes here; run the trusted controller from the host",
        )
        self.config.validate()
        require(
            (self.config.checkout / ".venv/bin/pyagent").is_file(), "Server executable is missing"
        )

    @contextmanager
    def lock(self) -> Iterator[None]:
        import fcntl

        database = self.config.database
        fd = os.open(
            str(database) + ".deployment.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = {key: self.config.payload()[key] for key in ("checkout", "database", "port")}
            path = Path(str(database) + ".deployment.json")
            if path.exists() or path.is_symlink():
                require(
                    read_private(path) == identity,
                    "This database is assigned to another checkout or port",
                )
            else:
                write_private(path, identity)
            yield
        finally:
            os.close(fd)

    def listeners(self) -> set[int]:
        result = subprocess.run(
            ["lsof", "-t", f"-iTCP:{self.config.port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        require(result.returncode in (0, 1) and not result.stderr, "Cannot inspect the listener")
        return {int(value) for value in result.stdout.split()}

    def owns_listener(self, pid: int) -> bool:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (
            result.returncode == 0 and "n" + str(self.config.checkout) in result.stdout.splitlines()
        )

    def environment(self) -> dict[str, str]:
        config = self.config
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("AGENT_", "PYTHON", "DYLD_", "LD_"))
        }
        env.update(
            AGENT_ENV_FILE=str(config.env_file) if config.env_file else "",
            AGENT_MODELS_FILE=str(config.models_file) if config.models_file else "",
            AGENT_HOST="127.0.0.1",
            AGENT_PORT=str(config.port),
            AGENT_DATA_DIR=str(config.data_dir),
            AGENT_DATABASE_URL=f"sqlite+aiosqlite:///{config.database}",
            AGENT_FRONTEND_DIST=str(config.checkout / "frontend/dist"),
            AGENT_SHELL_SANDBOX="required",
            AGENT_AUTH_OPEN_BROWSER="false",
            PYTHONPATH=str(config.checkout / "src"),
            PYTHONNOUSERSITE="1",
        )
        return env

    def snapshot(self) -> dict[str, Any]:
        with closing(sqlite3.connect(self.config.database.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("begin")
            require(
                not db.execute(
                    "select 1 from sessions where status in "
                    "('pending','running','awaiting_confirmation') limit 1"
                ).fetchone(),
                "Active sessions must not be interrupted",
            )
            require(db.execute("pragma quick_check").fetchone()[0] == "ok", "Database check failed")
            require(
                not db.execute("pragma foreign_key_check").fetchall(), "Foreign key check failed"
            )
            names = {
                row[0] for row in db.execute("select name from sqlite_master where type='table'")
            }
            counts, fingerprints = {}, {}
            for table in TABLES:
                if table not in names:
                    continue
                rows = sorted(
                    json.dumps(list(row), default=bytes.hex)
                    for row in db.execute(f"select * from {table}")
                )
                counts[table] = len(rows)
                fingerprints[table] = hashlib.sha256("\n".join(rows).encode()).hexdigest()
            return {
                "counts": counts,
                "fingerprints": fingerprints,
                "revision": db.execute("select version_num from alembic_version").fetchone()[0],
            }

    def start(self) -> int:
        self.preflight()
        require(not self.listeners(), "Port is occupied; no process was started")
        fd = os.open(
            self.config.data_dir / "server.log",
            os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "a") as log:
            os.fchmod(log.fileno(), 0o600)
            process = subprocess.Popen(
                [str(self.config.checkout / ".venv/bin/pyagent")],
                cwd=self.config.checkout,
                env=self.environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                require(process.poll() is None, "Startup failed; inspect the private server log")
                if self.listeners() == {process.pid} and self.owns_listener(process.pid):
                    try:
                        url = f"http://127.0.0.1:{self.config.port}/api/health"
                        with opener.open(url, timeout=1) as response:
                            if (
                                response.status == 200
                                and json.loads(response.read(4096))["status"] == "ok"
                            ):
                                return process.pid
                    except OSError:
                        pass
                time.sleep(0.2)
            raise RuntimeError("Server health check timed out")
        except BaseException:
            # The unreaped child belongs to this invocation; no persisted PID is signalled.
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            raise

    def validate_update(self, kind: str, source: Path, old: str, new: str) -> int:
        target = self.config.checkout
        require(git(target, "rev-parse", "HEAD") == old, "Server base changed")
        require(
            not git(target, "status", "--porcelain", "--untracked-files=no"), "Server tree is dirty"
        )
        require(
            git(target, "rev-parse", "--verify", new + "^{commit}") == new,
            "Target commit is missing",
        )
        require(git(source, "rev-parse", "HEAD") == new, "Source commit changed")
        require(
            not git(source, "status", "--porcelain", "--untracked-files=no"), "Source tree is dirty"
        )
        changed = git(source, "diff", "--name-only", old, new).splitlines()
        if kind == "restart":
            require(old == new, "Restart must preserve the commit")
        else:
            require(kind in ("full", "backend") and old != new, "Invalid deployment kind")
        if kind == "backend":
            require(
                not any(name.startswith("frontend/") for name in changed),
                "Frontend changes require full deployment",
            )
        if kind == "full":
            require(source != target, "Use separate source and server checkouts")
            dist = source / "frontend/dist"
            newest = max(
                path.stat().st_mtime
                for path in (source / "frontend/src").rglob("*")
                if path.is_file()
            )
            require((dist / "index.html").stat().st_mtime > newest, "Frontend build is stale")
            require(
                not any(path.is_symlink() for path in dist.rglob("*")),
                "Build contains symbolic links",
            )
        pids = self.listeners()
        require(len(pids) == 1, "Expected exactly one server listener")
        pid = next(iter(pids))
        require(self.owns_listener(pid), "Listener belongs to another directory")
        self.snapshot()
        return pid

    def backup_database(self, backup: Path) -> None:
        with (
            closing(sqlite3.connect(self.config.database.as_uri() + "?mode=ro", uri=True)) as db,
            closing(sqlite3.connect(backup / "database.sqlite")) as target,
        ):
            deadline = time.monotonic() + 15

            def progress(*_: Any) -> None:
                require(time.monotonic() < deadline, "Database backup timed out")

            db.backup(target, pages=100, progress=progress)
        (backup / "database.sqlite").chmod(0o600)

    def preparation_profile(self, temporary: Path) -> str:
        common = Path(
            git(self.config.checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")
        ).resolve()
        writable = (self.config.checkout, self.config.data_dir, common, temporary)
        require(
            all(path != Path(path.anchor) and path != Path.home() for path in writable),
            "Preparation write path is too broad",
        )
        filters = " ".join(f"(subpath {quote_path(path)})" for path in writable)
        return (
            "(version 1)\n(deny default)\n(allow file-read*)\n(allow process*)\n"
            "(allow signal)\n(allow sysctl-read)\n(allow mach*)\n(allow ipc-posix*)\n"
            f'(allow file-write* {filters} (literal "/dev/null"))\n'
        )

    def prepare(self, kind: str, source: Path, old: str, new: str, backup: Path) -> None:
        pid = self.validate_update(kind, source, old, new)
        backup.mkdir(mode=0o700)
        dist = self.config.checkout / "frontend/dist"
        if dist.is_dir():
            require(
                not any(path.is_symlink() for path in dist.rglob("*")),
                "Server build contains symbolic links",
            )
            shutil.copytree(dist, backup / "dist")
        for name in git(source, "diff", "--name-only", old, new).splitlines():
            previous = self.config.checkout / name
            if previous.is_file():
                destination = backup / "code" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(previous, destination)
        self.backup_database(backup)
        require(
            self.validate_update(kind, source, old, new) == pid, "Listener changed during backup"
        )
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Server did not stop; code was not replaced")
        state = self.snapshot()
        self.backup_database(backup)
        if old != new:
            git(self.config.checkout, "checkout", "--detach", new)
        if kind == "full":
            for item in (source / "frontend/dist").rglob("*"):
                if item.is_file() and item.name != "index.html":
                    destination = dist / item.relative_to(source / "frontend/dist")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, destination)
            shutil.copy2(source / "frontend/dist/index.html", dist / "index.next.html")
            (dist / "index.next.html").replace(dist / "index.html")
        write_private(
            backup / "prepared.json",
            {"kind": kind, "from": old, "to": new, "old_pid": pid, "state": state},
        )

    def update(self, kind: str, source: Path, old: str, new: str) -> dict[str, Any]:
        self.preflight()
        with self.lock():
            pid = self.validate_update(kind, source, old, new)
            parent = self.config.data_dir / "backups"
            parent.mkdir(mode=0o700, exist_ok=True)
            backup = parent / datetime.now().strftime("server-%Y%m%d-%H%M%S-%f")
            with tempfile.TemporaryDirectory(prefix="local-server-") as raw:
                temporary = Path(raw).resolve()
                profile = temporary / "prepare.sb"
                profile.write_text(self.preparation_profile(temporary))
                arguments = temporary / "prepare.json"
                write_private(
                    arguments,
                    {
                        "config": self.config.payload(),
                        "kind": kind,
                        "source": str(source),
                        "old": old,
                        "new": new,
                        "backup": str(backup),
                    },
                )
                subprocess.run(
                    [
                        "/usr/bin/sandbox-exec",
                        "-f",
                        str(profile),
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_prepare",
                        str(arguments),
                    ],
                    check=True,
                    timeout=120,
                )
            receipt = read_private(backup / "prepared.json")
            require(
                (receipt["kind"], receipt["from"], receipt["to"], receipt["old_pid"])
                == (kind, old, new, pid),
                "Preparation receipt does not match this operation",
            )
            require(
                git(self.config.checkout, "rev-parse", "HEAD") == new, "Prepared commit is invalid"
            )
            require(not self.listeners(), "Port occupied after preparation")
            require(self.snapshot() == receipt["state"], "Data changed after preparation")
            new_pid = self.start()
            after = self.snapshot()
            require(
                all(
                    after["counts"].get(table) == count
                    for table, count in receipt["state"]["counts"].items()
                ),
                "Data counts changed during startup",
            )
            if kind == "restart":
                require(after == receipt["state"], "Restart changed data or schema")
            with closing(
                sqlite3.connect(self.config.database.as_uri() + "?mode=ro", uri=True)
            ) as db:
                require(
                    {"url", "header_secrets", "http_protocol"}
                    <= {row[1] for row in db.execute("pragma table_info(mcp_servers)")},
                    "MCP HTTP migration is missing",
                )
            receipt.update(pid=new_pid, backup=str(backup), revision=after["revision"])
            write_private(backup / "completed.json", receipt)
            return {key: receipt[key] for key in ("pid", "from", "to", "backup", "revision")}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("operation", choices=("preflight", "start", "restart", "deploy"))
    for name in ("checkout", "data-dir", "database"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--port", type=int, required=True)
    result.add_argument("--env-file", type=Path)
    result.add_argument("--models-file", type=Path)
    result.add_argument("--source", type=Path)
    result.add_argument("--from-commit")
    result.add_argument("--to-commit")
    result.add_argument("--backend-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == "_prepare":
        payload = read_private(Path(args[1]))
        config = Configuration.from_payload(payload["config"])
        backup = Path(payload["backup"])
        require(
            backup.parent == config.data_dir / "backups" and backup.name.startswith("server-"),
            "Invalid backup destination",
        )
        LocalServer(config).prepare(
            payload["kind"], Path(payload["source"]), payload["old"], payload["new"], backup
        )
        return
    options = parser().parse_args(args)
    config = Configuration(
        checkout=options.checkout.resolve(),
        data_dir=options.data_dir.resolve(),
        database=options.database.resolve(),
        port=options.port,
        env_file=options.env_file.resolve() if options.env_file is not None else None,
        models_file=options.models_file.resolve() if options.models_file is not None else None,
    )
    server = LocalServer(config)
    if options.operation == "preflight":
        server.preflight()
        print("Child sandbox creation is available")
    elif options.operation == "start":
        server.preflight()
        with server.lock():
            print(json.dumps({"pid": server.start()}))
    else:
        source = options.source.resolve() if options.source else config.checkout
        if options.operation == "restart":
            old = new = git(config.checkout, "rev-parse", "HEAD")
            kind = "restart"
        else:
            require(
                options.from_commit is not None and options.to_commit is not None,
                "Deployment requires both commits",
            )
            old = git(source, "rev-parse", "--verify", options.from_commit + "^{commit}")
            new = git(source, "rev-parse", "--verify", options.to_commit + "^{commit}")
            kind = "backend" if options.backend_only else "full"
        print(json.dumps(server.update(kind, source, old, new)))


if __name__ == "__main__":
    main()
