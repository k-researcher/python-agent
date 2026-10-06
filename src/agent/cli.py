"""``agent`` command: run the server or manage logins and configuration.

Kept free of ``agent.main`` imports so auxiliary commands never build the app or supervisor.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence


async def _auth_link() -> str:
    from agent.auth import issue_login_code, login_url
    from agent.config import get_settings
    from agent.database import init_database, session_factory

    settings = get_settings()
    await init_database(recover_running=False)
    async with session_factory() as db:
        return login_url(settings, await issue_login_code(db, settings))


async def _auth_revoke_all() -> int:
    from agent.auth import revoke
    from agent.database import init_database, session_factory

    await init_database(recover_running=False)
    async with session_factory() as db:
        return await revoke(db)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent", description="Python Agent")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the API and UI (default)")
    auth = commands.add_parser("auth", help="browser logins").add_subparsers(
        dest="auth_command", required=True
    )
    auth.add_parser("link", help="print a new one-time login link")
    auth.add_parser("revoke-all", help="log out every browser session")
    config = commands.add_parser("config", help="configuration").add_subparsers(
        dest="config_command", required=True
    )
    config.add_parser("check", help="validate config/models.yaml [--file PATH] [--strict]")

    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:2] == ["config", "check"]:
        # check_main owns its own options (--file, --strict).
        from agent.model_config import check_main

        return check_main(raw[2:])
    args = parser.parse_args(raw)
    if args.command in (None, "serve"):
        from agent.main import run

        run()
        return 0
    if args.command == "auth" and args.auth_command == "link":
        print(asyncio.run(_auth_link()))
        return 0
    if args.command == "auth" and args.auth_command == "revoke-all":
        print(f"revoked: {asyncio.run(_auth_revoke_all())}")
        return 0
    parser.error("unknown command")
    return 2
