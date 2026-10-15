"""``pyagent`` command: run the server, manage logins and configuration, or talk to the agent.

Kept free of ``agent.main`` imports so auxiliary commands never build the app or supervisor.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


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


HELP_TEXT = "показать справку и выйти"
OPTIONS_TITLE = "параметры"
ARGUMENTS_TITLE = "аргументы"
EXIT_INTERRUPTED = 130
EXAMPLES = """\
Примеры:
  pyagent                       запустить сервер
  pyagent chat                  разговор в текущей папке
  pyagent run "обнови README"   одна задача в текущей папке
  pyagent list                  сессии текущей папки
  pyagent continue              продолжить последнюю сессию
  pyagent stop                  остановить работающую сессию
  pyagent monitor               таблица сессий, обновляется каждые 2 с
"""


class _Formatter(argparse.RawDescriptionHelpFormatter):
    """Keep the examples line by line and print the usage prefix in Russian."""

    def add_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[argparse._MutuallyExclusiveGroup],
        prefix: str | None = None,
    ) -> None:
        # argparse passes prefix="" to build subcommand names; keep it.
        super().add_usage(usage, actions, groups, "использование: " if prefix is None else prefix)


def _localize(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Put the Russian help option and group titles on a parser made with add_help=False."""
    parser._optionals.title = OPTIONS_TITLE
    parser._positionals.title = ARGUMENTS_TITLE
    parser.add_argument("-h", "--help", action="help", help=HELP_TEXT)
    return parser


async def _run_command(argv: Sequence[str], command: str) -> int:
    """Dispatch a CLI client command; returns the process exit code."""
    from agent.cli_client import AgentApi
    from agent.cli_client.commands import build_config

    api = AgentApi(build_config())
    try:
        from agent.cli_client import commands

        project_root = Path.cwd().resolve()
        read = input
        interactive = sys.stdin.isatty()
        if command == "chat":
            return await commands.cmd_chat(api, sys.stdout, read, project_root)
        if command == "run":
            return await commands.cmd_run(
                api, sys.stdout, read, interactive, project_root, " ".join(argv)
            )
        if command == "list":
            return await commands.cmd_list(api, sys.stdout, read, project_root)
        if command == "continue":
            return await commands.cmd_continue(api, sys.stdout, read, project_root)
        if command == "stop":
            return await commands.cmd_stop(api, sys.stdout, read, project_root)
        return 2
    finally:
        await api.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _localize(
        argparse.ArgumentParser(
            prog="pyagent",
            description=(
                "Локальный ИИ-агент. Без аргументов запускается сервер; "
                "подкоманды ниже работают с уже запущенным агентом."
            ),
            epilog=EXAMPLES,
            formatter_class=_Formatter,
            add_help=False,
        )
    )
    commands = parser.add_subparsers(dest="command", title="команды", metavar="<команда>")

    def command(
        name: str, help_text: str, description: str | None = None
    ) -> argparse.ArgumentParser:
        return _localize(
            commands.add_parser(
                name,
                help=help_text,
                add_help=False,
                formatter_class=_Formatter,
                description=description,
            )
        )

    command("serve", "запустить сервер (по умолчанию)")
    command("chat", "разговор с агентом в текущей папке")
    run = command(
        "run",
        'одна задача: pyagent run "что исправить"',
        description="Выполнить одну задачу в текущей папке и напечатать результат.",
    )
    run.add_argument("task", nargs="*", metavar="задача", help="текст задачи")
    command("list", "сессии текущей папки: время, статус, название (последние 20)")
    command("continue", "продолжить последнюю сессию текущей папки в режиме chat")
    command("stop", "остановить работающую сессию текущей папки")
    monitor = command("monitor", "таблица сессий агента (читает локальную базу)")
    monitor.add_argument("--once", action="store_true", help="показать один раз и выйти")
    monitor.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="СЕКУНДЫ",
        help="период обновления, по умолчанию 2 (не меньше 0.5)",
    )

    auth = command("auth", "браузерные входы").add_subparsers(
        dest="auth_command", required=True, title="команды", metavar="<команда>"
    )
    _localize(
        auth.add_parser(
            "link",
            help="вывести новую одноразовую ссылку входа",
            add_help=False,
            formatter_class=_Formatter,
        )
    )
    _localize(
        auth.add_parser(
            "revoke-all",
            help="выйти из всех браузерных сессий",
            add_help=False,
            formatter_class=_Formatter,
        )
    )
    config = command("config", "конфигурация").add_subparsers(
        dest="config_command", required=True, title="команды", metavar="<команда>"
    )
    _localize(
        config.add_parser(
            "check",
            help="проверить config/models.yaml [--file PATH] [--strict]",
            add_help=False,
            formatter_class=_Formatter,
        )
    )

    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:2] == ["config", "check"]:
        # check_main owns its own options (--file, --strict).
        from agent.model_config import check_main

        return check_main(raw[2:])
    args = parser.parse_args(raw)
    if args.command in (None, "serve"):
        from agent.main import run as run_server

        run_server()
        return 0
    if args.command == "auth" and args.auth_command == "link":
        print(asyncio.run(_auth_link()))
        return 0
    if args.command == "auth" and args.auth_command == "revoke-all":
        print(f"revoked: {asyncio.run(_auth_revoke_all())}")
        return 0
    if args.command == "monitor":
        from agent.cli_client.monitor_view import run_monitor

        try:
            return asyncio.run(run_monitor(sys.stdout, once=args.once, interval=args.interval))
        except KeyboardInterrupt:
            # Ctrl+C is the normal way to close the monitor.
            print()
            return 0
    if args.command in ("chat", "run", "list", "continue", "stop"):
        rest = list(args.task) if args.command == "run" else []
        try:
            return asyncio.run(_run_command(rest, args.command))
        except KeyboardInterrupt:
            # Ctrl+C outside an agent answer ends the command without a traceback.
            print()
            return EXIT_INTERRUPTED
    parser.error("неизвестная команда")
    return 2
