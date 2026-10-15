"""Structured JSON logging and per-session run statistics.

This module provides a JSON logging formatter, helpers to time
operations, and an accumulator for session-level usage statistics.
It is not connected to the application yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from typing import Any, TextIO, cast

from agent.redaction import redact
from agent.usage import UsageRecord

_MAX_VALUE_CHARS = 2000


def _as_text(value: Any) -> str:
    """Return a redacted string, serializing complex values via repr()."""
    if isinstance(value, str):
        text = value
    else:
        try:
            json.dumps(value)
            text = json.dumps(value)
        except (TypeError, ValueError):
            text = repr(value)
    return redact(text)[:_MAX_VALUE_CHARS]


class JsonLogFormatter(logging.Formatter):
    """Format each log record as a single JSON line."""

    def __init__(
        self,
        *,
        include_stack: bool = False,
        secrets: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._include_stack = include_stack
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """Return one JSON line with the record fields."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact_text(record.getMessage()),
        }
        self._add_extra(payload, record)
        self._add_exception(payload, record)
        return json.dumps(payload, ensure_ascii=True, default=_default_json)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Return UTC ISO timestamp for the record."""
        return time.strftime(datefmt or "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))

    def _redact_text(self, value: str) -> str:
        return redact(value, self._secrets)[:_MAX_VALUE_CHARS]

    def _add_extra(self, payload: dict[str, Any], record: logging.LogRecord) -> None:
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {
                "message",
                "asctime",
                "created",
                "msecs",
                "relativeCreated",
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "process",
                "processName",
                "thread",
                "threadName",
                "taskName",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = self._redact_text(value) if isinstance(value, str) else value
            else:
                payload[key] = _as_text(value)

    def _add_exception(self, payload: dict[str, Any], record: logging.LogRecord) -> None:
        if not record.exc_info:
            return
        exc_info = cast(Any, record.exc_info)
        if exc_info is True:
            exc_info = sys.exc_info()
        exc_type, exc_value, _traceback = exc_info
        payload["exc_type"] = exc_type.__name__ if exc_type is not None else None
        payload["exc_message"] = (
            self._redact_text(str(exc_value)) if exc_value is not None else None
        )
        if self._include_stack and exc_info is not None:
            payload["stack"] = self._redact_text(self.formatException(exc_info))


def _default_json(value: Any) -> str:
    """Provide a repr fallback for JSON serialization."""
    return repr(value)


def configure_json_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Install the JSON formatter on the root logger.

    A repeated call does not add a second handler.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in root.handlers:
        if isinstance(handler.formatter, JsonLogFormatter):
            return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)


@contextlib.contextmanager
def timed(name: str, logger: logging.Logger | None = None, **fields: Any) -> Iterator[None]:
    """Run a synchronous block and log its duration.

    The exception is re-raised after the log line is written.
    """
    target = logger if logger is not None else logging.getLogger("agent.observability")
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
        extra: dict[str, Any] = {
            **fields,
            "tool": name,
            "duration_ms": round(duration_ms, 3),
            "outcome": outcome,
            "event": "timed",
        }
        target.info("timed", extra=extra)


@contextlib.asynccontextmanager
async def atimed(
    name: str, logger: logging.Logger | None = None, **fields: Any
) -> AsyncIterator[None]:
    """Run an asynchronous block and log its duration.

    The exception is re-raised after the log line is written.
    """
    target = logger if logger is not None else logging.getLogger("agent.observability")
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
        extra: dict[str, Any] = {
            **fields,
            "tool": name,
            "duration_ms": round(duration_ms, 3),
            "outcome": outcome,
            "event": "timed",
        }
        target.info("timed", extra=extra)


class RunStats:
    """Accumulate usage and timing statistics for one session."""

    def __init__(self) -> None:
        self.steps = 0
        self.llm_calls = 0
        self.tool_calls: Counter[str] = Counter()
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        self.cached_tokens: int | None = None
        self.unknown_tokens = False
        self.cost: Decimal = Decimal(0)
        self.total_duration_ms: float = 0.0
        self.max_duration_ms: float = 0.0

    def record_step(self) -> None:
        """Count one agent step."""
        self.steps += 1

    def record_llm(self, usage: UsageRecord | None = None) -> None:
        """Count an LLM call and merge its token usage."""
        self.llm_calls += 1
        if usage is None:
            self.unknown_tokens = True
            return
        self._merge_tokens("prompt", usage.prompt_tokens, core=True)
        self._merge_tokens("completion", usage.completion_tokens, core=True)
        self._merge_tokens("reasoning", usage.reasoning_tokens)
        self._merge_tokens("cached", usage.cached_tokens)

    def record_tool(self, name: str, duration_ms: float | None = None) -> None:
        """Count a tool call and record its duration."""
        self.tool_calls[name] += 1
        if duration_ms is not None:
            self.record_duration(duration_ms)

    def add_cost(self, value: Decimal | None) -> None:
        """Add a cost value; None marks the total as partially unknown."""
        if value is None:
            self.unknown_tokens = True
        else:
            self.cost += value

    def record_duration(self, duration_ms: float) -> None:
        """Add a duration to total and maximum counters."""
        self.total_duration_ms += duration_ms
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot of the current state."""
        return {
            "steps": self.steps,
            "llm_calls": self.llm_calls,
            "tool_calls": dict(self.tool_calls),
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "reasoning": self.reasoning_tokens,
                "cached": self.cached_tokens,
            },
            "unknown_tokens": self.unknown_tokens,
            "cost": str(self.cost),
            "total_duration_ms": self.total_duration_ms,
            "max_duration_ms": self.max_duration_ms,
        }

    def _merge_tokens(self, name: str, value: int | None, core: bool = False) -> None:
        target_name = f"{name}_tokens"
        if value is None:
            if core:
                self.unknown_tokens = True
            return
        current = getattr(self, target_name)
        if current is None:
            current = 0
        setattr(self, target_name, current + value)
