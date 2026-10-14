"""Test the observability module: JSON logging, timing, and stats."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from decimal import Decimal

import pytest

from agent.observability import (
    JsonLogFormatter,
    RunStats,
    atimed,
    configure_json_logging,
    timed,
)
from agent.usage import UsageRecord

_SK_SECRET = "sk-abcdef123456789"
_SETTINGS_LIKE = f"api_key={_SK_SECRET};user=alice;role=admin"


def _make_record(
    message: str = "hello", extra: dict[str, object] | None = None
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


def test_json_format_basic() -> None:
    formatter = JsonLogFormatter()
    record = _make_record("hello world")
    line = formatter.format(record)
    parsed = json.loads(line)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "agent.test"
    assert parsed["message"] == "hello world"
    assert parsed["ts"].endswith("Z")
    assert len(parsed) >= 4


def test_json_format_extra_fields() -> None:
    formatter = JsonLogFormatter()
    record = _make_record(
        "run started",
        extra={"session_id": "sess-1", "run_generation": 3, "tool": "search"},
    )
    parsed = json.loads(formatter.format(record))
    assert parsed["session_id"] == "sess-1"
    assert parsed["run_generation"] == 3
    assert parsed["tool"] == "search"


def test_json_format_redacts_secrets() -> None:
    formatter = JsonLogFormatter()
    record = _make_record(f"using {_SK_SECRET}")
    parsed = json.loads(formatter.format(record))
    assert _SK_SECRET not in parsed["message"]
    assert "[REDACTED]" in parsed["message"]


def test_json_format_redacts_settings_like_string() -> None:
    formatter = JsonLogFormatter()
    record = _make_record(_SETTINGS_LIKE)
    parsed = json.loads(formatter.format(record))
    assert _SK_SECRET not in parsed["message"]


def test_json_format_extra_redacts_secret() -> None:
    formatter = JsonLogFormatter()
    record = _make_record("ok", extra={"api_key": _SK_SECRET})
    parsed = json.loads(formatter.format(record))
    assert _SK_SECRET not in parsed["api_key"]
    assert "[REDACTED]" in parsed["api_key"]


def test_json_format_exception() -> None:
    formatter = JsonLogFormatter()
    exc_info = None
    try:
        raise ValueError("bad value")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="agent.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    parsed = json.loads(formatter.format(record))
    assert parsed["exc_type"] == "ValueError"
    assert "bad value" in parsed["exc_message"]
    assert "stack" not in parsed


def test_json_format_exception_with_stack() -> None:
    formatter = JsonLogFormatter(include_stack=True)
    exc_info = None
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="agent.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    parsed = json.loads(formatter.format(record))
    assert "stack" in parsed
    assert "RuntimeError" in parsed["stack"]


def test_json_format_non_serializable_value() -> None:
    formatter = JsonLogFormatter()

    class Unserializable:
        def __repr__(self) -> str:
            return "Unserializable()"

    record = _make_record("done", extra={"complex": Unserializable()})
    parsed = json.loads(formatter.format(record))
    assert parsed["complex"] == "Unserializable()"


def test_json_format_truncates_to_2000() -> None:
    formatter = JsonLogFormatter()
    record = _make_record("x" * 5000)
    parsed = json.loads(formatter.format(record))
    assert len(parsed["message"]) <= 2000


def test_configure_json_logging_single_handler() -> None:
    root = logging.getLogger()
    configure_json_logging(level="DEBUG")
    json_handlers = [h for h in root.handlers if isinstance(h.formatter, JsonLogFormatter)]
    assert len(json_handlers) == 1
    configure_json_logging(level="INFO")
    json_handlers = [h for h in root.handlers if isinstance(h.formatter, JsonLogFormatter)]
    assert len(json_handlers) == 1


def test_timed_ok(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with caplog.at_level(logging.INFO, logger=logger.name), timed("op", logger=logger):
        pass
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "timed"
    extras = {
        k: v
        for k, v in record.__dict__.items()
        if k
        not in {
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
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
            "asctime",
        }
    }
    assert extras["tool"] == "op"
    assert extras["outcome"] == "ok"
    assert extras["duration_ms"] >= 0
    assert extras["event"] == "timed"


def test_timed_error_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(ValueError, match="inner"),
        timed("op", logger=logger),
    ):
        raise ValueError("inner")
    record = caplog.records[0]
    extras = {
        k: v
        for k, v in record.__dict__.items()
        if k
        not in {
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
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
            "asctime",
        }
    }
    assert extras["outcome"] == "error"


def test_timed_cancelled_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(asyncio.CancelledError),
        timed("op", logger=logger),
    ):
        raise asyncio.CancelledError()
    record = caplog.records[0]
    extras = {
        k: v
        for k, v in record.__dict__.items()
        if k
        not in {
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
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
            "asctime",
        }
    }
    assert extras["outcome"] == "cancelled"


async def test_atimed_ok(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with caplog.at_level(logging.INFO, logger=logger.name):
        async with atimed("op", logger=logger):
            pass
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "timed"
    assert record.tool == "op"
    assert record.outcome == "ok"
    assert record.duration_ms >= 0


async def test_atimed_error_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(ValueError, match="inner"),
    ):
        async with atimed("op", logger=logger):
            raise ValueError("inner")
    record = caplog.records[0]
    assert record.outcome == "error"


async def test_atimed_cancelled_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with caplog.at_level(logging.INFO, logger=logger.name), pytest.raises(asyncio.CancelledError):
        async with atimed("op", logger=logger):
            raise asyncio.CancelledError()
    record = caplog.records[0]
    assert record.outcome == "cancelled"


def test_run_stats_basic() -> None:
    stats = RunStats()
    stats.record_step()
    stats.record_step()
    stats.record_llm(UsageRecord(prompt_tokens=100, completion_tokens=50))
    stats.record_llm(UsageRecord(prompt_tokens=200, completion_tokens=75))
    stats.record_tool("search", 10.5)
    stats.record_tool("read", 5.0)
    stats.record_tool("search", 3.5)
    stats.record_duration(20.0)
    stats.add_cost(Decimal("0.001"))
    snap = stats.snapshot()
    assert snap["steps"] == 2
    assert snap["llm_calls"] == 2
    assert snap["tool_calls"] == {"search": 2, "read": 1}
    assert snap["tokens"]["prompt"] == 300
    assert snap["tokens"]["completion"] == 125
    assert snap["unknown_tokens"] is False
    assert snap["cost"] == "0.001"
    assert snap["total_duration_ms"] == 39.0
    assert snap["max_duration_ms"] == 20.0


def test_run_stats_none_tokens_flag_unknown() -> None:
    stats = RunStats()
    stats.record_llm(UsageRecord(prompt_tokens=100, completion_tokens=50))
    stats.record_llm(UsageRecord(prompt_tokens=None, completion_tokens=10))
    snap = stats.snapshot()
    assert snap["llm_calls"] == 2
    assert snap["tokens"]["prompt"] == 100
    assert snap["tokens"]["completion"] == 60
    assert snap["unknown_tokens"] is True


def test_run_stats_none_usage_marks_unknown() -> None:
    stats = RunStats()
    stats.record_llm(None)
    snap = stats.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["unknown_tokens"] is True
    assert snap["tokens"]["prompt"] is None


def test_run_stats_reasoning_and_cached() -> None:
    stats = RunStats()
    stats.record_llm(
        UsageRecord(
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=20,
            cached_tokens=30,
        )
    )
    snap = stats.snapshot()
    assert snap["tokens"]["reasoning"] == 20
    assert snap["tokens"]["cached"] == 30


def test_run_stats_decimal_as_string() -> None:
    stats = RunStats()
    stats.add_cost(Decimal("0.123456"))
    stats.add_cost(Decimal("0.000001"))
    snap = stats.snapshot()
    assert isinstance(snap["cost"], str)
    assert snap["cost"] == "0.123457"


def test_run_stats_max_duration() -> None:
    stats = RunStats()
    stats.record_duration(1.5)
    stats.record_duration(3.0)
    stats.record_duration(2.0)
    snap = stats.snapshot()
    assert snap["total_duration_ms"] == 6.5
    assert snap["max_duration_ms"] == 3.0


def test_timed_fields_do_not_override_computed(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        timed("op", logger=logger, outcome="FORCED", duration_ms=999, tool="fake"),
    ):
        pass
    record = caplog.records[0]
    assert record.tool == "op"
    assert record.outcome == "ok"
    assert record.duration_ms >= 0
    assert record.event == "timed"


async def test_atimed_fields_do_not_override_computed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("agent.observability")
    with caplog.at_level(logging.INFO, logger=logger.name):
        async with atimed("op", logger=logger, outcome="FORCED", duration_ms=999, tool="fake"):
            pass
    record = caplog.records[0]
    assert record.tool == "op"
    assert record.outcome == "ok"
    assert record.duration_ms >= 0
    assert record.event == "timed"


def test_run_stats_typical_usage_no_unknown_flag() -> None:
    stats = RunStats()
    stats.record_llm(UsageRecord(prompt_tokens=100, completion_tokens=50))
    snap = stats.snapshot()
    assert snap["tokens"]["prompt"] == 100
    assert snap["tokens"]["completion"] == 50
    assert snap["unknown_tokens"] is False


def test_configure_json_logging_stream_output() -> None:
    buffer = io.StringIO()
    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers = []
    try:
        configure_json_logging(level="INFO", stream=buffer)
        root.info("hello stream")
    finally:
        root.handlers = saved
    output = buffer.getvalue()
    parsed = json.loads(output)
    assert parsed["message"] == "hello stream"
    assert parsed["level"] == "INFO"


def test_timed_system_exit_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("agent.observability")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(SystemExit),
        timed("op", logger=logger),
    ):
        raise SystemExit(1)
    record = caplog.records[0]
    assert record.outcome == "error"


def test_timed_decimal_and_nested_dict_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("agent.observability")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        timed(
            "op",
            logger=logger,
            price=Decimal("1.25"),
            metadata={"layer": "core", "depth": 2},
        ),
    ):
        pass
    record = caplog.records[0]
    formatter = JsonLogFormatter()
    parsed = json.loads(formatter.format(record))
    assert parsed["price"] == "Decimal('1.25')"
    assert parsed["metadata"] == '{"layer": "core", "depth": 2}'


async def test_atimed_decimal_and_nested_dict_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("agent.observability")
    with caplog.at_level(logging.INFO, logger=logger.name):
        async with atimed(
            "op",
            logger=logger,
            price=Decimal("2.50"),
            metadata={"layer": "edge", "depth": 3},
        ):
            pass
    record = caplog.records[0]
    formatter = JsonLogFormatter()
    parsed = json.loads(formatter.format(record))
    assert parsed["price"] == "Decimal('2.50')"
    assert parsed["metadata"] == '{"layer": "edge", "depth": 3}'
