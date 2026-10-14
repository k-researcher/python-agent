"""Test the deterministic decision layer."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from agent.decision_engine import (
    ApprovalItem,
    Decision,
    FallbackEngine,
    RuleEngine,
    TaskFeatures,
)
from agent.tools.base import RiskLevel

ALL_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def task(
    *,
    prompt_chars: int = 100,
    has_code: bool = False,
    tool_count_so_far: int = 0,
    failed_attempts: int = 0,
    mode: str = "dev",
) -> TaskFeatures:
    return TaskFeatures(prompt_chars, has_code, tool_count_so_far, failed_attempts, mode)


def approval(
    item_id: str, risk: RiskLevel, waiting: float = 1.0, tool: str = "tool"
) -> ApprovalItem:
    return ApprovalItem(item_id, tool, risk, waiting)


class FailingEngine:
    """Engine that raises on every call."""

    async def choose_effort(self, task, allowed):
        raise RuntimeError("primary is broken")

    async def choose_model(self, task, candidates):
        raise RuntimeError("primary is broken")

    async def rank_approvals(self, items):
        raise RuntimeError("primary is broken")


class SlowEngine:
    """Engine that sleeps longer than any reasonable timeout."""

    async def choose_effort(self, task, allowed):
        await asyncio.sleep(10)
        return Decision("high", "slow", 1.0, "slow")

    async def choose_model(self, task, candidates):
        await asyncio.sleep(10)
        return Decision(candidates[0], "slow", 1.0, "slow")

    async def rank_approvals(self, items):
        await asyncio.sleep(10)
        return Decision([], "slow", 1.0, "slow")


class InvalidEngine:
    """Engine that returns values outside the allowed set."""

    async def choose_effort(self, task, allowed):
        return Decision("bogus", "wrong", 1.0, "invalid")

    async def choose_model(self, task, candidates):
        return Decision("not-in-candidates", "wrong", 1.0, "invalid")

    async def rank_approvals(self, items):
        return Decision(["nope"], "wrong", 1.0, "invalid")


class RecordingEngine:
    """Engine that returns a preset decision."""

    def __init__(self, result: Decision) -> None:
        self._result = result

    async def choose_effort(self, task, allowed):
        return self._result

    async def choose_model(self, task, candidates):
        return self._result

    async def rank_approvals(self, items):
        return self._result


# ---------------------------------------------------------------- data types


def test_decision_is_frozen_and_holds_fields() -> None:
    decision = Decision("low", "reason one", 0.9, "rule")
    assert decision.value == "low"
    assert decision.reason == "reason one"
    assert decision.confidence == 0.9
    assert decision.engine == "rule"
    with pytest.raises(FrozenInstanceError):
        decision.value = "high"  # type: ignore[misc]


def test_task_features_frozen() -> None:
    features = task()
    with pytest.raises(FrozenInstanceError):
        features.prompt_chars = 999  # type: ignore[misc]


def test_approval_item_frozen() -> None:
    item = approval("a1", RiskLevel.read_only)
    with pytest.raises(FrozenInstanceError):
        item.id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------- effort


async def test_effort_ask_mode_is_low() -> None:
    chosen = await RuleEngine().choose_effort(task(mode="ask"), ALL_EFFORTS)
    assert chosen.value == "low"
    assert chosen.engine == "rule"


async def test_effort_dev_mode_is_medium() -> None:
    chosen = await RuleEngine().choose_effort(task(mode="dev"), ALL_EFFORTS)
    assert chosen.value == "medium"


async def test_effort_code_task_goes_higher() -> None:
    chosen = await RuleEngine().choose_effort(task(has_code=True), ALL_EFFORTS)
    assert chosen.value == "high"


async def test_effort_long_task_goes_higher() -> None:
    chosen = await RuleEngine().choose_effort(task(prompt_chars=9000), ALL_EFFORTS)
    assert chosen.value == "high"


async def test_effort_many_tools_goes_higher() -> None:
    chosen = await RuleEngine().choose_effort(task(tool_count_so_far=10), ALL_EFFORTS)
    assert chosen.value == "high"


async def test_effort_one_failed_step_goes_higher() -> None:
    chosen = await RuleEngine().choose_effort(task(failed_attempts=1), ALL_EFFORTS)
    assert chosen.value == "high"


async def test_effort_two_failed_steps_goes_max() -> None:
    chosen = await RuleEngine().choose_effort(task(failed_attempts=2), ALL_EFFORTS)
    assert chosen.value == "max"


async def test_effort_two_failed_matches_reasoning_auto() -> None:
    from agent.model_config import ReasoningSpec, ResolvedModel
    from agent.reasoning import AUTO, decide

    model = ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url="https://llm.example.com/v1",
        api_key="",
        configured=True,
        model="m",
        context_window=32000,
        max_tokens=4000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(
            supported=True,
            allowed_efforts=list(ALL_EFFORTS),
            default_effort="medium",
            wire_parameter="reasoning_effort",  # type: ignore[arg-type]
        ),
        pricing=None,
    )
    rule_value = (
        await RuleEngine().choose_effort(task(mode="dev", failed_attempts=2), ALL_EFFORTS)
    ).value
    auto_value = decide(
        model, message_effort=AUTO, session_effort=None, mode="dev", recent_errors=2
    ).effective
    assert rule_value == auto_value == "max"


async def test_effort_combined_signals_go_higher() -> None:
    chosen = await RuleEngine().choose_effort(
        task(has_code=True, prompt_chars=9000, failed_attempts=2), ALL_EFFORTS
    )
    assert chosen.value == "max"


async def test_effort_picks_nearest_allowed() -> None:
    engine = RuleEngine()
    allowed = ("none", "high")
    decided = await engine.choose_effort(task(mode="dev"), allowed)
    assert decided.value == "high"
    # Ask mode uses "low". With a tie, the higher level wins.
    allowed_ask = ("none", "high")
    decided_ask = await engine.choose_effort(task(mode="ask"), allowed_ask)
    assert decided_ask.value == "high"


async def test_effort_raises_on_empty_allowed() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await RuleEngine().choose_effort(task(), [])


async def test_effort_raises_when_allowed_has_no_valid_levels() -> None:
    with pytest.raises(ValueError, match="no valid effort"):
        await RuleEngine().choose_effort(task(), ("bogus-1", "bogus-2"))


# ---------------------------------------------------------------- model


async def test_model_raises_on_empty_candidates() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await RuleEngine().choose_model(task(), [])


async def test_model_returns_first_candidate() -> None:
    chosen = await RuleEngine().choose_model(task(), ["fast", "smart", "cheap"])
    assert chosen.value == "fast"
    assert chosen.engine == "rule"


# ---------------------------------------------------------------- approvals


async def test_rank_approvals_sorts_by_risk() -> None:
    engine = RuleEngine()
    items = [
        approval("a", RiskLevel.read_only),
        approval("b", RiskLevel.destructive),
        approval("c", RiskLevel.local_write),
    ]
    result = await engine.rank_approvals(items)
    assert result.value == ["b", "c", "a"]


async def test_rank_approvals_ties_by_waiting_time() -> None:
    engine = RuleEngine()
    items = [
        approval("a", RiskLevel.local_write, waiting=1.0),
        approval("b", RiskLevel.local_write, waiting=5.0),
    ]
    result = await engine.rank_approvals(items)
    assert result.value == ["b", "a"]


async def test_rank_approvals_ties_by_id() -> None:
    engine = RuleEngine()
    items = [
        approval("z", RiskLevel.read_only, waiting=2.0),
        approval("a", RiskLevel.read_only, waiting=2.0),
    ]
    result = await engine.rank_approvals(items)
    assert result.value == ["z", "a"]


async def test_rank_approvals_empty_returns_empty_ids() -> None:
    result = await RuleEngine().rank_approvals([])
    assert result.value == []


# ---------------------------------------------------------------- determinism


async def test_effort_is_deterministic() -> None:
    engine = RuleEngine()
    features = task(has_code=True, prompt_chars=5000, failed_attempts=1)
    first = await engine.choose_effort(features, ALL_EFFORTS)
    second = await engine.choose_effort(features, ALL_EFFORTS)
    assert first == second


async def test_model_is_deterministic() -> None:
    engine = RuleEngine()
    first = await engine.choose_model(task(), ["m1", "m2", "m3"])
    second = await engine.choose_model(task(), ["m1", "m2", "m3"])
    assert first == second


async def test_rank_approvals_is_deterministic() -> None:
    engine = RuleEngine()
    items = [
        approval("x", RiskLevel.network_access, waiting=3.0),
        approval("y", RiskLevel.destructive, waiting=1.0),
        approval("z", RiskLevel.destructive, waiting=5.0),
    ]
    first = await engine.rank_approvals(items)
    second = await engine.rank_approvals(items)
    assert first == second


async def test_decision_value_is_always_from_allowed() -> None:
    engine = RuleEngine()
    allowed = ("low", "high")
    for cases in (
        task(mode="ask"),
        task(mode="dev"),
        task(has_code=True),
        task(prompt_chars=9000),
        task(failed_attempts=2),
    ):
        result = await engine.choose_effort(cases, allowed)
        assert result.value in allowed


# ---------------------------------------------------------------- fallback


async def test_fallback_on_exception() -> None:
    primary = FailingEngine()
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.choose_effort(task(), ALL_EFFORTS)
    assert result.value in ALL_EFFORTS
    assert "primary failed" in result.reason


async def test_fallback_on_timeout() -> None:
    primary = SlowEngine()
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.choose_effort(task(), ALL_EFFORTS)
    assert result.value in ALL_EFFORTS
    assert "primary failed" in result.reason


async def test_fallback_on_invalid_effort() -> None:
    primary = InvalidEngine()
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.choose_effort(task(), ALL_EFFORTS)
    assert result.value in ALL_EFFORTS
    assert "invalid value" in result.reason


async def test_fallback_on_invalid_model() -> None:
    primary = InvalidEngine()
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.choose_model(task(), ["fast", "smart"])
    assert result.value in ("fast", "smart")
    assert "invalid value" in result.reason


async def test_fallback_on_invalid_approvals() -> None:
    primary = InvalidEngine()
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    items = [
        approval("a", RiskLevel.read_only),
        approval("b", RiskLevel.local_write),
    ]
    result = await engine.rank_approvals(items)
    assert set(result.value) == {"a", "b"}
    assert "invalid value" in result.reason


async def test_fallback_on_incomplete_approvals() -> None:
    items = [approval("a", RiskLevel.read_only), approval("b", RiskLevel.local_write)]
    primary = RecordingEngine(Decision(["a"], "incomplete", 1.0, "recording"))
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.rank_approvals(items)
    assert set(result.value) == {"a", "b"}
    assert "invalid value" in result.reason


async def test_fallback_on_duplicate_approvals() -> None:
    items = [approval("a", RiskLevel.read_only), approval("b", RiskLevel.local_write)]
    primary = RecordingEngine(Decision(["a", "a"], "duplicated", 1.0, "recording"))
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.rank_approvals(items)
    assert set(result.value) == {"a", "b"}
    assert "invalid value" in result.reason


async def test_fallback_drops_all_approvals() -> None:
    items = [approval("a", RiskLevel.read_only), approval("b", RiskLevel.local_write)]
    primary = RecordingEngine(Decision([], "empty", 1.0, "recording"))
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.rank_approvals(items)
    assert set(result.value) == {"a", "b"}
    assert "invalid value" in result.reason


async def test_fallback_passes_through_primary_success() -> None:
    primary = RecordingEngine(Decision("low", "primary says", 0.9, "recording"))
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.choose_effort(task(mode="ask"), ALL_EFFORTS)
    assert result.value == "low"
    assert result.reason == "primary says"
    assert result.engine == "recording"


async def test_fallback_model_pass_through() -> None:
    primary = RecordingEngine(Decision("fast", "primary says", 0.9, "recording"))
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.choose_model(task(), ["fast", "smart"])
    assert result.value == "fast"
    assert "primary says" in result.reason


async def test_fallback_approvals_pass_through() -> None:
    items = [approval("a", RiskLevel.read_only), approval("b", RiskLevel.local_write)]
    primary = RecordingEngine(Decision(["b", "a"], "primary says", 0.9, "recording"))
    fallback = RuleEngine()
    engine = FallbackEngine(primary, fallback, timeout_seconds=0.2)
    result = await engine.rank_approvals(items)
    assert result.value == ["b", "a"]
    assert "primary says" in result.reason
