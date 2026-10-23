"""Deterministic decision layer for task effort, model choice and approvals."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from agent.modes import tool_profile
from agent.reasoning import EFFORT_ORDER
from agent.tools.base import RiskLevel

T = TypeVar("T")

LONG_TASK_CHARS = 8000
MANY_TOOLS = 10
DEFAULT_TIMEOUT = 2.0
_BASE_ASK = "low"
_BASE_DEV = "medium"

_RISK_INDEX = {level: index for index, level in enumerate(RiskLevel)}


@dataclass(frozen=True, slots=True)
class Decision[T]:
    """One immutable decision result."""

    value: T
    reason: str
    confidence: float
    engine: str


@dataclass(frozen=True, slots=True)
class TaskFeatures:
    """Facts that shape effort and model choices."""

    prompt_chars: int
    has_code: bool
    tool_count_so_far: int
    failed_attempts: int
    mode: str


@dataclass(frozen=True, slots=True)
class ApprovalItem:
    """One pending tool approval."""

    id: str
    tool: str
    risk: RiskLevel
    waiting_seconds: float


class DecisionEngine(Protocol):
    """Abstract decision layer. Implement it locally or inside a service."""

    async def choose_effort(self, task: TaskFeatures, allowed: Sequence[str]) -> Decision[str]: ...

    async def choose_model(
        self, task: TaskFeatures, candidates: Sequence[str]
    ) -> Decision[str]: ...

    async def rank_approvals(self, items: Sequence[ApprovalItem]) -> Decision[list[str]]: ...


class RuleEngine:
    """Decide from local rules without any external service.

    The effort rules are:
    - Start from the base level. It is 'low' for ask mode and 'medium' for dev mode.
    - Add one level for code tasks.
    - Add one level for long tasks.
    - Add one level for many tool steps.
    - Add one level for one failed step.
    - Use the maximum level for two or more failed steps.
    - Pick the nearest allowed level to the target. On a tie, pick the higher level.

    The model rule is:
    - Pick the first candidate in declaration order.

    The approval rule is:
    - Sort by risk, then by waiting time, then by id.
    - Return the ids in descending urgency order.

    An empty allowed or candidates list raises ValueError.
    The result is always one of the approved values.
    """

    engine = "rule"

    async def choose_effort(self, task: TaskFeatures, allowed: Sequence[str]) -> Decision[str]:
        if not allowed:
            raise ValueError("allowed efforts must not be empty")
        target = self._target_level(task)
        level = self._nearest(target, list(allowed))
        reason = self._effort_reason(task, target, level)
        return Decision(level, reason, 1.0, self.engine)

    async def choose_model(self, task: TaskFeatures, candidates: Sequence[str]) -> Decision[str]:
        if not candidates:
            raise ValueError("model candidates must not be empty")
        selected = candidates[0]
        return Decision(selected, "first matching candidate", 1.0, self.engine)

    async def rank_approvals(self, items: Sequence[ApprovalItem]) -> Decision[list[str]]:
        ordered = sorted(
            items,
            key=lambda item: (_RISK_INDEX[item.risk], item.waiting_seconds, item.id),
            reverse=True,
        )
        ids = [item.id for item in ordered]
        return Decision(ids, "risk, wait, then id", 1.0, self.engine)

    def _target_level(self, task: TaskFeatures) -> str:
        level = _BASE_ASK if tool_profile(task.mode) == "ask" else _BASE_DEV
        if task.has_code:
            level = self._next_level(level)
        if task.prompt_chars >= LONG_TASK_CHARS:
            level = self._next_level(level)
        if task.tool_count_so_far >= MANY_TOOLS:
            level = self._next_level(level)
        if task.failed_attempts >= 2:
            return EFFORT_ORDER[-1]
        if task.failed_attempts >= 1:
            level = self._next_level(level)
        return level

    def _next_level(self, level: str) -> str:
        index = EFFORT_ORDER.index(level)
        return EFFORT_ORDER[min(index + 1, len(EFFORT_ORDER) - 1)]

    def _nearest(self, wanted: str, allowed: list[str]) -> str:
        valid = [effort for effort in EFFORT_ORDER if effort in allowed]
        if not valid:
            raise ValueError("no valid effort level in allowed")
        target = EFFORT_ORDER.index(wanted)

        def distance(effort: str) -> tuple[int, int]:
            position = EFFORT_ORDER.index(effort)
            return abs(position - target), -position

        return min(valid, key=distance)

    def _effort_reason(self, task: TaskFeatures, target: str, level: str) -> str:
        parts = [f"mode {task.mode}"]
        if task.has_code:
            parts.append("code task")
        if task.prompt_chars >= LONG_TASK_CHARS:
            parts.append("long task")
        if task.tool_count_so_far >= MANY_TOOLS:
            parts.append("many tool steps")
        if task.failed_attempts:
            parts.append(f"{task.failed_attempts} failed steps")
        if level != target:
            parts.append(f"nearest allowed to {target}")
        return "; ".join(parts)


def _with_reason[T](decision: Decision[T], extra: str) -> Decision[T]:
    """Return a copy of the decision with the reason extended."""
    return Decision(
        value=decision.value,
        reason=f"{decision.reason}; {extra}",
        confidence=decision.confidence,
        engine=decision.engine,
    )


class FallbackEngine:
    """Wrap a primary engine and fall back when it fails.

    The primary engine is protected by a timeout. A fallback is used when:
    - The primary raises an exception.
    - The primary exceeds the timeout.
    - The primary returns a value outside the allowed set.
    - The primary returns an incomplete or duplicated approval ranking.

    The fallback decision is returned as-is. Its reason is extended with
    the cause of the rollback.
    """

    engine = "fallback"

    def __init__(
        self,
        primary: DecisionEngine,
        fallback: DecisionEngine,
        timeout_seconds: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._timeout = timeout_seconds

    async def choose_effort(self, task: TaskFeatures, allowed: Sequence[str]) -> Decision[str]:
        return await self._guarded(
            lambda: self._primary.choose_effort(task, allowed),
            lambda: self._fallback.choose_effort(task, allowed),
            lambda value: value in allowed,
        )

    async def choose_model(self, task: TaskFeatures, candidates: Sequence[str]) -> Decision[str]:
        return await self._guarded(
            lambda: self._primary.choose_model(task, candidates),
            lambda: self._fallback.choose_model(task, candidates),
            lambda value: value in candidates,
        )

    async def rank_approvals(self, items: Sequence[ApprovalItem]) -> Decision[list[str]]:
        valid_ids = {item.id for item in items}

        def is_complete(value: list[str]) -> bool:
            return set(value) == valid_ids

        return await self._guarded(
            lambda: self._primary.rank_approvals(items),
            lambda: self._fallback.rank_approvals(items),
            is_complete,
        )

    async def _guarded(
        self,
        primary_call: Callable[[], Awaitable[Decision[T]]],
        fallback_call: Callable[[], Awaitable[Decision[T]]],
        is_valid: Callable[[T], bool],
    ) -> Decision[T]:
        try:
            decision = await asyncio.wait_for(primary_call(), self._timeout)
        except Exception as exc:
            reason = f"primary failed: {type(exc).__name__}"
        else:
            if is_valid(decision.value):
                return decision
            reason = "primary returned an invalid value"
        fallen_back = await fallback_call()
        return _with_reason(fallen_back, reason)
