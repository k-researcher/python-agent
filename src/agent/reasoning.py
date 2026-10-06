"""Select the reasoning level for one model call and translate it to wire parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agent.model_config import ResolvedModel

AUTO = "auto"
# Ordered from the lowest to the highest level.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
THINKING_BUDGETS = {"minimal": 1024, "low": 1024, "medium": 4096, "high": 8192}
LONG_TASK_CHARS = 8000

Source = Literal["message", "session", "auto", "model", "none"]


@dataclass(frozen=True, slots=True)
class ReasoningDecision:
    requested: str | None
    effective: str | None
    source: Source
    reason: str


def validate_effort(model: ResolvedModel, effort: str | None) -> None:
    """Refuse a level that the model does not accept. "auto" is a policy, always accepted."""
    if effort is None or effort == AUTO:
        return
    if not model.reasoning.supported or effort not in model.reasoning.allowed_efforts:
        allowed = ", ".join([AUTO, *model.reasoning.allowed_efforts]) or "none"
        raise ValueError(
            f"Model {model.id} does not accept reasoning effort {effort!r}; use: {allowed}"
        )


def _nearest(model: ResolvedModel, wanted: str) -> str | None:
    """Return the allowed level closest to ``wanted``; on a tie, the higher level."""
    allowed = [effort for effort in EFFORT_ORDER if effort in model.reasoning.allowed_efforts]
    if not allowed:
        return None
    target = EFFORT_ORDER.index(wanted)

    def distance(effort: str) -> tuple[int, int]:
        position = EFFORT_ORDER.index(effort)
        return abs(position - target), -position

    return min(allowed, key=distance)


def _auto(model: ResolvedModel, mode: str, task_chars: int, recent_errors: int) -> tuple[str, str]:
    level = "low" if mode == "ask" else "medium"
    reason = f"mode {mode}"
    if task_chars >= LONG_TASK_CHARS:
        level, reason = "high", "long task"
    if recent_errors >= 2:
        level, reason = "max", "two failed steps in a row"
    elif recent_errors == 1:
        level = EFFORT_ORDER[min(EFFORT_ORDER.index(level) + 1, len(EFFORT_ORDER) - 1)]
        reason = "one failed step"
    return _nearest(model, level) or level, reason


def decide(
    model: ResolvedModel,
    *,
    message_effort: str | None,
    session_effort: str | None,
    mode: str,
    task_chars: int = 0,
    recent_errors: int = 0,
) -> ReasoningDecision:
    """Message override, then the session level, then the model default.

    "auto" selects a level from the mode, the task size and recent tool errors. The same
    interface will later take a decision from the decision layer (Jev).
    """
    if not model.reasoning.supported:
        return ReasoningDecision(None, None, "none", "model has no reasoning control")
    requested = message_effort or session_effort
    source: Source = "message" if message_effort else "session" if session_effort else "model"
    if requested == AUTO:
        effective, reason = _auto(model, mode, task_chars, recent_errors)
        return ReasoningDecision(AUTO, effective, "auto", reason)
    if requested and requested in model.reasoning.allowed_efforts:
        return ReasoningDecision(requested, requested, source, f"{source} setting")
    default = model.reasoning.default_effort
    return ReasoningDecision(requested, default, "model", "model default")


def wire_parameters(
    model: ResolvedModel, decision: ReasoningDecision, max_tokens: int
) -> dict[str, Any]:
    """Payload fields for the selected level, in the dialect that the model declares."""
    effort = decision.effective
    parameter = model.reasoning.wire_parameter
    if effort is None or parameter is None:
        return {}
    if parameter == "reasoning_effort":
        return {"reasoning_effort": effort}
    if parameter == "enable_thinking":
        return {"enable_thinking": effort != "none"}
    if effort == "none":
        return {"thinking": {"type": "disabled"}}
    budget = max(1024, min(THINKING_BUDGETS.get(effort, 16_384), max_tokens - 1))
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}
