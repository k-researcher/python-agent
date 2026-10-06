from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class Budget:
    """Token budget of one LLM request."""

    context_window: int
    output_tokens: int  # max(reserved_tokens, max_tokens)
    safety_margin: int
    input_tokens: int  # tokens left for the request input


def safety_margin(context_window: int) -> int:
    """Return max(256, 2 % of context_window), rounded up."""
    return max(256, ceil(context_window * 2 / 100))


def input_budget(context_window: int, reserved_tokens: int, max_tokens: int) -> Budget:
    """Return the token budget that is left for the request input."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if reserved_tokens < 0 or max_tokens < 0:
        raise ValueError("Tokens must be non-negative")

    margin = safety_margin(context_window)
    output = max(reserved_tokens, max_tokens)

    # No lower bound: a budget of zero or less means that the request cannot fit.
    input_tokens = context_window - output - margin

    return Budget(
        context_window=context_window,
        output_tokens=output,
        safety_margin=margin,
        input_tokens=input_tokens,
    )


def overflow_target(budget: Budget) -> int:
    """Return min(budget.input_tokens, int(0.60 * budget.context_window))."""
    return min(budget.input_tokens, budget.context_window * 60 // 100)


def fits(estimated_tokens: int, budget: Budget) -> bool:
    """Return True when the estimated input fits in the budget."""
    return estimated_tokens <= budget.input_tokens
