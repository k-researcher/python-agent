from __future__ import annotations

import pytest

from src.agent.context_budget import (
    Budget,
    fits,
    input_budget,
    overflow_target,
    safety_margin,
)


def test_safety_margin():
    # Small window (4096) -> 2% is 81.92, max(256, 81) = 256
    assert safety_margin(4096) == 256
    # Large window (131072): 2 % is 2621.44, rounded up to 2622.
    assert safety_margin(131072) == 2622


def test_input_budget_max_tokens_higher():
    # context: 10000, reserved: 500, max: 1000
    # margin: max(256, 200) = 256
    # output: max(500, 1000) = 1000
    # input: (10000 - 1000 - 256) = 8744
    budget = input_budget(10000, 500, 1000)
    assert budget.output_tokens == 1000
    assert budget.input_tokens == 8744


def test_input_budget_reserved_higher():
    # context: 10000, reserved: 2000, max: 500
    # margin: 256
    # output: max(2000, 500) = 2000
    # input: (10000 - 2000 - 256) = 7744
    budget = input_budget(10000, 2000, 500)
    assert budget.output_tokens == 2000
    assert budget.input_tokens == 7744


def test_input_budget_has_no_lower_bound() -> None:
    # A small window must not get a hidden minimum: 1200 - 100 - 256 = 844.
    assert input_budget(1200, 100, 100).input_tokens == 844
    # The answer limit itself does not fit: the budget is zero or less.
    # 1000 - max(min(100, 250), 900) - 256 = -156.
    assert input_budget(1000, 100, 900).input_tokens <= 0


def test_reserve_is_capped_to_a_quarter_of_a_small_window() -> None:
    # A local model with an 8192 window and the global reserve of 12000.
    # reserve: min(12000, 8192 // 4) = 2048; output: max(2048, 4096) = 4096
    # input: 8192 - 4096 - 256 = 3840
    budget = input_budget(8192, 12000, 4096)
    assert budget.output_tokens == 4096
    assert budget.input_tokens == 3840


def test_capped_reserve_wins_over_a_smaller_answer_limit() -> None:
    # reserve: min(12000, 2048) = 2048; output: max(2048, 1000) = 2048
    # input: 8192 - 2048 - 256 = 5888
    budget = input_budget(8192, 12000, 1000)
    assert budget.output_tokens == 2048
    assert budget.input_tokens == 5888


def test_large_window_keeps_the_full_reserve() -> None:
    # A quarter of 256000 is 64000, so the reserve of 12000 is not cut.
    # margin: ceil(5120) = 5120
    assert input_budget(256000, 12000, 8192).output_tokens == 12000
    # The answer limit is larger than the reserve: output 16000.
    budget = input_budget(256000, 12000, 16000)
    assert budget.output_tokens == 16000
    assert budget.input_tokens == 256000 - 16000 - 5120


def test_overflow_target():
    # Case 1: input_tokens is smaller
    # budget: context 10000, input 2000
    # target: min(2000, 6000) = 2000
    b1 = Budget(10000, 1000, 256, 2000)
    assert overflow_target(b1) == 2000

    # Case 2: 60% of context is smaller
    # budget: context 10000, input 8000
    # target: min(8000, 6000) = 6000
    b2 = Budget(10000, 1000, 256, 8000)
    assert overflow_target(b2) == 6000


def test_fits():
    budget = Budget(10000, 1000, 256, 5000)
    assert fits(5000, budget) is True
    assert fits(4999, budget) is True
    assert fits(5001, budget) is False


def test_value_errors():
    with pytest.raises(ValueError, match="context_window must be positive"):
        input_budget(0, 100, 100)
    with pytest.raises(ValueError, match="context_window must be positive"):
        input_budget(-1, 100, 100)
    with pytest.raises(ValueError, match="Tokens must be non-negative"):
        input_budget(1000, -1, 100)
    with pytest.raises(ValueError, match="Tokens must be non-negative"):
        input_budget(1000, 100, -1)
