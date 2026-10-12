"""Exercise usage accounting: parsing, pricing and totals."""

from __future__ import annotations

from decimal import Decimal

from agent.model_config import PricingSpec
from agent.usage import UsageRecord, cost, total, usage_from_response


def test_full_usage_from_response() -> None:
    usage = usage_from_response(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 20},
            "prompt_tokens_details": {"cached_tokens": 30},
        }
    )
    assert usage == UsageRecord(
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=20,
        cached_tokens=30,
    )


def test_missing_details_default_to_none() -> None:
    usage = usage_from_response({"prompt_tokens": 10, "completion_tokens": 10})
    assert usage.reasoning_tokens is None
    assert usage.cached_tokens is None


def test_missing_counts_default_to_none() -> None:
    usage = usage_from_response({})
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None


def test_none_usage_returns_empty_record() -> None:
    record = usage_from_response(None)
    assert record == UsageRecord(
        prompt_tokens=None, completion_tokens=None, reasoning_tokens=None, cached_tokens=None
    )
    assert record.prompt_tokens is None
    assert record.completion_tokens is None


def test_zero_rates_cost_zero() -> None:
    record = UsageRecord(prompt_tokens=100, completion_tokens=50)
    result = cost(record, PricingSpec(input_per_mtok=0.0, output_per_mtok=0.0))
    assert result is not None
    assert isinstance(result, Decimal)
    assert result == Decimal(0)


def test_unknown_value_gives_none() -> None:
    pricing = PricingSpec(input_per_mtok=1.0, output_per_mtok=2.0)
    assert cost(UsageRecord(prompt_tokens=None, completion_tokens=10), pricing) is None
    assert cost(UsageRecord(prompt_tokens=10, completion_tokens=None), pricing) is None
    assert cost(UsageRecord(prompt_tokens=10, completion_tokens=10), None) is None


def test_reasoning_tokens_are_not_double_counted() -> None:
    pricing = PricingSpec(input_per_mtok=1.0, output_per_mtok=2.0)
    record = UsageRecord(prompt_tokens=100, completion_tokens=50, reasoning_tokens=20)
    assert cost(record, pricing) == Decimal("0.000200")


def test_cost_decimal_precision() -> None:
    pricing = PricingSpec(input_per_mtok=0.5, output_per_mtok=1.5)
    record = UsageRecord(prompt_tokens=1_000_000, completion_tokens=2_000_000)
    result = cost(record, pricing)
    assert result is not None
    assert isinstance(result, Decimal)
    assert result == Decimal("3.500000")


def test_cost_rounds_to_six_decimal_places_up() -> None:
    pricing = PricingSpec(input_per_mtok=0.6, output_per_mtok=0.0)
    record = UsageRecord(prompt_tokens=1, completion_tokens=0)
    assert cost(record, pricing) == Decimal("0.000001")


def test_cost_rounds_to_six_decimal_places_down() -> None:
    pricing = PricingSpec(input_per_mtok=0.4, output_per_mtok=0.0)
    record = UsageRecord(prompt_tokens=1, completion_tokens=0)
    assert cost(record, pricing) == Decimal("0.000000")


def test_total_sums_known_and_flags_unknown() -> None:
    result, has_unknown = total([Decimal("1.5"), None, Decimal("0.25"), Decimal("0.125")])
    assert result == Decimal("1.875")
    assert has_unknown is True


def test_total_without_unknown_has_false_flag() -> None:
    result, has_unknown = total([Decimal("1"), Decimal("2"), Decimal("0.5")])
    assert result == Decimal("3.5")
    assert has_unknown is False


def test_total_empty() -> None:
    result, has_unknown = total([])
    assert result == Decimal(0)
    assert has_unknown is False
