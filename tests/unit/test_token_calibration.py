"""Tests for the token calibration EWMA factor."""

from __future__ import annotations

import pytest

from agent.token_calibration import (
    MAX_FACTOR,
    MIN_FACTOR,
    CalibrationKey,
    TokenCalibrator,
)


def test_initial_factor_and_samples() -> None:
    calibrator = TokenCalibrator()
    assert calibrator.factor("gpt-4o") == 1.0
    assert calibrator.samples("gpt-4o") == 0


def test_observe_applies_ewma_formula() -> None:
    calibrator = TokenCalibrator()
    new = calibrator.observe("m", 100, 200)
    assert new == pytest.approx(1.2)  # 0.8 * 1.0 + 0.2 * 2.0
    assert calibrator.samples("m") == 1


def test_converges_to_actual_ratio() -> None:
    calibrator = TokenCalibrator()
    for _ in range(200):
        calibrator.observe("slow", 100, 150)
    assert calibrator.factor("slow") == pytest.approx(1.5, abs=1e-6)


def test_factor_is_clamped_to_minimum() -> None:
    calibrator = TokenCalibrator()
    for _ in range(50):
        calibrator.observe("over", 100, 1)
    assert calibrator.factor("over") == pytest.approx(MIN_FACTOR)


def test_factor_is_clamped_to_maximum() -> None:
    calibrator = TokenCalibrator()
    for _ in range(50):
        calibrator.observe("under", 1, 1000)
    assert calibrator.factor("under") == pytest.approx(MAX_FACTOR)


def test_invalid_observations_are_ignored() -> None:
    calibrator = TokenCalibrator()
    assert calibrator.observe("m", 100, None) == 1.0
    assert calibrator.observe("m", 100, 0) == 1.0
    assert calibrator.observe("m", 100, -5) == 1.0
    assert calibrator.observe("m", 0, 50) == 1.0
    assert calibrator.observe("m", -3, 50) == 1.0
    assert calibrator.samples("m") == 0
    assert calibrator.factor("m") == 1.0


def test_keys_are_independent() -> None:
    calibrator = TokenCalibrator()
    calibrator.observe("a", 100, 50)
    calibrator.observe("b", 50, 500)
    assert calibrator.factor("a") != calibrator.factor("b")
    assert calibrator.samples("a") == 1
    assert calibrator.samples("b") == 1


def test_adjust_rounds_up() -> None:
    calibrator = TokenCalibrator.restore({"x": (1.5, 5)})
    assert calibrator.adjust("x", 100) == 150
    assert calibrator.adjust("x", 105) == 158  # ceil(157.5)


def test_snapshot_and_restore_roundtrip() -> None:
    calibrator = TokenCalibrator()
    calibrator.observe("a", 100, 120)
    calibrator.observe("a", 100, 140)
    calibrator.observe("b", 100, 10)
    data = calibrator.snapshot()
    restored = TokenCalibrator.restore(data)
    assert restored.snapshot() == data
    assert restored.factor("a") == calibrator.factor("a")
    assert restored.samples("a") == 2
    assert restored.samples("b") == 1


def test_restore_clamps_factors() -> None:
    restored = TokenCalibrator.restore({"low": (0.1, 2), "high": (9.0, 2)})
    assert restored.factor("low") == MIN_FACTOR
    assert restored.factor("high") == MAX_FACTOR


def test_calibration_key_encode_decode() -> None:
    key = CalibrationKey("openai", "gpt-4/turbo", "v1")
    encoded = key.encode()
    # Use exact string representation check to ensure separators and no spaces
    assert encoded == '["openai","gpt-4/turbo","v1"]'
    decoded = CalibrationKey.decode(encoded)
    assert decoded == key


def test_calibration_key_special_characters() -> None:
    # Test model names with ":" and "/"
    key = CalibrationKey("other-provider", "family-3:large/2024", "1.0")
    encoded = key.encode()
    assert "family-3:large/2024" in encoded
    assert CalibrationKey.decode(encoded) == key


def test_calibration_key_uniqueness() -> None:
    key1 = CalibrationKey("p1", "m", "v")
    key2 = CalibrationKey("p2", "m", "v")
    key3 = CalibrationKey("p1", "m2", "v")
    key4 = CalibrationKey("p1", "m", "v2")

    assert key1.encode() != key2.encode()
    assert key1.encode() != key3.encode()
    assert key1.encode() != key4.encode()


def test_calibration_key_decode_errors() -> None:
    # Not a JSON list
    with pytest.raises(ValueError, match="Invalid JSON format"):
        CalibrationKey.decode("not-json")
    with pytest.raises(ValueError, match="Invalid JSON format"):
        CalibrationKey.decode('{"')

    # Not a list
    with pytest.raises(ValueError, match="Expected a list of exactly 3 elements"):
        CalibrationKey.decode('["1", "2"]')

    # Wrong number of elements
    with pytest.raises(ValueError, match="Expected a list of exactly 3 elements"):
        CalibrationKey.decode('["1", "2", "3", "4"]')

    # Contains non-string
    with pytest.raises(ValueError, match="All elements must be non-empty strings"):
        CalibrationKey.decode('["a", 1, "c"]')

    # Contains empty string
    with pytest.raises(ValueError, match="All elements must be non-empty strings"):
        CalibrationKey.decode('["", "b", "c"]')

    # List is empty
    with pytest.raises(ValueError, match="Expected a list of exactly 3 elements"):
        CalibrationKey.decode("[]")


def test_calibration_key_integration_with_calibrator() -> None:
    calibrator = TokenCalibrator()
    key = CalibrationKey("openai", "gpt-4", "v1")
    encoded_key = key.encode()

    # Test observe
    calibrator.observe(encoded_key, 100, 150)
    assert calibrator.factor(encoded_key) == pytest.approx(1.1)  # 0.8*1 + 0.2*1.5

    # Test factor
    assert calibrator.factor(encoded_key) == pytest.approx(1.1)

    # Test adjust
    assert calibrator.adjust(encoded_key, 100) == 111
