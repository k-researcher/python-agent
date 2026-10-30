from datetime import UTC, datetime, timedelta

import pytest

from agent import model_health


def setup_function() -> None:
    model_health.clear()


def test_unchecked_model_has_no_health() -> None:
    assert model_health.health("m", "http://h/v1", "x") is None


def test_last_check_wins() -> None:
    model_health.record("m", "http://h/v1/", "x", ok=False, status=500)
    assert model_health.health("m", "http://h/v1", "x") == "failed"
    model_health.record("m", "http://h/v1", "x", ok=True, status=200)
    assert model_health.health("m", "http://h/v1", "x") == "ok"


def test_a_new_address_or_server_model_needs_a_new_check() -> None:
    model_health.record("m", "http://h/v1", "x", ok=False, status=500)
    assert model_health.health("m", "http://other/v1", "x") is None
    assert model_health.health("m", "http://h/v1", "y") is None


def test_a_failure_expires_but_a_success_stays(monkeypatch: pytest.MonkeyPatch) -> None:
    model_health.record("bad", "http://h/v1", "x", ok=False, status=None)
    model_health.record("good", "http://h/v1", "y", ok=True, status=200)
    later = datetime.now(UTC) + model_health.FAILED_FOR + timedelta(seconds=1)
    monkeypatch.setattr(model_health, "_now", lambda: later)
    assert model_health.health("bad", "http://h/v1", "x") is None
    assert model_health.health("good", "http://h/v1", "y") == "ok"
