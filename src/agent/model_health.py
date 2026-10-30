"""Remember the last check of each agent model, so a model that does not answer is not offered."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

Health = Literal["ok", "failed"]

# A failure can be a short network break. After this time the model is unchecked again
# and is offered, so one break does not hide the model until somebody checks it again.
FAILED_FOR = timedelta(minutes=5)


@dataclass(frozen=True)
class _Check:
    target: tuple[str, str]
    health: Health
    status: int | None
    checked_at: datetime


# The last check per model ID in this server process. A check is valid only for the same
# server address and server model; a change of either makes the model unchecked again.
_checks: dict[str, _Check] = {}


def _now() -> datetime:
    return datetime.now(UTC)


def record(model_id: str, base_url: str, model: str, ok: bool, status: int | None) -> None:
    _checks[model_id] = _Check(
        (base_url.rstrip("/"), model), "ok" if ok else "failed", status, _now()
    )


def health(model_id: str, base_url: str, model: str) -> Health | None:
    check = _checks.get(model_id)
    if check is None or check.target != (base_url.rstrip("/"), model):
        return None
    if check.health == "failed" and _now() - check.checked_at > FAILED_FOR:
        return None
    return check.health


def clear() -> None:
    _checks.clear()
