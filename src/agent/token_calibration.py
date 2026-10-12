"""Per-model token usage calibration via exponentially weighted moving averages."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

MIN_FACTOR = 0.5
MAX_FACTOR = 4.0
ALPHA = 0.2


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    """Unique key for a model based on provider, model name, and wire version."""

    provider: str
    model: str
    wire_version: str

    def encode(self) -> str:
        """Return a JSON array string: ["provider", "model", "wire_version"]."""
        return json.dumps(
            [self.provider, self.model, self.wire_version],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def decode(cls, value: str) -> CalibrationKey:
        """Parse the JSON array string. Raise ValueError for any other shape."""
        try:
            data = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON format") from e

        if not isinstance(data, list) or len(data) != 3:
            raise ValueError("Expected a list of exactly 3 elements")

        if not all(isinstance(x, str) and x for x in data):
            raise ValueError("All elements must be non-empty strings")

        return cls(provider=data[0], model=data[1], wire_version=data[2])


class TokenCalibrator:
    """Track a correction factor applied to estimated token counts per model key."""

    def __init__(self) -> None:
        self._factors: dict[str, float] = {}
        self._samples: dict[str, int] = {}

    def factor(self, key: str) -> float:
        """Return the calibration factor for a key, defaulting to 1.0."""
        return self._factors.get(key, 1.0)

    def samples(self, key: str) -> int:
        """Return the number of recorded observations for a key."""
        return self._samples.get(key, 0)

    def observe(self, key: str, estimated: int, actual: int | None) -> float:
        """Feed one observation and return the updated factor.

        Invalid observations (unknown, non-positive actual or non-positive
        estimated) leave the state untouched. Otherwise apply an EWMA update
        and clamp the result to ``[MIN_FACTOR, MAX_FACTOR]``.
        """
        if actual is None or actual <= 0 or estimated <= 0:
            return self.factor(key)
        old = self.factor(key)
        new = (1.0 - ALPHA) * old + ALPHA * (actual / estimated)
        new = min(max(new, MIN_FACTOR), MAX_FACTOR)
        self._factors[key] = new
        self._samples[key] = self._samples.get(key, 0) + 1
        return new

    def adjust(self, key: str, estimated: int) -> int:
        """Scale an estimate by the key's factor, rounding up."""
        return math.ceil(estimated * self.factor(key))

    def snapshot(self) -> dict[str, tuple[float, int]]:
        """Return per-key (factor, sample count) pairs for persistence."""
        return {key: (self._factors[key], self._samples[key]) for key in self._factors}

    @classmethod
    def restore(cls, data: Mapping[str, tuple[float, int]]) -> TokenCalibrator:
        """Rebuild a calibrator from a snapshot, clamping factors to the allowed range."""
        calibrator = cls()
        for key, (factor, count) in data.items():
            calibrator._factors[key] = min(max(factor, MIN_FACTOR), MAX_FACTOR)
            calibrator._samples[key] = count
        return calibrator
