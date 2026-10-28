"""Bounded exact periodic suffix detector for one text channel."""

from __future__ import annotations

import asyncio


class RepetitionDetector:
    def __init__(self, threshold: int | None) -> None:
        self.threshold = threshold
        self.ring: list[str] = []
        self.matches = [0] * 481
        self._limits = (
            [(max(128, (threshold + p - 1) // p) - 1) * p for p in range(32, 513)]
            if threshold is not None
            else []
        )
        self._position = 0
        self._hit = False

    async def feed(self, text: str) -> bool:
        if self.threshold is None or self._hit:
            return self._hit
        ring, matches, limits = self.ring, self.matches, self._limits
        position = self._position
        for offset, char in enumerate(text):
            size = len(ring)
            for i in range(min(481, max(0, size - 31))):
                if char == ring[(position - i - 32) % 512]:
                    count = matches[i] + 1
                    matches[i] = min(count, limits[i])
                    if count >= limits[i]:
                        self._hit = True
                else:
                    matches[i] = 0
            if size < 512:
                ring.append(char)
            else:
                ring[position % 512] = char
            position = (position + 1) % 512
            if offset % 256 == 255:
                await asyncio.sleep(0)
            if self._hit:
                break
        self._position = position
        await asyncio.sleep(0)
        return self._hit
