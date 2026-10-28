"""Exact periodic suffix detection with an independent whole-text oracle."""

import asyncio
import math
import random
import time

import pytest

from tests.inference_guard_helpers import guard_tasks as guard_tasks

pytestmark = pytest.mark.usefixtures("guard_tasks")


def oracle(text, threshold):
    return any(
        len(text) >= max(128, math.ceil(threshold / p)) * p
        and text[-max(128, math.ceil(threshold / p)) * p :]
        == text[-p:] * max(128, math.ceil(threshold / p))
        for p in range(32, 513)
    )


async def detect(text, threshold=65536, widths=(137,)):
    from agent.repetition import RepetitionDetector

    detector = RepetitionDetector(threshold)
    hit = False
    offset = 0
    for width in widths * (len(text) // sum(widths) + 1):
        hit = await detector.feed(text[offset : offset + width]) or hit
        offset += width
        if offset >= len(text):
            break
    return hit, detector


@pytest.mark.parametrize("period", [32, 67, 512])
async def test_exact_boundary_and_chunk_independence(period):
    async with asyncio.timeout(10):
        block = "".join(chr(1024 + i) for i in range(period))
        k = max(128, math.ceil(65536 / period))
        for text in [block * (k - 1), (block * k)[:-1], block * k]:
            hit, _ = await detect(text, widths=(1, 255, 4096))
            assert hit == oracle(text, 65536)


@pytest.mark.parametrize(
    "text,hit",
    [
        (
            "| name | count | status |\n| --- | --- | --- |\n" + "| item | 0 | unchanged |\n" * 40,
            False,
        ),
        ("```log\n" + "[info] task completed without changes\n" * 1000 + "```", False),
        ("```json\n[" + ",".join(["0"] * 10000) + "]\n```", False),
        ("```python\n" + "values.append(0)\n" * 40 + "```", False),
        ("[info] task completed without changes\n" * 2000, True),
        ("[" + ",".join(["0"] * 40000) + "]", True),
    ],
)
async def test_complete_fixtures(text, hit):
    async with asyncio.timeout(10):
        assert (await detect(text))[0] is hit


async def test_off_override_memory_and_saturation():
    async with asyncio.timeout(10):
        text = "x" * 131072
        assert not (await detect(text, None))[0]
        assert (await detect(text, 4096))[0]
        _, detector = await detect(text)
        assert len(detector.ring) <= 512
        assert len(detector.matches) == 481
        assert all(
            count <= (max(128, math.ceil(65536 / p)) - 1) * p
            for p, count in zip(range(32, 513), detector.matches, strict=True)
        )


async def test_long_nonperiodic_throughput_and_loop_latency():
    async with asyncio.timeout(10):
        rng = random.Random(431)
        text = "".join(chr(rng.randrange(0x400, 0xD700)) for _ in range(262144))
        from agent.repetition import RepetitionDetector

        detector = RepetitionDetector(65536)
        times = []
        running = True

        async def heartbeat():
            while running:
                times.append(time.perf_counter())
                await asyncio.sleep(0)

        task = asyncio.create_task(heartbeat())
        start = time.perf_counter()
        try:
            assert not await detector.feed(text)
        finally:
            running = False
            await task
        end = time.perf_counter()
        assert end - start <= 8
        assert max(b - a for a, b in zip([start, *times], [*times, end], strict=True)) <= 0.1


async def test_period_above_window_and_variable_repetition_can_pass():
    async with asyncio.timeout(10):
        block = "".join(chr(1024 + i) for i in range(513))
        assert not (await detect(block * 140))[0]
        text = "".join(f"{i}: unchanged\n" for i in range(10000))
        assert not (await detect(text))[0]


async def test_seeded_arbitrary_chunking_matches_whole_suffix_oracle():
    async with asyncio.timeout(10):
        rng = random.Random(912)
        for p in (32, 39, 121, 512):
            block = "".join(chr(rng.randrange(1024, 20000)) for _ in range(p))
            k = max(128, math.ceil(65536 / p))
            widths = tuple(rng.randrange(1, 901) for _ in range(40))
            for text in ["prefix\n" + block * (k - 1), "prefix\n" + block * k]:
                assert (await detect(text, widths=widths))[0] == oracle(text, 65536)


async def test_ring_cursor_and_counters_remain_bounded_without_a_hit():
    async with asyncio.timeout(5):
        from agent.repetition import RepetitionDetector

        detector = RepetitionDetector(65536)
        text = "".join(chr(1024 + i) for i in range(1024))
        assert not await detector.feed(text)
        assert 0 <= detector._position < 512


async def test_large_integer_threshold_is_exact():
    async with asyncio.timeout(5):
        from agent.repetition import RepetitionDetector

        threshold = 10**400
        detector = RepetitionDetector(threshold)
        assert detector._limits[0] == ((threshold + 31) // 32 - 1) * 32
        assert not await detector.feed("x" * 512)
