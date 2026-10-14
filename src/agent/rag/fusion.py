"""Reciprocal rank fusion for combining ranked result lists."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Combine ranked lists using reciprocal rank fusion.

    Each item scores the sum of 1 / (k + rank). Rank starts at one. A
    duplicate id inside one list uses its best rank. Equal scores break
    by the id. A negative k raises ValueError.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    merged: dict[str, float] = {}
    for ranking in rankings:
        used: set[str] = set()
        for position, doc_id in enumerate(ranking):
            if doc_id in used:
                continue
            used.add(doc_id)
            merged[doc_id] = merged.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    if limit is None:
        return ordered
    return ordered[:limit]
