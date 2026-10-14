"""Quality metrics for retrieval: recall@k, reciprocal rank and nDCG@k.

Relevance is binary. A document ID that the search returns more than once counts
only at its first position. The relevant IDs form a set.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One query and the IDs of the documents that are relevant to it."""

    query: str
    relevant: frozenset[str]


def _unique(retrieved: Iterable[str]) -> list[str]:
    """Return the IDs in their first-seen order, without repeats."""
    seen: set[str] = set()
    result: list[str] = []
    for item in retrieved:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _check(relevant: Collection[str], k: int | None = None) -> frozenset[str]:
    if k is not None and k < 1:
        raise ValueError("k must be at least 1")
    targets = frozenset(relevant)
    if not targets:
        raise ValueError("relevant must contain at least one ID")
    return targets


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Return the share of relevant IDs found in the first k unique results."""
    targets = _check(relevant, k)
    hits = sum(1 for item in _unique(retrieved)[:k] if item in targets)
    return hits / len(targets)


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """Return 1 / rank of the first relevant result, or 0.0 when there is none."""
    targets = _check(relevant)
    for rank, item in enumerate(_unique(retrieved), start=1):
        if item in targets:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Return the normalized discounted cumulative gain of the first k unique results."""
    targets = _check(relevant, k)
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(_unique(retrieved)[:k], start=1)
        if item in targets
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(targets), k) + 1))
    return gain / ideal


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read cases from a JSONL file: {"query": str, "relevant": [str, ...]} per line.

    Empty lines are skipped. A format error raises ValueError with the line number.
    """
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {number}: invalid JSON") from exc
            if not isinstance(data, dict):
                raise ValueError(f"line {number}: expected a JSON object")
            query = data.get("query")
            relevant = data.get("relevant")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"line {number}: query must be a non-empty string")
            if not isinstance(relevant, list) or not all(isinstance(i, str) for i in relevant):
                raise ValueError(f"line {number}: relevant must be a list of strings")
            if not relevant:
                raise ValueError(f"line {number}: relevant must not be empty")
            cases.append(EvalCase(query, frozenset(relevant)))
    return cases


def evaluate(
    cases: Sequence[EvalCase], search: Callable[[str], Sequence[str]], k: int = 10
) -> dict[str, float]:
    """Return mean recall@k, MRR and nDCG@k over the cases, and the number of queries."""
    if k < 1:
        raise ValueError("k must be at least 1")
    for case in cases:
        _check(case.relevant)
    if not cases:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "queries": 0.0}
    recall = rr = ndcg = 0.0
    for case in cases:
        retrieved = search(case.query)
        recall += recall_at_k(retrieved, case.relevant, k)
        rr += reciprocal_rank(retrieved, case.relevant)
        ndcg += ndcg_at_k(retrieved, case.relevant, k)
    count = len(cases)
    return {
        "recall_at_k": recall / count,
        "mrr": rr / count,
        "ndcg_at_k": ndcg / count,
        "queries": float(count),
    }
