from __future__ import annotations

import math
from pathlib import Path

import pytest

from agent.rag.metrics import (
    EvalCase,
    evaluate,
    load_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_hits_in_first_k() -> None:
    assert recall_at_k(["a", "x", "b", "c"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "x", "b", "c"], {"a", "b", "c"}, 4) == 1.0


def test_recall_ignores_repeated_results() -> None:
    # The repeat of "a" must not push "b" out of the first two places.
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0


def test_duplicate_relevant_ids_count_once() -> None:
    assert recall_at_k(["a"], ["a", "a", "b"], 5) == 0.5


def test_k_larger_than_results() -> None:
    assert recall_at_k(["a"], {"a", "b"}, 100) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_ndcg_known_values() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a"], {"a", "b"}, 2) == pytest.approx(expected)
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(["a"], {"a"}, k)
    with pytest.raises(ValueError, match="k must be at least 1"):
        ndcg_at_k(["a"], {"a"}, k)
    with pytest.raises(ValueError, match="k must be at least 1"):
        evaluate([EvalCase("q", frozenset({"a"}))], lambda _q: ["a"], k)


def test_empty_relevant_is_an_error() -> None:
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError, match="relevant"):
        reciprocal_rank(["a"], [])
    with pytest.raises(ValueError, match="relevant"):
        evaluate([EvalCase("q", frozenset())], lambda _q: ["a"])


def test_evaluate_means() -> None:
    cases = [EvalCase("q1", frozenset({"a"})), EvalCase("q2", frozenset({"b"}))]
    results = {"q1": ["a"], "q2": ["x", "b"]}
    report = evaluate(cases, lambda query: results[query], k=1)
    assert report == {
        "recall_at_k": 0.5,
        "mrr": pytest.approx(0.75),
        "ndcg_at_k": 0.5,
        "queries": 2.0,
    }


def test_evaluate_no_cases() -> None:
    assert evaluate([], lambda _q: [])["queries"] == 0.0


def test_load_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"query": "где ключ", "relevant": ["d1", "d2", "d1"]}\n\n'
        '{"query": "b", "relevant": ["x"]}\n',
        encoding="utf-8",
    )
    assert load_cases(path) == [
        EvalCase("где ключ", frozenset({"d1", "d2"})),
        EvalCase("b", frozenset({"x"})),
    ]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{bad", "line 2: invalid JSON"),
        ("[1]", "line 2: expected a JSON object"),
        ('{"relevant": ["a"]}', "line 2: query"),
        ('{"query": " ", "relevant": ["a"]}', "line 2: query"),
        ('{"query": "q", "relevant": "abc"}', "line 2: relevant must be a list"),
        ('{"query": "q", "relevant": [1]}', "line 2: relevant must be a list"),
        ('{"query": "q", "relevant": []}', "line 2: relevant must not be empty"),
    ],
)
def test_load_cases_errors(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"query": "ok", "relevant": ["a"]}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_cases(path)
