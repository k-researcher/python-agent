"""Unit tests for the reciprocal rank fusion module."""

import pytest

from agent.rag.fusion import reciprocal_rank_fusion


def test_fusion_known_example() -> None:
    first = ["a", "b", "c"]
    second = ["b", "c", "a"]
    results = dict(reciprocal_rank_fusion([first, second]))
    assert results["a"] == 1 / 61 + 1 / 63
    assert results["b"] == 1 / 62 + 1 / 61
    assert results["c"] == 1 / 63 + 1 / 62


def test_fusion_deterministic_tie_order() -> None:
    results = [doc_id for doc_id, _score in reciprocal_rank_fusion([["z"], ["x"]])]
    assert results == ["x", "z"]


def test_fusion_duplicate_within_list_uses_best_rank() -> None:
    ranking = ["a", "a", "b"]
    results = dict(reciprocal_rank_fusion([ranking]))
    assert results["a"] == 1 / 61
    assert results["b"] == 1 / 63


def test_fusion_limit_is_respected() -> None:
    results = reciprocal_rank_fusion([["a", "b", "c"]], limit=2)
    assert [doc_id for doc_id, _score in results] == ["a", "b"]


def test_fusion_negative_k_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_fusion_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_fusion_empty_rankings_inside_returns_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
