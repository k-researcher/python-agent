"""Unit tests for the BM25 lexical search module."""

import math

from agent.rag.lexical import Bm25Index, tokenize


def test_tokenize_lowercases_and_splits_words() -> None:
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_yo_becomes_e() -> None:
    assert tokenize("ёлка Ёж") == ["елка", "еж"]


def test_tokenize_keeps_russian_and_english() -> None:
    assert tokenize("Привет мир test") == ["привет", "мир", "test"]


def test_tokenize_separates_digits() -> None:
    assert tokenize("abc123") == ["abc", "123"]


def test_tokenize_drops_short_tokens() -> None:
    assert tokenize("a be see") == ["be", "see"]


def test_tokenize_drops_empty() -> None:
    assert tokenize("") == []


def _scores_for(query: str, limit: int | None = None) -> dict[str, float]:
    index = Bm25Index()
    index.add("d1", "the fox jumps over the lazy dog")
    index.add("d2", "a quick brown fox runs fast")
    index.add("d3", "the cat sleeps all day")
    params = {"query": query}
    if limit is not None:
        params["limit"] = limit
    return dict(index.search(**params))


def test_bm25_scores_match_hand_computation() -> None:
    index = Bm25Index(k1=1.5, b=0.75)
    index.add("d1", "the fox")
    index.add("d2", "the fox fox")
    num_documents = 2
    avg_length = (2 + 3) / num_documents
    idf = math.log(1 + (num_documents - 2 + 0.5) / (2 + 0.5))
    denom_d1 = 1.5 * (1 - 0.75 + 0.75 * (2 / avg_length)) + 1
    denom_d2 = 1.5 * (1 - 0.75 + 0.75 * (3 / avg_length)) + 2
    expected_d1 = idf * (1 * 2.5) / denom_d1
    expected_d2 = idf * (2 * 2.5) / denom_d2
    results = index.search("fox")
    assert len(results) == 2
    by_id = dict(results)
    assert math.isclose(by_id["d1"], expected_d1, rel_tol=1e-9)
    assert math.isclose(by_id["d2"], expected_d2, rel_tol=1e-9)
    assert by_id["d2"] > by_id["d1"]


def test_bm25_rank_documents_by_relevance() -> None:
    scores = _scores_for("fox")
    assert scores["d2"] > scores["d1"]
    assert "d3" not in scores


def test_bm25_replacement_replaces_document() -> None:
    index = Bm25Index()
    index.add("d1", "the fox")
    index.add("d2", "the cat")
    index.add("d1", "the dog")
    results = dict(index.search("fox"))
    assert results == {}
    results = dict(index.search("dog"))
    assert results["d1"] > 0


def test_bm25_remove_drops_document() -> None:
    index = Bm25Index()
    index.add("d1", "the fox")
    index.add("d2", "the cat")
    index.remove("d1")
    results = dict(index.search("fox"))
    assert results == {}


def test_bm25_empty_query_returns_empty() -> None:
    index = Bm25Index()
    index.add("d1", "the fox")
    assert index.search("") == []


def test_bm25_empty_index_returns_empty() -> None:
    index = Bm25Index()
    assert index.search("fox") == []


def test_bm25_deterministic_tie_order() -> None:
    index = Bm25Index()
    index.add("b", "fox fox")
    index.add("a", "fox fox")
    results = [doc_id for doc_id, _score in index.search("fox")]
    assert results == ["a", "b"]


def test_bm25_limit_is_respected() -> None:
    scores = _scores_for("fox", limit=1)
    assert len(scores) == 1


def test_bm25_russian_document() -> None:
    index = Bm25Index()
    index.add("d1", "быстрая лиса прыгает")
    index.add("d2", "медленная черепаха")
    results = dict(index.search("лиса"))
    assert results["d1"] > 0
    assert "d2" not in results


def test_removing_an_empty_document_leaves_no_length_entry() -> None:
    index = Bm25Index()
    index.add("empty", "")
    index.remove("empty")
    assert "empty" not in index._doc_lengths
    assert index._total_length == 0
