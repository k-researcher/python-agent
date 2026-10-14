"""Tests for the vector store used in retrieval-augmented generation."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest

from agent.rag.vector_store import (
    MemoryVectorStore,
    SqliteVectorStore,
    VectorItem,
    VectorStore,
)

StoreFactory = Callable[[], VectorStore]


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(tmp_path: Path, request: pytest.FixtureRequest) -> StoreFactory:
    """Return a factory that creates a fresh vector store of the requested kind."""

    def make() -> VectorStore:
        if request.param == "memory":
            return MemoryVectorStore()
        return SqliteVectorStore(str(tmp_path / "store.db"))

    return make


def _upsert(store: VectorStore, *items: VectorItem) -> None:
    store.upsert(list(items))


def test_exact_cosine_values(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(
        store,
        VectorItem("a1", "s", [1.0, 0.0]),
        VectorItem("a2", "s", [0.0, 1.0]),
        VectorItem("a3", "s", [-1.0, 0.0]),
        VectorItem("a5", "s", [1.0, 1.0]),
    )
    result = dict(store.search([1.0, 0.0], 10))
    assert result["a1"] == pytest.approx(1.0)
    assert result["a2"] == pytest.approx(0.0)
    assert result["a3"] == pytest.approx(-1.0)
    assert result["a5"] == pytest.approx(1.0 / 2**0.5)


def test_empty_store_returns_nothing(store_factory: StoreFactory) -> None:
    store = store_factory()
    assert store.count() == 0
    assert store.search([1.0, 0.0], 10) == []


def test_search_ordering_and_ties(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(
        store,
        VectorItem("b", "s", [1.0, 0.0]),
        VectorItem("a", "s", [1.0, 0.0]),
        VectorItem("z", "s", [0.0, 1.0]),
        VectorItem("m", "s", [1.0, 1.0]),
    )
    result = store.search([1.0, 0.0], 10)
    ids = [record_id for record_id, _ in result]
    assert ids == ["a", "b", "m", "z"]


def test_replaces_by_identifier(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(store, VectorItem("x", "s1", [1.0, 0.0]))
    _upsert(store, VectorItem("x", "s2", [0.0, 1.0]))
    assert store.count() == 1
    result = store.search([0.0, 1.0], 10)
    assert result == [("x", pytest.approx(1.0))]


def test_delete(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(
        store,
        VectorItem("x", "s", [1.0, 0.0]),
        VectorItem("y", "s", [0.0, 1.0]),
    )
    store.delete(["x"])
    assert store.count() == 1
    assert [record_id for record_id, _ in store.search([1.0, 0.0], 10)] == ["y"]


def test_filter_by_source(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(
        store,
        VectorItem("x", "s1", [1.0, 0.0]),
        VectorItem("y", "s2", [1.0, 0.0]),
    )
    result = store.search([1.0, 0.0], 10, filter_source="s1")
    assert [record_id for record_id, _ in result] == ["x"]
    assert store.search([1.0, 0.0], 10, filter_source="absent") == []


def test_dimension_error(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(store, VectorItem("x", "s", [1.0, 0.0]))
    with pytest.raises(ValueError):
        _upsert(store, VectorItem("y", "s", [1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], 10)


def test_zero_vector_error(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(store, VectorItem("x", "s", [1.0, 0.0]))
    with pytest.raises(ValueError):
        _upsert(store, VectorItem("y", "s", [0.0, 0.0]))
    with pytest.raises(ValueError):
        store.search([0.0, 0.0], 10)


def test_nan_and_inf_error(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(store, VectorItem("x", "s", [1.0, 0.0]))
    with pytest.raises(ValueError):
        _upsert(store, VectorItem("y", "s", [float("nan"), 1.0]))
    with pytest.raises(ValueError):
        _upsert(store, VectorItem("y", "s", [float("inf"), 1.0]))
    with pytest.raises(ValueError):
        store.search([float("nan"), 1.0], 10)


def test_negative_limit_error(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(store, VectorItem("x", "s", [1.0, 0.0]))
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], -1)


def test_persistence_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.db"
    first = SqliteVectorStore(str(path))
    _upsert(
        first,
        VectorItem("x", "s1", [1.0, 0.0]),
        VectorItem("y", "s2", [0.0, 1.0]),
    )
    second = SqliteVectorStore(str(path))
    assert second.count() == 2
    result = dict(second.search([1.0, 0.0], 10))
    assert result["x"] == pytest.approx(1.0)
    assert result["y"] == pytest.approx(0.0)


def test_random_memory_and_sqlite_agree(tmp_path: Path) -> None:
    rng = random.Random(42)
    items = [
        VectorItem(
            f"id{i}",
            f"src{i % 3}",
            [rng.uniform(-1.0, 1.0) for _ in range(8)],
        )
        for i in range(50)
    ]
    memory = MemoryVectorStore()
    sqlite = SqliteVectorStore(str(tmp_path / "random.db"))
    memory.upsert(items)
    sqlite.upsert(items)
    for _ in range(20):
        query = [rng.uniform(-1.0, 1.0) for _ in range(8)]
        memory_result = memory.search(query, 10)
        sqlite_result = sqlite.search(query, 10)
        assert [record_id for record_id, _ in memory_result] == [
            record_id for record_id, _ in sqlite_result
        ]
        for (_, memory_score), (_, sqlite_score) in zip(memory_result, sqlite_result, strict=True):
            assert memory_score == pytest.approx(sqlite_score, abs=1e-5)


def test_random_replacement_and_delete(tmp_path: Path) -> None:
    rng = random.Random(7)
    dimension = 6
    memory = MemoryVectorStore()
    sqlite = SqliteVectorStore(str(tmp_path / "mutate.db"))
    base = [
        VectorItem(
            f"id{i}",
            f"src{i % 2}",
            [rng.uniform(-1.0, 1.0) for _ in range(dimension)],
        )
        for i in range(30)
    ]
    memory.upsert(base)
    sqlite.upsert(base)
    replacements = [
        VectorItem(f"id{i}", f"src-{i}", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for i in range(0, 10)
    ]
    memory.upsert(replacements)
    sqlite.upsert(replacements)
    memory.delete([f"id{i}" for i in range(20, 30)])
    sqlite.delete([f"id{i}" for i in range(20, 30)])
    assert memory.count() == sqlite.count()
    query = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    memory_result = memory.search(query, 10)
    sqlite_result = sqlite.search(query, 10)
    assert [record_id for record_id, _ in memory_result] == [
        record_id for record_id, _ in sqlite_result
    ]


def test_empty_store_forgets_dimension(store_factory: StoreFactory) -> None:
    # After the last record is deleted, both stores accept a new dimension.
    store = store_factory()
    _upsert(store, VectorItem("x", "s", [1.0, 0.0]))
    store.delete(["x"])
    assert store.search([1.0, 0.0, 0.0], 10) == []
    _upsert(store, VectorItem("y", "s", [0.0, 1.0, 0.0]))
    assert store.search([0.0, 1.0, 0.0], 10) == [("y", pytest.approx(1.0))]


def test_wrong_dimension_is_reported_before_value_checks(store_factory: StoreFactory) -> None:
    store = store_factory()
    _upsert(store, VectorItem("x", "s", [1.0, 0.0]))
    with pytest.raises(ValueError, match="expected dimension 2"):
        store.search([0.0, 0.0, 0.0], 10)
