"""Vector stores for retrieval-augmented generation.

The module provides a common ``VectorStore`` protocol plus a SQLite-backed
implementation and an in-memory implementation used for tests and small
datasets. Later a pgvector or sqlite-vec backend may replace the SQLite
implementation behind the same interface.
"""

from __future__ import annotations

import math
import sqlite3
from array import array
from collections.abc import Sequence
from typing import NamedTuple, Protocol


class VectorItem(NamedTuple):
    """A single record for the vector store."""

    id: str
    source: str
    vector: Sequence[float]


class VectorStore(Protocol):
    """Common interface shared by all vector store backends."""

    def upsert(self, items: Sequence[VectorItem]) -> None:
        """Insert records or replace records that have the same identifier."""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """Delete records that match the given identifiers."""
        ...

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        *,
        filter_source: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return identifiers and cosine scores in descending order."""
        ...

    def count(self) -> int:
        """Return the number of stored records."""
        ...


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(sum(v * v for v in vector))


def _validate_upsert_vector(vector: Sequence[float], dim: int) -> None:
    """Raise when a stored vector has an unexpected dimension or bad values."""
    if len(vector) != dim:
        raise ValueError(f"expected dimension {dim}, got {len(vector)}")
    for value in vector:
        if math.isnan(value) or math.isinf(value):
            raise ValueError("vector contains NaN or infinite values")
    if _norm(vector) == 0.0:
        raise ValueError("vector must not be zero")


def _validate_query_vector(vector: Sequence[float], dim: int | None) -> None:
    """Raise when a query vector has an unexpected dimension or bad values."""
    if dim is not None and len(vector) != dim:
        raise ValueError(f"expected dimension {dim}, got {len(vector)}")
    for value in vector:
        if math.isnan(value) or math.isinf(value):
            raise ValueError("vector contains NaN or infinite values")
    if _norm(vector) == 0.0:
        raise ValueError("vector must not be zero")


class SqliteVectorStore:
    """A vector store persisted in a local SQLite database."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                norm REAL NOT NULL
            )
            """
        )
        self._connection.commit()

    def _current_dim(self) -> int | None:
        row = self._connection.execute("SELECT dim FROM vectors LIMIT 1").fetchone()
        return row[0] if row is not None else None

    def upsert(self, items: Sequence[VectorItem]) -> None:
        """Insert records or replace records that have the same identifier."""
        if not items:
            return
        dim = self._current_dim()
        if dim is None:
            dim = len(items[0].vector)
        for item in items:
            _validate_upsert_vector(item.vector, dim)
        with self._connection:
            for item in items:
                blob = array("f", [float(v) for v in item.vector]).tobytes()
                norm = _norm(item.vector)
                self._connection.execute(
                    """
                    INSERT INTO vectors (id, source, dim, vector, norm)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source = excluded.source,
                        dim = excluded.dim,
                        vector = excluded.vector,
                        norm = excluded.norm
                    """,
                    (item.id, item.source, dim, blob, norm),
                )

    def delete(self, ids: Sequence[str]) -> None:
        """Delete records that match the given identifiers."""
        if not ids:
            return
        with self._connection:
            self._connection.executemany(
                "DELETE FROM vectors WHERE id = ?",
                [(record_id,) for record_id in ids],
            )

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        *,
        filter_source: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return identifiers and cosine scores in descending order."""
        if limit < 0:
            raise ValueError("limit must not be negative")
        dim = self._current_dim()
        _validate_query_vector(vector, dim)
        if dim is None:
            return []
        norm_query = _norm(vector)
        if filter_source is None:
            rows = self._connection.execute("SELECT id, vector, norm FROM vectors").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT id, vector, norm FROM vectors WHERE source = ?",
                (filter_source,),
            ).fetchall()
        results: list[tuple[str, float]] = []
        for record_id, blob, stored_norm in rows:
            stored = array("f")
            stored.frombytes(blob)
            dot = sum(a * b for a, b in zip(stored, vector, strict=True))
            cosine = dot / (norm_query * stored_norm)
            results.append((record_id, cosine))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:limit]

    def count(self) -> int:
        """Return the number of stored records."""
        row = self._connection.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(row[0])


class MemoryVectorStore:
    """An in-memory vector store for tests and small datasets."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[str, tuple[float, ...], float]] = {}
        self._dim: int | None = None

    def upsert(self, items: Sequence[VectorItem]) -> None:
        """Insert records or replace records that have the same identifier."""
        if not items:
            return
        dim = self._dim if self._dim is not None else len(items[0].vector)
        for item in items:
            _validate_upsert_vector(item.vector, dim)
        self._dim = dim
        for item in items:
            vec = tuple(float(v) for v in item.vector)
            self._items[item.id] = (item.source, vec, _norm(vec))

    def delete(self, ids: Sequence[str]) -> None:
        """Delete records that match the given identifiers."""
        for record_id in ids:
            self._items.pop(record_id, None)
        if not self._items:
            # An empty store has no dimension, as in SqliteVectorStore.
            self._dim = None

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        *,
        filter_source: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return identifiers and cosine scores in descending order."""
        if limit < 0:
            raise ValueError("limit must not be negative")
        _validate_query_vector(vector, self._dim)
        if self._dim is None:
            return []
        norm_query = _norm(vector)
        results: list[tuple[str, float]] = []
        for record_id, (source, vec, norm) in self._items.items():
            if filter_source is not None and source != filter_source:
                continue
            dot = sum(a * b for a, b in zip(vec, vector, strict=True))
            results.append((record_id, dot / (norm_query * norm)))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:limit]

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._items)
