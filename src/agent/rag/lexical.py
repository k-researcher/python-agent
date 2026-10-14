"""BM25 lexical search over an in-memory document collection."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
_DIGIT_PATTERN = re.compile(r"\d+")
_MIN_TOKEN_LENGTH = 2
_YO_TO_E = str.maketrans({"ё": "е", "Ё": "Е"})


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens for lexical matching.

    Words use Unicode letters for English and Russian. Each digit run is
    a separate token. ё becomes е. Tokens shorter than two characters
    are dropped.
    """
    lowered = text.lower().translate(_YO_TO_E)
    tokens: list[str] = []
    for match in _WORD_PATTERN.finditer(lowered):
        if len(match.group()) >= _MIN_TOKEN_LENGTH:
            tokens.append(match.group())
    for match in _DIGIT_PATTERN.finditer(lowered):
        if len(match.group()) >= _MIN_TOKEN_LENGTH:
            tokens.append(match.group())
    return tokens


class Bm25Index:
    """Rank documents by BM25 scores for a set of query terms."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[str, list[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_length = 0

    def add(self, doc_id: str, text: str) -> None:
        """Store or replace a document and rebuild its term postings."""
        tokens = tokenize(text)
        if doc_id in self._documents:
            self.remove(doc_id)
        self._documents[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)
        self._total_length += len(tokens)
        frequencies = Counter(tokens)
        for term, count in frequencies.items():
            postings = self._postings.setdefault(term, {})
            postings[doc_id] = count

    def remove(self, doc_id: str) -> None:
        """Drop a document and its term postings."""
        tokens = self._documents.pop(doc_id, [])
        # Drop the length entry also for an empty document.
        self._total_length -= self._doc_lengths.pop(doc_id, 0)
        if not tokens:
            return
        for term in set(tokens):
            postings = self._postings.get(term)
            if postings is None:
                continue
            postings.pop(doc_id, None)
            if not postings:
                self._postings.pop(term, None)

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return document ids ranked by BM25 score, descending.

        Empty query or an empty index returns an empty list. Ties break
        by the document id.
        """
        terms = tokenize(query)
        if not terms or not self._documents:
            return []
        scores: dict[str, float] = {}
        num_documents = len(self._documents)
        average_length = self._total_length / num_documents
        for term in set(terms):
            postings = self._postings.get(term)
            if postings is None:
                continue
            freq = len(postings)
            idf = math.log(1 + (num_documents - freq + 0.5) / (freq + 0.5))
            for doc_id, tf in postings.items():
                length = self._doc_lengths[doc_id]
                denominator = self.k1 * (1 - self.b + self.b * (length / average_length)) + tf
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    (tf * (self.k1 + 1)) / denominator
                )
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ordered[:limit]
