"""OpenAI-compatible embedding client for retrieval-augmented generation.

The client posts batches of texts to {base_url}/embeddings and returns their
vectors in input order. Only 429, 5xx and network errors are retried by the
retry policy; other 4xx responses are terminal.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Self

import httpx

from agent.llm_errors import classify_exception, classify_http_error
from agent.model_config import ProviderSpec
from agent.redaction import redact
from agent.retry_policy import backoff

__all__ = ["EmbeddingClient", "EmbeddingError"]


class EmbeddingError(RuntimeError):
    """An embedding request failed. The message never contains credentials."""


class EmbeddingClient:
    """OpenAI-compatible /embeddings client bound to one provider and model."""

    def __init__(
        self,
        provider: ProviderSpec,
        model: str,
        *,
        batch_size: int = 64,
    ) -> None:
        """Initialize the client with a provider spec and an embedding model name."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._provider = provider
        self._model = model
        self._batch_size = batch_size
        self._http: httpx.AsyncClient | None = None
        self._api_key = os.environ.get(provider.api_key_env, "") if provider.api_key_env else ""

    @property
    def endpoint(self) -> str:
        """Return the /embeddings URL for this client."""
        return f"{self._provider.base_url.rstrip('/')}/embeddings"

    @property
    def batch_size(self) -> int:
        """Return the maximum number of texts per request."""
        return self._batch_size

    def _client(self) -> httpx.AsyncClient:
        """Return the pooled HTTP client, creating it on first use."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._provider.timeout_seconds, connect=15.0),
                follow_redirects=False,
            )
        return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    def _safe(self, message: str) -> str:
        """Redact the configured API key from a message for error text or logs."""
        secrets = (self._api_key,) if self._api_key else ()
        return redact(message, secrets)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed all texts and return vectors in input order.

        Empty or whitespace-only strings raise EmbeddingError before any
        request. An empty sequence returns an empty list. Raises EmbeddingError
        when the returned vectors have inconsistent dimensions.
        """
        if not texts:
            return []
        for text in texts:
            if not text.strip():
                raise EmbeddingError("embedding input must not be empty or whitespace-only")
        indexed: list[tuple[int, list[float]]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            indexed.extend(await self._embed_batch(batch, start))
        indexed.sort(key=lambda item: item[0])
        vectors = [vector for _, vector in indexed]
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding response is missing vectors")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors[1:]):
            raise EmbeddingError("embedding responses have inconsistent dimensions")
        return vectors

    async def _embed_batch(self, texts: list[str], offset: int) -> list[tuple[int, list[float]]]:
        """Send one batch of texts and return (global index, vector) pairs."""
        if self._provider.api_key_env and not self._api_key:
            # A provider that names a key variable needs it; a local provider without
            # api_key_env (for example LM Studio or Ollama) works without a key.
            raise EmbeddingError("embedding API key is not configured")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model, "input": texts}
        body = json.dumps(payload).encode("utf-8")
        client = self._client()
        retries = self._provider.max_retries

        for attempt in range(retries + 1):
            try:
                response = await client.post(self.endpoint, headers=headers, content=body)
            except asyncio.CancelledError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                info = classify_exception(exc)
                info = replace(info, message=self._safe(info.message))
                if attempt >= retries:
                    raise EmbeddingError(f"embedding network error: {info.message}") from exc
                await asyncio.sleep(backoff(attempt))
                continue
            except Exception as exc:
                raise EmbeddingError(f"embedding transport error: {self._safe(str(exc))}") from exc

            info = classify_http_error(response.status_code, response.text)
            if info.retryable and attempt < retries:
                await asyncio.sleep(backoff(attempt))
                continue
            if response.status_code >= 400:
                raise EmbeddingError(
                    f"embedding returned HTTP {response.status_code}: {self._safe(info.message)}"
                )

            try:
                data = response.json()
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("embedding returned an invalid response") from exc
            if not isinstance(data, dict):
                raise EmbeddingError("embedding returned an invalid response")

            pairs = self._extract_vectors(data, offset)
            # Each input of the batch must get exactly one vector: no gaps, no repeats.
            if sorted(index for index, _ in pairs) != list(range(offset, offset + len(texts))):
                raise EmbeddingError("embedding response indexes do not match the batch")
            return pairs

        raise EmbeddingError("embedding request failed after retries")

    @staticmethod
    def _extract_vectors(body: dict[str, Any], offset: int) -> list[tuple[int, list[float]]]:
        """Parse the response data list into (global index, vector) pairs."""
        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError("embedding response is missing data list")
        parsed: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingError("embedding response item is not an object")
            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                raise EmbeddingError("embedding response item is malformed")
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError):
                raise EmbeddingError("embedding values must be numbers") from None
            if not vector or not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("embedding vector must be non-empty and finite")
            parsed.append((offset + index, vector))
        return parsed
