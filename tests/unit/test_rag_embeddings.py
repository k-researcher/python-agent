"""Unit tests for the RAG embedding client (no network access)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agent.model_config import ProviderSpec
from agent.rag.embeddings import EmbeddingClient, EmbeddingError

_KEY_ENV = "AGENT_TEST_EMBEDDING_KEY"
_SECRET = "sk-test-embed-key-value"


def _provider(*, max_retries: int = 0) -> ProviderSpec:
    return ProviderSpec(
        kind="openai",
        base_url="https://emb.example/v1",
        api_key_env=_KEY_ENV,
        timeout_seconds=10,
        max_retries=max_retries,
    )


def _client(
    handler: Any, monkeypatch: Any, *, max_retries: int = 0, batch_size: int = 64
) -> EmbeddingClient:
    """Create a client with a MockTransport and a test API key in the environment."""
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    client = EmbeddingClient(
        _provider(max_retries=max_retries), "text-emb-3", batch_size=batch_size
    )
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _ok_response(count: int, *, reverse: bool = False) -> dict[str, Any]:
    """Return a valid embedding response for ``count`` texts."""
    indices = range(count)
    if reverse:
        indices = reversed(list(indices))
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i), float(i) * 2]} for i in indices
    ]
    return {"data": data, "model": "text-emb-3"}


def test_embed_empty_input(monkeypatch: Any) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_ok_response(0))

    client = _client(handler, monkeypatch)
    assert asyncio.run(client.embed([])) == []
    assert calls == 0
    asyncio.run(client.aclose())


def test_embed_whitespace_raises_before_request(monkeypatch: Any) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_ok_response(1))

    client = _client(handler, monkeypatch)
    with pytest.raises(EmbeddingError, match="empty or whitespace"):
        asyncio.run(client.embed(["  \t\n  "]))
    assert calls == 0
    asyncio.run(client.aclose())


def test_embed_sorts_by_index(monkeypatch: Any) -> None:
    """Response items arrive out of order; the client restores input order."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response(3, reverse=True))

    client = _client(handler, monkeypatch)
    vectors = asyncio.run(client.embed(["a", "b", "c"]))
    assert vectors == [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]
    asyncio.run(client.aclose())


def test_embed_batching_exact_request_count(monkeypatch: Any) -> None:
    """Batch size 2 with 5 texts produces exactly 3 HTTP requests."""
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        texts = payload["input"]
        batch_sizes.append(len(texts))
        return httpx.Response(200, json=_ok_response(len(texts)))

    client = _client(handler, monkeypatch, batch_size=2)
    vectors = asyncio.run(client.embed(["t0", "t1", "t2", "t3", "t4"]))
    assert batch_sizes == [2, 2, 1]
    assert len(vectors) == 5
    asyncio.run(client.aclose())


def test_embed_batch_payload_shape(monkeypatch: Any) -> None:
    """The request body contains the model name and the input list."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "text-emb-3"
        assert payload["input"] == ["a", "b"]
        assert request.headers["authorization"] == f"Bearer {_SECRET}"
        return httpx.Response(200, json=_ok_response(2))

    client = _client(handler, monkeypatch)
    vectors = asyncio.run(client.embed(["a", "b"]))
    assert len(vectors) == 2
    asyncio.run(client.aclose())


def test_embed_inconsistent_dimensions(monkeypatch: Any) -> None:
    """Vectors with different dimensions raise EmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 2.0, 3.0]},
                    {"index": 1, "embedding": [1.0, 2.0]},
                ]
            },
        )

    client = _client(handler, monkeypatch)
    with pytest.raises(EmbeddingError, match="inconsistent dimensions"):
        asyncio.run(client.embed(["a", "b"]))
    asyncio.run(client.aclose())


def test_embed_400_no_retry(monkeypatch: Any) -> None:
    """A 400 response is terminal and is never retried."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    client = _client(handler, monkeypatch, max_retries=2)
    with pytest.raises(EmbeddingError) as caught:
        asyncio.run(client.embed(["a"]))
    assert calls == 1
    assert "bad request" in str(caught.value)
    asyncio.run(client.aclose())


def test_embed_503_then_success_retries(monkeypatch: Any) -> None:
    """A 503 response triggers one retry and then succeeds."""
    statuses = iter([503, 200])
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = next(statuses)
        if status == 503:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_ok_response(1))

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.rag.embeddings.asyncio.sleep", instant_sleep)
    client = _client(handler, monkeypatch, max_retries=1)
    vectors = asyncio.run(client.embed(["a"]))
    assert calls == 2
    assert vectors == [[0.0, 0.0]]
    asyncio.run(client.aclose())


def test_embed_network_error_retries(monkeypatch: Any) -> None:
    """A network error is retried and then succeeds."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_ok_response(1))

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.rag.embeddings.asyncio.sleep", instant_sleep)
    client = _client(handler, monkeypatch, max_retries=2)
    vectors = asyncio.run(client.embed(["a"]))
    assert calls == 2
    assert vectors == [[0.0, 0.0]]
    asyncio.run(client.aclose())


def test_key_not_in_error_message(monkeypatch: Any) -> None:
    """The API key never appears in the error message text."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": f"key {_SECRET} rejected"}})

    client = _client(handler, monkeypatch)
    with pytest.raises(EmbeddingError) as caught:
        asyncio.run(client.embed(["a"]))
    error_text = str(caught.value)
    assert _SECRET not in error_text
    assert "sk-test" not in error_text
    asyncio.run(client.aclose())


def test_key_not_in_network_error(monkeypatch: Any) -> None:
    """The API key never appears in a network error message."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {_SECRET}")

    client = _client(handler, monkeypatch)
    with pytest.raises(EmbeddingError) as caught:
        asyncio.run(client.embed(["a"]))
    assert _SECRET not in str(caught.value)
    asyncio.run(client.aclose())


def test_embed_missing_vectors_raises(monkeypatch: Any) -> None:
    """A server that returns too few embeddings triggers EmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response(1))

    client = _client(handler, monkeypatch)
    with pytest.raises(EmbeddingError, match="missing vectors|indexes do not match"):
        asyncio.run(client.embed(["a", "b"]))
    asyncio.run(client.aclose())


async def test_aclose_closes_http(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response(1))

    client = _client(handler, monkeypatch)
    await client.embed(["a"])
    await client.aclose()
    assert client._http is None or client._http.is_closed


async def test_async_context_manager(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response(1))

    async with _client(handler, monkeypatch) as client:
        vectors = await client.embed(["a"])
        assert vectors == [[0.0, 0.0]]


def test_missing_api_key_raises(monkeypatch: Any) -> None:
    """A missing env var produces an EmbeddingError before any request."""
    monkeypatch.delenv(_KEY_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    provider = _provider()
    client = EmbeddingClient(provider, "text-emb-3")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="API key"):
        asyncio.run(client.embed(["a"]))
    asyncio.run(client.aclose())


async def test_local_provider_without_key_sends_no_authorization() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok_response(1))

    provider = ProviderSpec(kind="openai", base_url="http://127.0.0.1:1234/v1", timeout_seconds=5)
    client = EmbeddingClient(provider, "local-emb")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        vectors = await client.embed(["hello"])
    assert len(vectors) == 1
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    "items",
    [
        [{"index": 0, "embedding": [1.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}],
        [{"index": 0, "embedding": [1.0, 2.0]}, {"index": 5, "embedding": [3.0, 4.0]}],
        [{"index": 0, "embedding": [1.0, 2.0]}],
    ],
)
async def test_indexes_must_cover_the_batch_exactly(
    monkeypatch: Any, items: list[dict[str, Any]]
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": items})

    async with _client(handler, monkeypatch) as client:
        with pytest.raises(EmbeddingError, match="indexes do not match"):
            await client.embed(["a", "b"])


@pytest.mark.parametrize("vector", ["[]", "[1.0, NaN]", "[Infinity, 0.0]"])
async def test_empty_or_non_finite_vector_is_rejected(monkeypatch: Any, vector: str) -> None:
    body = '{"data": [{"index": 0, "embedding": ' + vector + "}]}"

    def handler(_request: httpx.Request) -> httpx.Response:
        # Raw JSON: providers can send NaN and Infinity, which json.loads accepts.
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "application/json"}
        )

    async with _client(handler, monkeypatch) as client:
        with pytest.raises(EmbeddingError, match="non-empty and finite"):
            await client.embed(["a"])
