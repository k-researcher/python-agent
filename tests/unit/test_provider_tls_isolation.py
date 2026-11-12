"""Check that a provider CA bundle is trusted only by that provider's connections.

The tests use a real local TLS server and test CAs. They do not use an external
service, and they do not change the system trust store.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import ssl
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from agent.llm import LLMClient, LLMError
from agent.model_config import OAuthAuthSpec, ProviderSpec, ReasoningSpec, ResolvedModel
from agent.provider_auth import AuthError, build_ssl_context, provider_ssl_context
from agent.rag.embeddings import EmbeddingClient


@dataclass(frozen=True)
class _Authority:
    key: ec.EllipticCurvePrivateKey
    cert: x509.Certificate

    def pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _authority(name: str) -> _Authority:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - datetime.timedelta(days=1))
        .not_valid_after(_now() + datetime.timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return _Authority(key, cert)


def _server_files(authority: _Authority, folder: Path) -> tuple[Path, Path]:
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(authority.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - datetime.timedelta(days=1))
        .not_valid_after(_now() + datetime.timedelta(days=2))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(authority.key.public_key()),
            critical=False,
        )
        .sign(authority.key, hashes.SHA256())
    )
    cert_file = folder / "server.pem"
    key_file = folder / "server.key"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the standard library sets this name.
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - the standard library sets this name.
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps({"access_token": "local-token", "expires_in": 600}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def tls_url(tmp_path: Path) -> Iterator[tuple[str, _Authority]]:
    """Start a local HTTPS server that a test CA signs; yield its URL and the CA."""
    authority = _authority("Provider Test CA")
    cert_file, key_file = _server_files(authority, tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_address[1]}/", authority
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str, verify: ssl.SSLContext | bool) -> int:
    with httpx.Client(verify=verify, trust_env=False, timeout=5) as client:
        return client.get(url).status_code


def _bundle(tmp_path: Path, *authorities: _Authority) -> str:
    path = tmp_path / "bundle.pem"
    path.write_bytes(b"\n".join(authority.pem() for authority in authorities))
    return str(path)


def test_provider_bundle_connects_to_a_server_that_its_ca_signs(
    tls_url: tuple[str, _Authority], tmp_path: Path
) -> None:
    url, authority = tls_url
    assert _get(url, build_ssl_context(_bundle(tmp_path, authority))) == 200


def test_default_verification_rejects_a_server_that_only_the_bundle_trusts(
    tls_url: tuple[str, _Authority], tmp_path: Path
) -> None:
    url, authority = tls_url
    build_ssl_context(_bundle(tmp_path, authority))
    # A client without the bundle uses the default trust store and must reject the server.
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        _get(url, True)
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        _get(url, ssl.create_default_context())


def test_provider_bundle_does_not_change_the_default_trust_store(
    tls_url: tuple[str, _Authority], tmp_path: Path
) -> None:
    _, authority = tls_url
    before = ssl.create_default_context().get_ca_certs(binary_form=True)
    build_ssl_context(_bundle(tmp_path, authority))
    after = ssl.create_default_context().get_ca_certs(binary_form=True)
    marker = authority.cert.public_bytes(serialization.Encoding.DER)
    assert marker not in after
    assert after == before


def test_provider_bundle_trusts_only_its_own_certificates(
    tls_url: tuple[str, _Authority], tmp_path: Path
) -> None:
    url, authority = tls_url
    other = _authority("Other Test CA")
    context = build_ssl_context(_bundle(tmp_path, other))
    assert isinstance(context, ssl.SSLContext)
    # The context holds only the bundle certificates, with no default roots added.
    assert context.get_ca_certs(binary_form=True) == [
        other.cert.public_bytes(serialization.Encoding.DER)
    ]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    # A server that a different CA signs is rejected, so the bundle never widens trust.
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        _get(url, context)
    assert authority.cert != other.cert


def test_provider_bundle_keeps_the_host_name_check(
    tls_url: tuple[str, _Authority], tmp_path: Path
) -> None:
    url, authority = tls_url
    context = build_ssl_context(_bundle(tmp_path, authority))
    # The server listens on 127.0.0.1, but its certificate names only "localhost".
    wrong_host = url.replace("localhost", "127.0.0.1")
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        _get(wrong_host, context)


def test_bundle_without_certificates_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pem"
    empty.write_text("not a certificate\n", encoding="utf-8")
    with pytest.raises(AuthError, match="CA bundle"):
        build_ssl_context(str(empty))


def test_bundle_with_two_certificates_and_no_line_break_is_rejected(tmp_path: Path) -> None:
    first, second = _authority("First CA"), _authority("Second CA")
    joined = tmp_path / "joined.pem"
    # Copy commands can join two files with no line break between the certificates.
    joined.write_bytes(first.pem().rstrip(b"\n") + second.pem())
    with pytest.raises(AuthError, match="could not be read"):
        build_ssl_context(str(joined))


# --- Production clients of the agent ----------------------------------------


def _model(base_url: str, ca_bundle: str | None, **changes: object) -> ResolvedModel:
    return ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url=base_url,
        api_key="",
        configured=True,
        model="m",
        context_window=16_000,
        max_tokens=1_000,
        timeout_seconds=5,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
        ca_bundle=ca_bundle,
        **changes,  # type: ignore[arg-type]
    )


async def _llm_get(model: ResolvedModel, url: str) -> int:
    client = LLMClient(model)
    try:
        return (await client._client().get(url)).status_code
    finally:
        await client.aclose()


@pytest.fixture
def polluted_env(
    tls_url: tuple[str, _Authority], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Point SSL_CERT_FILE and SSL_CERT_DIR at the test CA, as a careless setup can do."""
    _, authority = tls_url
    folder = tmp_path / "env-ca"
    folder.mkdir()
    bundle = folder / "env.pem"
    bundle.write_bytes(authority.pem())
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.setenv("SSL_CERT_DIR", str(folder))
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    return str(bundle)


def test_llm_client_with_bundle_connects(
    tls_url: tuple[str, _Authority], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, authority = tls_url
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    model = _model(url, _bundle(tmp_path, authority))
    assert asyncio.run(_llm_get(model, url)) == 200


def test_llm_client_without_bundle_ignores_ca_from_environment(
    tls_url: tuple[str, _Authority], polluted_env: str
) -> None:
    url, _ = tls_url
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        asyncio.run(_llm_get(_model(url, None), url))


def test_environment_cannot_widen_the_bundle_of_a_provider(
    tls_url: tuple[str, _Authority], polluted_env: str, tmp_path: Path
) -> None:
    url, _ = tls_url
    other = _authority("Other Test CA")
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        asyncio.run(_llm_get(_model(url, _bundle(tmp_path, other)), url))


def _oauth(url: str) -> OAuthAuthSpec:
    return OAuthAuthSpec(
        type="oauth_client_credentials", token_url=url, key_env="AUTH_KEY", scope="API"
    )


async def _token_headers(model: ResolvedModel) -> dict[str, str]:
    client = LLMClient(model)
    try:
        provider = client._auth_provider()
        assert provider is not None
        return await provider.headers()
    finally:
        await client.aclose()


def test_oauth_token_request_uses_only_the_provider_bundle(
    tls_url: tuple[str, _Authority], polluted_env: str, tmp_path: Path
) -> None:
    url, authority = tls_url
    trusted = _model(url, _bundle(tmp_path, authority), auth=_oauth(url), auth_secret="k")
    assert asyncio.run(_token_headers(trusted))["Authorization"] == "Bearer local-token"
    # Without the bundle, the CA from the environment must not make the token request work.
    untrusted = _model(url, None, auth=_oauth(url), auth_secret="k")
    with pytest.raises((AuthError, LLMError, httpx.ConnectError)):
        asyncio.run(_token_headers(untrusted))


def test_connection_check_gets_only_the_bundle_certificates(
    tls_url: tuple[str, _Authority], polluted_env: str, tmp_path: Path
) -> None:
    url, authority = tls_url

    async def settings() -> ssl.SSLContext | bool:
        client = LLMClient(_model(url, _bundle(tmp_path, authority)))
        try:
            return (await client.transport_settings())[1]
        finally:
            await client.aclose()

    context = asyncio.run(settings())
    assert isinstance(context, ssl.SSLContext)
    assert context.get_ca_certs(binary_form=True) == [
        authority.cert.public_bytes(serialization.Encoding.DER)
    ]


def test_default_provider_context_ignores_ca_from_environment(
    tls_url: tuple[str, _Authority], polluted_env: str
) -> None:
    _, authority = tls_url
    marker = authority.cert.public_bytes(serialization.Encoding.DER)
    assert marker not in provider_ssl_context(None).get_ca_certs(binary_form=True)


async def _embedding_get(provider: ProviderSpec, url: str) -> int:
    client = EmbeddingClient(provider, "embed")
    try:
        return (await client._client().get(url)).status_code
    finally:
        await client.aclose()


def test_embedding_client_uses_only_the_provider_bundle(
    tls_url: tuple[str, _Authority], polluted_env: str, tmp_path: Path
) -> None:
    url, authority = tls_url
    trusted = ProviderSpec(kind="openai", base_url=url, ca_bundle=_bundle(tmp_path, authority))
    assert asyncio.run(_embedding_get(trusted, url)) == 200
    plain = ProviderSpec(kind="openai", base_url=url)
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        asyncio.run(_embedding_get(plain, url))


def test_connection_check_client_without_bundle_ignores_ca_from_environment(
    tls_url: tuple[str, _Authority], polluted_env: str
) -> None:
    from agent.settings_api import connection_client

    url, _ = tls_url

    async def get() -> int:
        async with connection_client() as client:
            return (await client.get(url)).status_code

    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        asyncio.run(get())
