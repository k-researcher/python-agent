"""Tests for :mod:`agent.tools.web_fetch`.

NOTE: ``src/agent/tools/web_fetch.py`` is currently an incomplete stub:
``_fetch`` is a placeholder that always raises ``ToolError`` and
``_read_body`` raises ``NotImplementedError`` (lines 331-387).  The tests
below encode the documented contract and will fail against the stub,
pinpointing the missing behaviour.  See the task summary for the defects.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest

from agent.config import Settings
from agent.tools.base import ToolContext, ToolError
from agent.tools.web_fetch import (
    HtmlToMarkdown,
    WebFetchTool,
    _decode_body,
    _extract_title,
)

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_settings(*, max_bytes: int = 1_000_000) -> Settings:
    return Settings(
        network_allowlist=["example.com"],
        http_max_response_bytes=max_bytes,
    )


def make_tool(settings: Settings, handler: Handler) -> WebFetchTool:
    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return WebFetchTool(settings, client_factory=client_factory)


class AuditRecorder:
    """Records every call to the audit egress callback."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, str, str, str | None]] = []

    async def __call__(
        self,
        category: str,
        destination: str,
        operation: str,
        payload: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        self.events.append((category, destination, operation, payload, status, detail))


def make_context(audit: AuditRecorder | None = None) -> ToolContext:
    return ToolContext(session_id="s", project_root=Path("/tmp/project"), audit_egress=audit)


def patched_validate(calls: list[str], allow: str = "example.com") -> Callable[..., Awaitable[str]]:
    """Fake ``validate_outbound_url`` that records every call target."""

    async def fake(url: str, allowlist: list[str]) -> str:
        calls.append(url)
        if allow not in url:
            raise ToolError(f"Destination host is not in AGENT_NETWORK_ALLOWLIST: {url}")
        return url

    return fake


def html_page(*, title: str = "Hello, World!") -> str:
    return (
        "<html><head><title>"
        + title
        + "</title></head><body>"
        + "<h1>Intro</h1>"
        + "<p>Some <b>text</b> here and a "
        + '<a href="/guide">relative link</a> plus an '
        + '<a href="mailto:a@b.c">email</a> plus a '
        + '<a href="javascript:alert(1)">evil</a> and a '
        + '<a href="#">fragment</a> link.</p>'
        + "<script>var x = 1;</script>"
        + "<style>p { color: red; }</style>"
        + "<ul><li>alpha</li><li>beta</li></ul>"
        + "<ol><li>one</li><li>two</li></ol>"
        + "<pre><code>print('hi')</code></pre>"
        + "<table><tr><th>Name</th><th>Value</th></tr>"
        + "<tr><td>a</td><td>1</td></tr></table>"
        + "</body></html>"
    )


# --------------------------------------------------------------------------- #
# HtmlToMarkdown (pure, real code)
# --------------------------------------------------------------------------- #


def render(raw: str, base_url: str = "https://example.com/start") -> str:
    parser = HtmlToMarkdown(base_url)
    parser.feed(raw)
    return parser.markdown()


def test_html_to_markdown_headings_links_lists_code_table() -> None:
    md = render(html_page())

    assert "# Intro" in md
    assert "Some **text** here" in md or "Some text here" in md
    # relative link became absolute
    assert "[relative link](https://example.com/guide)" in md
    assert "[email](mailto:a@b.c)" in md
    # javascript: link must produce no URL target
    assert "[evil](javascript:alert(1))" not in md
    assert "alert(1)" not in md
    # lists
    assert "- alpha" in md
    assert "- beta" in md
    assert "1. one" in md
    assert "2. two" in md
    # code fence
    assert "```" in md
    assert "print('hi')" in md
    # table
    assert "Name | Value" in md
    assert "a | 1" in md


def test_script_and_style_content_is_removed() -> None:
    raw = "<html><head><style>.x{}</style></head><body><script>bad()</script>ok</body></html>"
    md = render(raw)
    assert "bad()" not in md
    assert ".x{}" not in md
    assert "ok" in md


def test_charset_cp1251_decode() -> None:
    body = "Привет".encode("cp1251")
    assert _decode_body(body, "text/html; charset=cp1251") == "Привет"


def test_extract_title() -> None:
    assert _extract_title(html_page()) == "Hello, World!"
    assert _extract_title("<html><body>no title</body></html>") == ""


# --------------------------------------------------------------------------- #
# full fetch contract (fails: _fetch / _read_body are stubs)
# --------------------------------------------------------------------------- #


def make_page_handler(body: bytes, content_type: str = "text/html") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": content_type}, content=body)

    return handler


async def test_fetch_converts_sample_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = html_page().encode("utf-8")
    calls: list[str] = []
    monkeypatch.setattr("agent.tools.web_fetch.validate_outbound_url", patched_validate(calls))
    tool = make_tool(make_settings(), make_page_handler(body))
    result = await tool.execute(make_context(), {"url": "https://example.com/"})
    assert "https://example.com/" in calls
    assert "# Intro" in result["markdown"]
    assert result["truncated"] is False


async def test_fetch_removes_script_style_via_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "<html><body><script>bad()</script><style>.x{}</style>ok</body></html>"
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), make_page_handler(raw.encode("utf-8")))
    result = await tool.execute(make_context(), {"url": "https://example.com/"})
    assert "bad()" not in result["markdown"]
    assert ".x{}" not in result["markdown"]
    assert "ok" in result["markdown"]


async def test_javascript_link_without_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '<a href="javascript:alert(1)">click</a>'
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), make_page_handler(raw.encode("utf-8")))
    result = await tool.execute(make_context(), {"url": "https://example.com/"})
    assert "javascript:" not in result["markdown"]
    assert "click" in result["markdown"]


async def test_relative_link_becomes_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '<a href="/guide">rel</a>'
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), make_page_handler(raw.encode("utf-8")))
    result = await tool.execute(make_context(), {"url": "https://example.com/a/b"})
    assert "[rel](https://example.com/guide)" in result["markdown"]


async def test_redirect_to_allowed_host_is_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"Location": "https://example.com/final"},
                content=b"",
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=b"<html><body>landed</body></html>",
        )

    calls: list[str] = []
    monkeypatch.setattr("agent.tools.web_fetch.validate_outbound_url", patched_validate(calls))
    tool = make_tool(make_settings(), handler)
    result = await tool.execute(make_context(), {"url": "https://example.com/start"})
    assert calls == ["https://example.com/start", "https://example.com/final"]
    assert "landed" in result["markdown"]


async def test_redirect_to_forbidden_host_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://evil.com/x"}, content=b"")

    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),  # raises because evil.com not allowed
    )
    tool = make_tool(make_settings(), handler)
    with pytest.raises(ToolError, match="ALLOWLIST"):
        await tool.execute(make_context(), {"url": "https://example.com/start"})


async def test_six_redirects_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.path.strip("/") or "0")
        return httpx.Response(
            302, headers={"Location": f"https://example.com/{n + 1}"}, content=b""
        )

    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), handler)
    with pytest.raises(ToolError, match="Too many redirects"):
        await tool.execute(make_context(), {"url": "https://example.com/0"})


async def test_response_over_max_bytes_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = b"x" * 2048
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(max_bytes=1024), make_page_handler(big))
    with pytest.raises(ToolError):
        await tool.execute(make_context(), {"url": "https://example.com/"})


async def test_unsupported_content_type_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), make_page_handler(b"pdf", content_type="application/pdf"))
    with pytest.raises(ToolError):
        await tool.execute(make_context(), {"url": "https://example.com/file.pdf"})


async def test_truncation_sets_truncated_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ("<html><body>" + "word " * 3000 + "</body></html>").encode("utf-8")
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    # web_fetch.max_chars has a 1000 lower bound; use a large body so 1000 is a cut.
    tool = make_tool(make_settings(), make_page_handler(body))
    result = await tool.execute(make_context(), {"url": "https://example.com/", "max_chars": 1000})
    assert len(result["markdown"]) <= 1000
    assert result["truncated"] is True


async def test_audit_started_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    audit = AuditRecorder()
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), make_page_handler(b"<html><body>hi</body></html>"))
    await tool.execute(make_context(audit), {"url": "https://example.com/"})
    statuses = [e[4] for e in audit.events]
    assert statuses == ["started", "completed"]
    # completed audit uses the final (post-redirect) url


async def test_audit_started_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = AuditRecorder()
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(
        make_settings(),
        make_page_handler(b"<html></html>", content_type="application/pdf"),
    )
    with pytest.raises(ToolError):
        await tool.execute(make_context(audit), {"url": "https://example.com/"})
    statuses = [e[4] for e in audit.events]
    assert statuses == ["started", "error"]


# --------------------------------------------------------------------------- #
# validation edge cases (network.py is real)
# --------------------------------------------------------------------------- #


async def test_max_chars_out_of_range_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.tools.web_fetch.validate_outbound_url",
        patched_validate([]),
    )
    tool = make_tool(make_settings(), make_page_handler(b"<html></html>"))
    for bad in (999, 200001):
        with pytest.raises(ToolError, match="max_chars"):
            await tool.execute(make_context(), {"url": "https://example.com/", "max_chars": bad})
