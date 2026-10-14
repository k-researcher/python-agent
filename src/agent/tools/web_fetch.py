from __future__ import annotations

import asyncio
import contextlib
import enum
import html
import re
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from agent.config import Settings
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.network import pinned_client, validate_outbound_url

MAX_REDIRECTS = 5
USER_AGENT = "python-agent/0.1"
URL_SCHEMES = frozenset({"http", "https", "mailto"})
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
PLAIN_CONTENT_TYPES = ("text/plain", "text/markdown", "application/json")
SKIPPED_TAGS = frozenset({"script", "style", "noscript", "svg", "iframe", "head"})
LIST_ITEM_MARKER = "- "


class _HeadingLevel(enum.IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


AuditFn = Callable[[ToolContext, str, str, str, str, str, str | None], Awaitable[None]]
ClientFactory = Callable[[], httpx.AsyncClient]


async def audit(
    context: ToolContext,
    category: str,
    destination: str,
    operation: str,
    payload: str,
    status: str,
    detail: str | None = None,
) -> None:
    """Report an operation to the audit callback, if one is registered."""
    if context.audit_egress is not None:
        await context.audit_egress(category, destination, operation, payload, status, detail)


class HtmlToMarkdown(HTMLParser):
    """Convert a well-formed HTML fragment into Markdown.

    Uses only the standard library. Tables render as pipe-separated rows,
    lists as bullets or numbers, pre/code as fenced blocks.
    """

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._sb: list[str] = []
        # Stack of open tags. "a" is pushed as the href so data can use it.
        self._tags: list[str] = []
        self._list_stack: list[str] = []
        self._ol_counter: list[int] = []
        self._in_pre = False
        self._in_table = False
        self._row: list[str] = []
        self._cell_sb: list[str] = []
        self._in_cell = False
        self._blockquote_depth = 0

    # -- helpers -----------------------------------------------------------

    def _newline(self) -> None:
        if self._sb and self._sb[-1] != "\n":
            self._sb.append("\n")

    def _ensure_blank_line(self) -> None:
        """Introduce a block break, collapsing runs of newlines."""
        while self._sb and self._sb[-1] == "\n":
            self._sb.pop()
        if self._sb:
            self._sb.append("\n\n")

    def _push(self, tag: str, href: str | None = None) -> None:
        self._tags.append(href if tag == "a" and href is not None else tag)

    def _pop_until(self, tag: str | Callable[[str], bool]) -> bool:
        """Pop stack entries until ``tag`` matches; returns True if found."""
        while self._tags:
            top = self._tags.pop()
            if tag(top) if callable(tag) else top == tag:
                return True
        return False

    # -- HTMLParser overrides ----------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIPPED_TAGS:
            self._tags.append(f"skip:{tag}")
            return
        if self._tags and self._tags[-1].startswith("skip:"):
            return
        if tag == "pre":
            self._in_pre = True
            self._newline()
            self._sb.append("```\n")
            self._tags.append("pre")
            return
        if tag == "code" and self._in_pre:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._ensure_blank_line()
            level = _HeadingLevel[tag.upper()].value
            self._sb.append(f"{'#' * level} ")
            self._tags.append("h:" + tag)
            return
        if tag in {"ul", "ol"}:
            self._list_stack.append(tag)
            if tag == "ol":
                self._ol_counter.append(1)
            self._tags.append(tag)
            return
        if tag == "li":
            self._ensure_blank_line() if not self._sb else None
            list_type = self._list_stack[-1] if self._list_stack else "ul"
            if list_type == "ol":
                number = self._ol_counter[-1]
                self._ol_counter[-1] = number + 1
                self._sb.append(f"{number}. ")
            else:
                self._sb.append("- ")
            self._tags.append("li")
            return
        if tag == "br":
            self._newline()
            return
        if tag == "p":
            self._ensure_blank_line()
            self._tags.append("p")
            return
        if tag == "blockquote":
            self._blockquote_depth += 1
            self._ensure_blank_line()
            self._sb.append("> ")
            self._tags.append("blockquote")
            return
        if tag in {"table"}:
            self._in_table = True
            self._ensure_blank_line()
            self._tags.append("table")
            return
        if tag == "tr":
            self._row = []
            self._tags.append("tr")
            return
        if tag in {"td", "th"}:
            self._in_cell = True
            self._cell_sb = []
            self._tags.append("cell:" + tag)
            return
        if tag == "a":
            href = ""
            for attr, value in attrs:
                if attr == "href":
                    href = value or ""
                    break
            href = href.strip()
            if href.startswith("#") or not href:
                href = ""
            self._tags.append(f"a:{href}")
            return
        # Unknown inline tag: keep content, remember to pop it.
        self._tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIPPED_TAGS:
            self._pop_until("skip:" + tag)
            return
        if tag == "pre":
            self._sb.append("\n```")
            self._in_pre = False
            self._pop_until("pre")
            self._newline()
            return
        if tag == "code" and self._in_pre:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._pop_until("h:" + tag)
            self._newline()
            return
        if tag in {"ul", "ol"}:
            if self._list_stack and self._list_stack[-1] == tag:
                self._list_stack.pop()
            if tag == "ol" and self._ol_counter:
                self._ol_counter.pop()
            self._pop_until(tag)
            self._newline()
            return
        if tag == "li":
            self._pop_until("li")
            self._newline()
            return
        if tag == "p":
            self._pop_until("p")
            self._newline()
            return
        if tag == "blockquote":
            self._blockquote_depth = max(0, self._blockquote_depth - 1)
            self._pop_until("blockquote")
            self._newline()
            return
        if tag == "table":
            self._in_table = False
            self._pop_until("table")
            self._newline()
            return
        if tag == "tr":
            self._finish_row()
            self._pop_until("tr")
            return
        if tag in {"td", "th"}:
            if self._in_cell:
                cell = "".join(self._cell_sb).strip()
                if cell:
                    self._row.append(cell)
            self._in_cell = False
            self._pop_until("cell:" + tag)
            return
        if tag == "a":
            self._pop_until(lambda top: top.startswith("a:"))
            return
        # Unknown inline tag: pop the matching one at top if present.
        if self._tags and self._tags[-1] == tag:
            self._tags.pop()

    def handle_data(self, data: str) -> None:
        if self._tags and self._tags[-1].startswith("skip:"):
            return
        if self._in_pre:
            if data:
                self._sb.append(data)
            return
        if self._in_cell:
            self._cell_sb.append(data)
            return
        stacked = self._tags
        if stacked and stacked[-1].startswith("a:"):
            href = stacked[-1][2:]
            text = data.strip()
            if not href:
                self._sb.append(html.escape(data))
                return
            absolute = urljoin(self.base_url, href)
            scheme = urlparse(absolute).scheme.lower()
            if scheme in URL_SCHEMES:
                self._sb.append(f"[{html.escape(text)}]({absolute})")
            else:
                self._sb.append(html.escape(text))
            return
        if data.strip():
            self._sb.append(data)

    def _finish_row(self) -> None:
        if not self._in_table:
            return
        cells = [cell.strip() for cell in self._row]
        if cells:
            self._sb.append(" | ".join(cells) + "\n")

    # -- output ------------------------------------------------------------

    def markdown(self) -> str:
        self.close()
        text = "".join(self._sb)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _extract_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = re.sub(r"<[^>]+>", "", match.group(1))
    return html.unescape(title).strip()


def _decode_body(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    if match:
        charset = match.group(1)
    try:
        return body.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Download a web page and convert HTML to Markdown."
    risk_level = RiskLevel.network_access
    network_capability = True
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "minimum": 1000,
                "maximum": 200000,
                "default": 40000,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        settings: Settings,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory

    def _get_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        # Redirects are followed manually: each target must pass validate_outbound_url, and
        # the pinned transport checks the address again at connect time.
        return pinned_client(self.settings.network_allowlist)

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments["url"])
        try:
            max_chars = int(arguments.get("max_chars", 40000))
        except (TypeError, ValueError) as exc:
            raise ToolError("max_chars must be an integer") from exc
        if not 1000 <= max_chars <= 200000:
            raise ToolError("max_chars must be between 1000 and 200000")

        headers = {"User-Agent": USER_AGENT}
        await audit(context, "web_fetch", url, "GET", "", "started")
        client = self._get_client()
        try:
            result = await self._fetch(context, client, url, max_chars, headers)
        except asyncio.CancelledError:
            await audit(context, "web_fetch", url, "GET", "", "error", "cancelled")
            raise
        except Exception as exc:
            await audit(context, "web_fetch", url, "GET", "", "error", str(exc))
            raise
        finally:
            # A failure to close the client must not hide the result or the audit record.
            with contextlib.suppress(Exception):
                await client.aclose()
        await audit(context, "web_fetch", result["url"], "GET", "", "completed")
        return result

    async def _fetch(
        self,
        context: ToolContext,
        client: httpx.AsyncClient,
        url: str,
        max_chars: int,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        current_url = url
        redirects = 0
        final_status = 0
        final_content_type = ""
        raw = b""

        for _ in range(MAX_REDIRECTS + 1):
            await validate_outbound_url(current_url, self.settings.network_allowlist)
            body, status, content_type, location = await self._read_body(
                client, current_url, headers
            )
            if 300 <= status < 400:
                redirects += 1
                if redirects > MAX_REDIRECTS:
                    raise ToolError("Too many redirects")
                if not location:
                    raise ToolError("Redirect response is missing a Location header")
                current_url = urljoin(current_url, location)
                continue
            final_status = status
            final_content_type = content_type
            raw = bytes(body)
            break
        else:
            raise ToolError("Too many redirects")

        parsed_content_type = final_content_type.split(";")[0].strip().lower()
        text = _decode_body(raw, final_content_type)
        if parsed_content_type in HTML_CONTENT_TYPES:
            title = _extract_title(text)
            parser = HtmlToMarkdown(current_url)
            parser.feed(text)
            markdown = parser.markdown()
        elif parsed_content_type in PLAIN_CONTENT_TYPES:
            title = ""
            markdown = text
        else:
            raise ToolError(f"Unsupported content type: {final_content_type}")

        return {
            "url": current_url,
            "status": final_status,
            "title": title,
            "markdown": markdown[:max_chars],
            "truncated": len(markdown) > max_chars,
        }

    async def _read_body(
        self, client: httpx.AsyncClient, url: str, headers: dict[str, str]
    ) -> tuple[bytearray, int, str, str]:
        collected = bytearray()
        status = 0
        content_type = ""
        location = ""
        async with client.stream("GET", url, headers=headers) as response:
            status = response.status_code
            content_type = response.headers.get("content-type", "")
            location = response.headers.get("location", "")
            async for chunk in response.aiter_bytes():
                collected.extend(chunk)
                if len(collected) > self.settings.http_max_response_bytes:
                    raise ToolError("HTTP response exceeded the configured size limit")
        return collected, status, content_type, location
