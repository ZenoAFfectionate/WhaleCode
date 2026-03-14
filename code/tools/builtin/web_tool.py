"""Web search and fetch tools for the coding agent.

Provides internet search capabilities using DuckDuckGo and web page content
extraction for reading documentation, articles, and other online resources.

Usage:
    >>> from hello_agents.tools.builtin.web_tool import WebSearchTool, WebFetchTool
    >>> registry.register_tool(WebSearchTool())
    >>> registry.register_tool(WebFetchTool())
"""

from __future__ import annotations

import html as html_module
from html.parser import HTMLParser
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse


# ---------------------------------------------------------------------------
# HTML text extractor (stdlib only, no extra dependencies)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML using the stdlib parser."""

    SKIP_TAGS = frozenset({
        "script", "style", "nav", "footer", "header",
        "noscript", "svg", "iframe", "form",
    })
    BLOCK_TAGS = frozenset({
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "pre", "br", "hr",
        "section", "article", "main", "dt", "dd",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(html_module.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(html_module.unescape(f"&#{name};"))

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = [line.strip() for line in raw.splitlines()]
        result: List[str] = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        return "\n".join(result).strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # Fallback: strip tags with regex if the parser chokes
        text = re.sub(r"<[^>]+>", " ", html)
        text = html_module.unescape(text)
        return re.sub(r"\s+", " ", text).strip()
    return extractor.get_text()


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------

class WebSearchTool(Tool):
    """Search the web using DuckDuckGo.

    Requires the ``ddgs`` package::

        pip install ddgs
    """

    DEFAULT_MAX_RESULTS = 5
    MAX_RESULTS_LIMIT = 20

    def __init__(self, name: str = "WebSearch") -> None:
        super().__init__(
            name=name,
            description=(
                "Search the web for current information using DuckDuckGo. "
                "Use this when you need up-to-date information, documentation, "
                "error solutions, API references, or any knowledge not available "
                "in the local repository. Returns titles, URLs, and snippets."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description=(
                    "Search query string. Be specific and include relevant "
                    "keywords such as language/framework names."
                ),
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description=(
                    f"Maximum number of results to return "
                    f"(default: {self.DEFAULT_MAX_RESULTS}, "
                    f"max: {self.MAX_RESULTS_LIMIT})."
                ),
                required=False,
                default=self.DEFAULT_MAX_RESULTS,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = str(parameters.get("query", "")).strip()
        max_results = min(
            int(parameters.get("max_results", self.DEFAULT_MAX_RESULTS)),
            self.MAX_RESULTS_LIMIT,
        )

        if not query:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="query must be a non-empty string.",
            )

        # Import ddgs (formerly duckduckgo-search)
        try:
            from ddgs import DDGS  # type: ignore[import-untyped]
        except ImportError:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=(
                    "ddgs package is not installed. "
                    "Install it with: pip install ddgs"
                ),
            )

        # Execute search
        try:
            ddgs = DDGS()
            raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.NETWORK_ERROR,
                message=f"Web search failed: {exc}",
            )

        if not raw_results:
            return ToolResponse.success(
                text=f"No results found for: {query}",
                data={"query": query, "results": [], "count": 0},
            )

        # Format results for the LLM
        lines = [
            f"Search results for: {query}",
            f"Found {len(raw_results)} results:",
            "",
        ]

        cleaned: List[Dict[str, str]] = []
        for i, item in enumerate(raw_results, 1):
            title = item.get("title", "No title")
            url = item.get("href", item.get("link", ""))
            snippet = item.get("body", item.get("snippet", ""))

            lines.append(f"[{i}] {title}")
            lines.append(f"    URL: {url}")
            if snippet:
                lines.append(f"    {snippet}")
            lines.append("")

            cleaned.append({"title": title, "url": url, "snippet": snippet})

        return ToolResponse.success(
            text="\n".join(lines),
            data={"query": query, "results": cleaned, "count": len(cleaned)},
        )


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------

class WebFetchTool(Tool):
    """Fetch and extract readable text content from a web URL."""

    MAX_CONTENT_LENGTH = 50_000   # characters
    REQUEST_TIMEOUT = 15          # seconds

    def __init__(self, name: str = "WebFetch") -> None:
        super().__init__(
            name=name,
            description=(
                "Fetch a web page and extract its readable text content. "
                "Use this after WebSearch to read the full content of a result, "
                "or to read online documentation, blog posts, and articles by URL."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="The URL to fetch (must include http:// or https://).",
                required=True,
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description=(
                    f"Maximum content length in characters "
                    f"(default: {self.MAX_CONTENT_LENGTH})."
                ),
                required=False,
                default=self.MAX_CONTENT_LENGTH,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        url = str(parameters.get("url", "")).strip()
        max_length = int(parameters.get("max_length", self.MAX_CONTENT_LENGTH))

        if not url:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="url must be a non-empty string.",
            )

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid URL: {url}. Must start with http:// or https://.",
            )

        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="requests package is not installed.",
            )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; CodeAgent/1.0; "
                "+https://github.com/hello-agents)"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        }

        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as exc:
            name = type(exc).__name__
            if "Timeout" in name:
                code = ToolErrorCode.TIMEOUT
            elif "ConnectionError" in name:
                code = ToolErrorCode.NETWORK_ERROR
            elif "HTTPError" in name:
                code = ToolErrorCode.API_ERROR
            else:
                code = ToolErrorCode.NETWORK_ERROR
            return ToolResponse.error(code=code, message=f"Request failed: {exc}")

        content_type = resp.headers.get("Content-Type", "")

        if any(ct in content_type for ct in ("text/html", "application/xhtml")):
            text = _html_to_text(resp.text)
        elif any(ct in content_type for ct in ("text/plain", "application/json",
                                                "text/xml", "application/xml")):
            text = resp.text
        else:
            return ToolResponse.partial(
                text=(
                    f"Unsupported content type: {content_type}. "
                    "Only HTML, plain text, JSON, and XML are supported."
                ),
                data={"url": url, "content_type": content_type},
            )

        truncated = len(text) > max_length
        if truncated:
            text = text[:max_length]

        lines = [
            f"Content from: {url}",
            f"Content type: {content_type}",
            f"Length: {len(text)} chars" + (" (truncated)" if truncated else ""),
            "",
            text,
        ]

        factory = ToolResponse.partial if truncated else ToolResponse.success
        return factory(
            text="\n".join(lines),
            data={
                "url": url,
                "content_type": content_type,
                "length": len(text),
                "truncated": truncated,
            },
        )
