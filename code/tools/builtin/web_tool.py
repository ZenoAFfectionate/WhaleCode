"""Web search and fetch tools for the coding agent.

Provides controllable internet access for discovery and retrieval without relying
on Exa/MCP-style backends. Search uses DuckDuckGo via ``ddgs`` and fetch uses
``requests`` with bounded downloads, strict parameter validation, and tool-local
output truncation/persistence.
"""

from __future__ import annotations

import html as html_module
import os
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple
from urllib.parse import urlparse

from ...context.truncator import ObservationTruncator
from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _trim_display_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


_DOMAIN_PATTERN = re.compile(
    r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)


def _is_valid_domain(host: str) -> bool:
    host = host.lower().strip(".")
    return bool(host) and ".." not in host and bool(_DOMAIN_PATTERN.fullmatch(host))


def _parse_domain_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = []
        for item in value:
            if not isinstance(item, str):
                return None
            raw_items.append(item.strip())
    else:
        return None

    normalized: List[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not item:
            continue
        host = _normalize_domain(item)
        if not host or not _is_valid_domain(host):
            return None
        if host in seen:
            continue
        seen.add(host)
        normalized.append(host)
    return normalized


def _normalize_domain(value: str) -> str:
    candidate = value.strip().lower()
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = "//" + candidate
    parsed = urlparse(candidate)
    host = parsed.netloc or parsed.path
    host = host.split("@")[-1].split(":")[0].strip().strip(".")
    return host


def _host_matches_domain(host: str, domain: str) -> bool:
    host = host.lower().strip(".")
    domain = domain.lower().strip(".")
    return bool(host and domain) and (host == domain or host.endswith("." + domain))


def _extract_host(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower().strip(".")


def _build_effective_search_query(
    query: str,
    *,
    include_domains: List[str],
    exclude_domains: List[str],
) -> str:
    parts = [query.strip()]
    if include_domains:
        if len(include_domains) == 1:
            parts.append(f"site:{include_domains[0]}")
        else:
            include_clause = " OR ".join(f"site:{domain}" for domain in include_domains)
            parts.append(f"({include_clause})")
    parts.extend(f"-site:{domain}" for domain in exclude_domains)
    return _normalize_whitespace(" ".join(part for part in parts if part))


def _punctuation_count(text: str) -> int:
    punctuation = ".,:;!?，。；：！？"
    return sum(text.count(char) for char in punctuation)


class _ResponseTooLargeError(Exception):
    """Raised when a fetched response exceeds the configured size limit."""


class _SearchBackend(Protocol):
    """Minimal interface for pluggable search backends."""

    def search_text(
        self,
        query: str,
        *,
        max_results: int,
        region: str,
        safesearch: str,
        timelimit: Optional[str],
        timeout_seconds: int,
        backend: str,
    ) -> List[Dict[str, Any]]:
        ...


class _DuckDuckGoSearchBackend:
    """DuckDuckGo-backed search backend implemented with ddgs."""

    def __init__(self, proxy: Optional[str] = None, verify: bool | str = True) -> None:
        self.proxy = proxy
        self.verify = verify

    def search_text(
        self,
        query: str,
        *,
        max_results: int,
        region: str,
        safesearch: str,
        timelimit: Optional[str],
        timeout_seconds: int,
        backend: str,
    ) -> List[Dict[str, Any]]:
        from ddgs import DDGS  # type: ignore[import-untyped]

        with DDGS(proxy=self.proxy, timeout=timeout_seconds, verify=self.verify) as ddgs:
            return list(
                ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    backend=backend,
                )
            )


class _HTMLNode:
    """Lightweight HTML node used for semantic main-content extraction."""

    def __init__(
        self,
        tag: str,
        attrs: Optional[Dict[str, str]] = None,
        *,
        parent: Optional["_HTMLNode"] = None,
    ) -> None:
        self.tag = tag.lower()
        self.attrs = attrs or {}
        self.parent = parent
        self.children: List["_HTMLNode"] = []
        self.text_parts: List[str] = []

    def add_child(self, child: "_HTMLNode") -> None:
        self.children.append(child)

    def add_text(self, text: str) -> None:
        if text:
            self.text_parts.append(text)

    def iter_nodes(self) -> Iterable["_HTMLNode"]:
        yield self
        for child in self.children:
            yield from child.iter_nodes()


class _HTMLTreeBuilder(HTMLParser):
    """Build a lightweight DOM tree from HTML."""

    SKIP_TAGS = frozenset({
        "script", "style", "noscript", "iframe", "svg", "canvas", "template",
    })
    VOID_TAGS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.root = _HTMLNode("document")
        self._stack: List[_HTMLNode] = [self.root]
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in self.SKIP_TAGS:
                self._skip_depth += 1
            return
        if tag in self.SKIP_TAGS:
            self._skip_depth = 1
            return

        node = _HTMLNode(
            tag,
            {key.lower(): (value or "") for key, value in attrs},
            parent=self._stack[-1],
        )
        self._stack[-1].add_child(node)
        if tag not in self.VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in self.SKIP_TAGS:
                self._skip_depth = max(0, self._skip_depth - 1)
            return

        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._stack[-1].add_text(data)

    def handle_entityref(self, name: str) -> None:
        if not self._skip_depth:
            self._stack[-1].add_text(html_module.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if not self._skip_depth:
            self._stack[-1].add_text(html_module.unescape(f"&#{name};"))


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML using the stdlib parser."""

    SKIP_TAGS = frozenset({
        "script", "style", "nav", "footer", "header",
        "noscript", "svg", "iframe", "form", "canvas",
    })
    BLOCK_TAGS = frozenset({
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "pre", "br", "hr",
        "section", "article", "main", "dt", "dd",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
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
        text = re.sub(r"<[^>]+>", " ", html)
        text = html_module.unescape(text)
        return re.sub(r"\s+", " ", text).strip()
    return extractor.get_text()


def _extract_html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = html_module.unescape(match.group(1))
    return _normalize_whitespace(title)


_CONTENT_BLOCK_TAGS = frozenset({
    "main", "article", "section", "div", "p", "ul", "ol", "li", "pre",
    "blockquote", "table", "thead", "tbody", "tr", "td", "th", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6",
})
_POSITIVE_ATTR_MARKERS = (
    "content", "article", "post", "entry", "doc", "docs", "documentation",
    "markdown", "readme", "manual", "wiki", "story", "main", "body",
)
_NEGATIVE_ATTR_MARKERS = (
    "nav", "menu", "sidebar", "footer", "header", "comment", "comments",
    "share", "social", "cookie", "banner", "advert", "ads", "promo",
    "breadcrumb", "toc", "pagination", "pager", "toolbar", "related",
    "recommend", "subscribe",
)
_UNLIKELY_MAIN_TAGS = frozenset({"nav", "aside", "footer", "header", "form"})


def _build_html_tree(html: str) -> Optional[_HTMLNode]:
    parser = _HTMLTreeBuilder()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return None
    return parser.root


def _collect_text(node: _HTMLNode) -> str:
    parts: List[str] = []
    for raw in node.text_parts:
        text = _normalize_whitespace(raw)
        if text:
            parts.append(text)
    for child in node.children:
        child_text = _collect_text(child)
        if child_text:
            parts.append(child_text)
    return " ".join(parts).strip()


def _collect_code_text(node: _HTMLNode) -> str:
    parts: List[str] = list(node.text_parts)
    for child in node.children:
        parts.append(_collect_code_text(child))
    return "".join(parts)


def _render_node_text(node: _HTMLNode) -> str:
    if node.tag == "pre":
        code = _collect_code_text(node).strip("\n")
        if not code.strip():
            return ""
        return f"\n```\n{code.rstrip()}\n```\n"
    if node.tag == "code":
        if node.parent and node.parent.tag == "pre":
            return _collect_code_text(node)
        inline = _collect_text(node)
        return f"`{inline}`" if inline else ""
    if node.tag == "br":
        return "\n"

    pieces: List[str] = []
    if node.tag in _CONTENT_BLOCK_TAGS and node.tag != "li":
        pieces.append("\n")
    if node.tag == "li":
        pieces.append("\n- ")

    for raw in node.text_parts:
        text = _normalize_whitespace(raw)
        if not text:
            continue
        if pieces:
            last = pieces[-1]
            if last and not last.endswith((" ", "\n", "`", "- ")):
                pieces.append(" ")
        pieces.append(text)

    for child in node.children:
        child_text = _render_node_text(child)
        if not child_text:
            continue
        if pieces:
            last = pieces[-1]
            if child_text.startswith("`") and last and not last.endswith((" ", "\n", "- ")):
                pieces.append(" ")
            elif not child_text.startswith(("\n", "`")) and last and not last.endswith((" ", "\n", "- ")):
                pieces.append(" ")
        pieces.append(child_text)

    if node.tag in _CONTENT_BLOCK_TAGS and node.tag != "li":
        pieces.append("\n")
    return "".join(pieces)


def _finalize_rendered_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: List[str] = []
    prev_empty = False
    for line in lines:
        if not line.strip():
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
            continue
        cleaned.append(line.strip())
        prev_empty = False
    return "\n".join(cleaned).strip()


def _node_attr_blob(node: _HTMLNode) -> str:
    return " ".join(
        part.strip().lower()
        for part in (
            node.attrs.get("id", ""),
            node.attrs.get("class", ""),
            node.attrs.get("role", ""),
            node.attrs.get("aria-label", ""),
            node.attrs.get("data-testid", ""),
        )
        if part and part.strip()
    )


def _node_metrics(node: _HTMLNode) -> Dict[str, Any]:
    text = _collect_text(node)
    text_len = len(text)
    tag_count = sum(1 for _ in node.iter_nodes()) - 1
    paragraph_count = sum(1 for child in node.iter_nodes() if child.tag == "p")
    heading_count = sum(1 for child in node.iter_nodes() if child.tag in {"h1", "h2", "h3", "h4", "h5", "h6"})
    list_item_count = sum(1 for child in node.iter_nodes() if child.tag == "li")
    code_block_count = sum(1 for child in node.iter_nodes() if child.tag == "pre")
    block_count = sum(1 for child in node.iter_nodes() if child.tag in _CONTENT_BLOCK_TAGS)
    link_text_len = sum(len(_collect_text(child)) for child in node.iter_nodes() if child.tag == "a")
    return {
        "text": text,
        "text_len": text_len,
        "tag_count": max(tag_count, 0),
        "paragraph_count": paragraph_count,
        "heading_count": heading_count,
        "list_item_count": list_item_count,
        "code_block_count": code_block_count,
        "block_count": block_count,
        "link_text_len": link_text_len,
        "punctuation_count": _punctuation_count(text),
    }


def _score_main_candidate(node: _HTMLNode, metrics: Dict[str, Any]) -> float:
    attr_blob = _node_attr_blob(node)
    link_density = metrics["link_text_len"] / max(metrics["text_len"], 1)
    density = metrics["text_len"] / max(metrics["tag_count"] + 1, 1)
    score = 0.0

    if node.tag == "main":
        score += 900
    if node.tag == "article":
        score += 750
    if node.attrs.get("role", "").lower() == "main":
        score += 700

    score += min(metrics["text_len"], 12000) * 1.0
    score += metrics["paragraph_count"] * 120
    score += metrics["heading_count"] * 90
    score += metrics["list_item_count"] * 24
    score += metrics["code_block_count"] * 180
    score += metrics["punctuation_count"] * 6
    score += min(density, 500) * 2

    for marker in _POSITIVE_ATTR_MARKERS:
        if marker in attr_blob:
            score += 140
    for marker in _NEGATIVE_ATTR_MARKERS:
        if marker in attr_blob:
            score -= 220

    if node.tag in _UNLIKELY_MAIN_TAGS:
        score -= 1200
    if link_density > 0.45:
        score -= 800
    elif link_density > 0.25:
        score -= 300
    if metrics["text_len"] < 120:
        score -= 600

    return score


def _select_main_content_node(root: _HTMLNode) -> Tuple[Optional[_HTMLNode], Dict[str, Any]]:
    best_node: Optional[_HTMLNode] = None
    best_score = float("-inf")
    best_metrics: Dict[str, Any] = {}

    for node in root.iter_nodes():
        if node.tag == "document":
            continue
        if node.tag in _UNLIKELY_MAIN_TAGS and _node_attr_blob(node).find("main") == -1:
            continue
        if node.tag not in _CONTENT_BLOCK_TAGS and not _node_attr_blob(node):
            continue

        metrics = _node_metrics(node)
        if metrics["text_len"] < 80:
            continue

        score = _score_main_candidate(node, metrics)
        if score > best_score:
            best_node = node
            best_score = score
            best_metrics = metrics

    if best_node is None:
        return None, {"strategy": "full_page_fallback", "score": None}
    return best_node, {
        "strategy": "main_content",
        "score": best_score,
        "text_len": best_metrics.get("text_len", 0),
        "paragraph_count": best_metrics.get("paragraph_count", 0),
        "heading_count": best_metrics.get("heading_count", 0),
        "link_text_len": best_metrics.get("link_text_len", 0),
    }


def _extract_html_text(html: str, *, mode: str) -> Tuple[str, Dict[str, Any]]:
    full_text = _html_to_text(html)
    if mode == "full":
        return full_text, {"strategy": "full_page", "used_main_candidate": False}

    root = _build_html_tree(html)
    if root is None:
        return full_text, {"strategy": "full_page_fallback", "used_main_candidate": False}

    candidate, info = _select_main_content_node(root)
    if candidate is None:
        return full_text, {"strategy": "full_page_fallback", "used_main_candidate": False}

    candidate_text = _finalize_rendered_text(_render_node_text(candidate))
    full_len = len(full_text)
    candidate_len = len(candidate_text)

    useful_candidate = bool(candidate_text) and (
        candidate_len >= 160
        or candidate_len >= max(80, full_len // 4)
        or info.get("score", 0) >= 1200
    )

    if mode == "main":
        if useful_candidate:
            return candidate_text, {
                **info,
                "used_main_candidate": True,
                "candidate_text_len": candidate_len,
                "full_text_len": full_len,
            }
        return full_text, {
            "strategy": "full_page_fallback",
            "used_main_candidate": False,
            "candidate_text_len": candidate_len,
            "full_text_len": full_len,
        }

    if useful_candidate:
        return candidate_text, {
            **info,
            "used_main_candidate": True,
            "candidate_text_len": candidate_len,
            "full_text_len": full_len,
        }
    return full_text, {
        "strategy": "full_page_fallback",
        "used_main_candidate": False,
        "candidate_text_len": candidate_len,
        "full_text_len": full_len,
    }


class _WebToolBase(Tool):
    """Shared helpers for web tools."""

    DEFAULT_OUTPUT_SUBDIR = "memory/tool-output"
    DEFAULT_RETENTION_DAYS = 7

    def __init__(
        self,
        *,
        name: str,
        description: str,
        project_root: str = ".",
        enabled: bool,
        output_dir: Optional[str],
        output_max_lines: int,
        output_max_bytes: int,
        truncate_direction: str,
        hint_message: str,
    ) -> None:
        super().__init__(name=name, description=description)
        self.project_root = Path(project_root).expanduser().resolve()
        self.enabled = enabled
        self.output_truncator = ObservationTruncator(
            max_lines=output_max_lines,
            max_bytes=output_max_bytes,
            truncate_direction=truncate_direction,
            output_dir=str(self._resolve_output_dir(output_dir)),
            retention_days=self.DEFAULT_RETENTION_DAYS,
            hint_message=hint_message,
        )

    def _resolve_output_dir(self, output_dir: Optional[str]) -> Path:
        if output_dir:
            candidate = Path(output_dir).expanduser()
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
            return candidate.resolve()
        return (self.project_root / self.DEFAULT_OUTPUT_SUBDIR).resolve()

    def _check_enabled(self, tool_id: str) -> Optional[ToolResponse]:
        if self.enabled:
            return None
        return ToolResponse.error(
            code=ToolErrorCode.ACCESS_DENIED,
            message=(
                f"{tool_id} is disabled. "
                f"Enable it with WEB_TOOLS_ENABLED=1 and the tool-specific flag for {tool_id}."
            ),
        )

    @staticmethod
    def _validate_required_string(parameters: Dict[str, Any], name: str) -> Tuple[Optional[str], Optional[ToolResponse]]:
        value = parameters.get(name)
        if value is None:
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid parameter `{name}`: expected non-empty string, got None.",
            )
        if not isinstance(value, str):
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Invalid parameter `{name}`: expected non-empty string, "
                    f"got {type(value).__name__}."
                ),
            )
        text = value.strip()
        if not text:
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid parameter `{name}`: expected non-empty string.",
            )
        return text, None

    @staticmethod
    def _validate_optional_string(
        parameters: Dict[str, Any],
        name: str,
        *,
        default: Optional[str] = None,
        allow_blank: bool = False,
    ) -> Tuple[Optional[str], Optional[ToolResponse]]:
        if name not in parameters or parameters.get(name) is None:
            return default, None
        value = parameters.get(name)
        if not isinstance(value, str):
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Invalid parameter `{name}`: expected string when provided, "
                    f"got {type(value).__name__}."
                ),
            )
        text = value.strip()
        if not text and not allow_blank:
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid parameter `{name}`: expected non-empty string when provided.",
            )
        return text or default, None

    @staticmethod
    def _validate_string_array(
        parameters: Dict[str, Any],
        name: str,
        *,
        default: Optional[List[str]] = None,
    ) -> Tuple[Optional[List[str]], Optional[ToolResponse]]:
        if name not in parameters or parameters.get(name) is None:
            return list(default or []), None
        value = parameters.get(name)
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()], None
        if not isinstance(value, list):
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Invalid parameter `{name}`: expected string array when provided, "
                    f"got {type(value).__name__}."
                ),
            )
        items: List[str] = []
        for item in value:
            if not isinstance(item, str):
                return None, ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=(
                        f"Invalid parameter `{name}`: expected array of strings, "
                        f"got element of type {type(item).__name__}."
                    ),
                )
            text = item.strip()
            if text:
                items.append(text)
        return items, None

    @staticmethod
    def _validate_int(
        parameters: Dict[str, Any],
        name: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> Tuple[Optional[int], Optional[ToolResponse]]:
        raw_value = parameters.get(name, default)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Invalid parameter `{name}`: expected integer between {minimum} and {maximum}, "
                    f"got {type(raw_value).__name__}."
                ),
            )
        if raw_value < minimum or raw_value > maximum:
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    f"Invalid parameter `{name}`: expected integer between {minimum} and {maximum}, "
                    f"got {raw_value}."
                ),
            )
        return raw_value, None

    @staticmethod
    def _validate_choice(
        parameters: Dict[str, Any],
        name: str,
        *,
        default: str,
        allowed: set[str],
    ) -> Tuple[Optional[str], Optional[ToolResponse]]:
        value, error = _WebToolBase._validate_optional_string(parameters, name, default=default)
        if error or value is None:
            return None, error
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            return None, ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid parameter `{name}`: expected one of [{choices}], got {value!r}.",
            )
        return value, None

    def _build_text_response(
        self,
        *,
        tool_name: str,
        output_text: str,
        metadata: Dict[str, Any],
        data: Dict[str, Any],
        force_partial: bool = False,
    ) -> ToolResponse:
        truncation = self.output_truncator.truncate(
            tool_name=tool_name,
            output=output_text,
            metadata=metadata,
        )
        observation_truncated = bool(truncation.get("truncated"))
        full_output_path = truncation.get("full_output_path")
        payload = dict(data)
        payload["output"] = truncation.get("display_preview", truncation.get("preview", output_text))
        payload["truncated"] = force_partial or observation_truncated or bool(payload.get("truncated"))
        payload["observation_truncated"] = observation_truncated
        payload["full_output_path"] = full_output_path
        payload["truncation_stats"] = truncation.get("stats", {})

        factory = ToolResponse.partial if (force_partial or observation_truncated) else ToolResponse.success
        return factory(text=payload["output"], data=payload)


class WebSearchTool(_WebToolBase):
    """Search the web using DuckDuckGo/ddgs without Exa/MCP."""

    DEFAULT_MAX_RESULTS = 8
    MAX_RESULTS_LIMIT = 20
    MAX_BACKEND_RESULTS = 50
    DEFAULT_TIMEOUT_SECONDS = 8
    MAX_TIMEOUT_SECONDS = 30
    DEFAULT_REGION = "us-en"
    DEFAULT_SAFESEARCH = "moderate"
    DEFAULT_BACKEND = "auto"
    FILTER_OVERSAMPLE_FACTOR = 4
    ALLOWED_SAFESEARCH = {"off", "moderate", "on"}
    MAX_SNIPPET_CHARS = 500
    OUTPUT_MAX_LINES = 250
    OUTPUT_MAX_BYTES = 16_000
    ENABLE_ENV = "WEB_SEARCH_ENABLED"

    @classmethod
    def is_enabled_by_default(cls) -> bool:
        return _env_enabled("WEB_TOOLS_ENABLED", True) and _env_enabled(cls.ENABLE_ENV, True)

    def __init__(
        self,
        name: str = "WebSearch",
        project_root: str = ".",
        enabled: Optional[bool] = None,
        output_dir: Optional[str] = None,
        output_max_lines: int = OUTPUT_MAX_LINES,
        output_max_bytes: int = OUTPUT_MAX_BYTES,
        truncate_direction: str = "head",
        search_backend: Optional[_SearchBackend] = None,
    ) -> None:
        super().__init__(
            name=name,
            description=(
                "Search the web for current information using DuckDuckGo via ddgs. "
                "Use this for up-to-date documentation, error solutions, release notes, "
                "or recent news beyond the local repository. When searching for current "
                f"information, include the relevant year such as {datetime.now().year} in the query."
            ),
            project_root=project_root,
            enabled=self.is_enabled_by_default() if enabled is None else bool(enabled),
            output_dir=output_dir,
            output_max_lines=output_max_lines,
            output_max_bytes=output_max_bytes,
            truncate_direction=truncate_direction,
            hint_message="Use WebFetch for a chosen URL or Read the saved output file for the full result list.",
        )
        self.search_backend = search_backend or _DuckDuckGoSearchBackend()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query string. Be specific and include framework, language, version, or year when needed.",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description=(
                    f"Maximum number of results to return (default: {self.DEFAULT_MAX_RESULTS}, "
                    f"max: {self.MAX_RESULTS_LIMIT})."
                ),
                required=False,
                default=self.DEFAULT_MAX_RESULTS,
            ),
            ToolParameter(
                name="region",
                type="string",
                description=f"Optional search region (default: {self.DEFAULT_REGION}).",
                required=False,
                default=self.DEFAULT_REGION,
            ),
            ToolParameter(
                name="safesearch",
                type="string",
                description="Optional safesearch mode: off, moderate, or on.",
                required=False,
                default=self.DEFAULT_SAFESEARCH,
            ),
            ToolParameter(
                name="timelimit",
                type="string",
                description="Optional DDGS time limit such as d, w, m, or y.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="backend",
                type="string",
                description="Optional ddgs backend selector, for example auto or a comma-separated engine list.",
                required=False,
                default=self.DEFAULT_BACKEND,
            ),
            ToolParameter(
                name="include_domains",
                type="array",
                description="Optional domains to include, for example ['docs.python.org', 'python.org']. Subdomains also match.",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="exclude_domains",
                type="array",
                description="Optional domains to exclude. Exclusions take precedence over inclusions.",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="timeout_seconds",
                type="integer",
                description=(
                    f"Request timeout in seconds (default: {self.DEFAULT_TIMEOUT_SECONDS}, "
                    f"max: {self.MAX_TIMEOUT_SECONDS})."
                ),
                required=False,
                default=self.DEFAULT_TIMEOUT_SECONDS,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        disabled = self._check_enabled("websearch")
        if disabled:
            return disabled

        query, error = self._validate_required_string(parameters, "query")
        if error:
            return error
        max_results, error = self._validate_int(
            parameters,
            "max_results",
            default=self.DEFAULT_MAX_RESULTS,
            minimum=1,
            maximum=self.MAX_RESULTS_LIMIT,
        )
        if error:
            return error
        region, error = self._validate_optional_string(parameters, "region", default=self.DEFAULT_REGION)
        if error:
            return error
        safesearch, error = self._validate_choice(
            parameters,
            "safesearch",
            default=self.DEFAULT_SAFESEARCH,
            allowed=self.ALLOWED_SAFESEARCH,
        )
        if error:
            return error
        backend, error = self._validate_optional_string(parameters, "backend", default=self.DEFAULT_BACKEND)
        if error:
            return error
        include_domains_input, error = self._validate_string_array(parameters, "include_domains")
        if error:
            return error
        include_domains = _parse_domain_list(include_domains_input)
        if include_domains is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    "Invalid parameter `include_domains`: expected a domain array such as "
                    "['docs.python.org', 'python.org']."
                ),
            )
        exclude_domains_input, error = self._validate_string_array(parameters, "exclude_domains")
        if error:
            return error
        exclude_domains = _parse_domain_list(exclude_domains_input)
        if exclude_domains is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=(
                    "Invalid parameter `exclude_domains`: expected a domain array such as "
                    "['example.com']."
                ),
            )
        timeout_seconds, error = self._validate_int(
            parameters,
            "timeout_seconds",
            default=self.DEFAULT_TIMEOUT_SECONDS,
            minimum=1,
            maximum=self.MAX_TIMEOUT_SECONDS,
        )
        if error:
            return error
        timelimit, error = self._validate_optional_string(parameters, "timelimit", default=None, allow_blank=True)
        if error:
            return error

        domain_filters_active = bool(include_domains or exclude_domains)
        effective_query = _build_effective_search_query(
            query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        backend_max_results = max_results
        if domain_filters_active:
            backend_max_results = min(
                self.MAX_BACKEND_RESULTS,
                max(max_results + 8, max_results * self.FILTER_OVERSAMPLE_FACTOR),
            )

        try:
            raw_results = self.search_backend.search_text(
                query=effective_query,
                max_results=backend_max_results,
                region=region or self.DEFAULT_REGION,
                safesearch=safesearch or self.DEFAULT_SAFESEARCH,
                timelimit=timelimit,
                timeout_seconds=timeout_seconds,
                backend=backend or self.DEFAULT_BACKEND,
            )
        except ImportError:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="ddgs package is not installed. Install it with: pip install ddgs",
            )
        except Exception as exc:
            if self._is_no_results_error(exc):
                return ToolResponse.success(
                    text=f"No results found for: {query}",
                data={
                    "query": query,
                    "query_original": query,
                    "query_effective": effective_query,
                    "results": [],
                    "count": 0,
                    "include_domains": include_domains,
                    "exclude_domains": exclude_domains,
                    "filtered_out_count": 0,
                    "provider": "duckduckgo-ddgs",
                },
            )
            return ToolResponse.error(
                code=self._map_search_error(exc),
                message=f"Web search failed: {exc}",
            )

        cleaned = self._clean_results(raw_results)
        filtered, filtered_out_count = self._filter_results_by_domain(
            cleaned,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        filtered = filtered[:max_results]
        if not cleaned:
            return ToolResponse.success(
                text=f"No results found for: {query}",
                data={
                    "query": query,
                    "query_original": query,
                    "query_effective": effective_query,
                    "results": [],
                    "count": 0,
                    "include_domains": include_domains,
                    "exclude_domains": exclude_domains,
                    "filtered_out_count": 0,
                    "provider": "duckduckgo-ddgs",
                },
            )
        if not filtered:
            return ToolResponse.success(
                text=f"No results found for: {query}",
                data={
                    "query": query,
                    "query_original": query,
                    "query_effective": effective_query,
                    "results": [],
                    "count": 0,
                    "include_domains": include_domains,
                    "exclude_domains": exclude_domains,
                    "filtered_out_count": filtered_out_count,
                    "provider": "duckduckgo-ddgs",
                },
            )

        lines = [
            f"Search results for: {query}",
            f"Returned {len(filtered)} result(s).",
            "",
        ]
        if effective_query != query:
            lines.append(f"Effective search query: {effective_query}")
        if include_domains:
            lines.append(f"Included domains: {', '.join(include_domains)}")
        if exclude_domains:
            lines.append(f"Excluded domains: {', '.join(exclude_domains)}")
        if domain_filters_active:
            lines.append(f"Filtered out {filtered_out_count} result(s) by domain policy.")
        if len(lines) > 3:
            lines.append("")
        for item in filtered:
            lines.append(f"[{item['rank']}] {item['title']}")
            lines.append(f"    URL: {item['url']}")
            if item.get("published"):
                lines.append(f"    Published: {item['published']}")
            if item.get("source"):
                lines.append(f"    Source: {item['source']}")
            if item.get("snippet"):
                lines.append(f"    Snippet: {item['snippet']}")
            lines.append("")

        return self._build_text_response(
            tool_name="websearch",
            output_text="\n".join(lines),
            metadata={
                "query": query,
                "query_original": query,
                "query_effective": effective_query,
                "provider": "duckduckgo-ddgs",
                "max_results": max_results,
                "backend_max_results": backend_max_results,
                "region": region,
                "safesearch": safesearch,
                "timelimit": timelimit,
                "backend": backend,
                "timeout_seconds": timeout_seconds,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "filtered_out_count": filtered_out_count,
                "count": len(filtered),
            },
            data={
                "query": query,
                "query_original": query,
                "query_effective": effective_query,
                "provider": "duckduckgo-ddgs",
                "count": len(filtered),
                "results": filtered,
                "max_results": max_results,
                "backend_max_results": backend_max_results,
                "region": region,
                "safesearch": safesearch,
                "timelimit": timelimit,
                "backend": backend,
                "timeout_seconds": timeout_seconds,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "filtered_out_count": filtered_out_count,
            },
        )

    def _filter_results_by_domain(
        self,
        results: List[Dict[str, Any]],
        *,
        include_domains: List[str],
        exclude_domains: List[str],
    ) -> Tuple[List[Dict[str, Any]], int]:
        filtered: List[Dict[str, Any]] = []
        filtered_out_count = 0
        for item in results:
            host = _extract_host(item.get("url", ""))
            include_match = not include_domains or any(
                _host_matches_domain(host, domain) for domain in include_domains
            )
            exclude_match = any(_host_matches_domain(host, domain) for domain in exclude_domains)
            if not include_match or exclude_match:
                filtered_out_count += 1
                continue

            cleaned_item = dict(item)
            cleaned_item["rank"] = len(filtered) + 1
            if host:
                cleaned_item["host"] = host
            filtered.append(cleaned_item)
        return filtered, filtered_out_count

    def _clean_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            url = str(item.get("href") or item.get("link") or item.get("url") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen_urls:
                continue
            seen_urls.add(url)

            title = _normalize_whitespace(str(item.get("title") or "No title")) or "No title"
            snippet = _normalize_whitespace(str(item.get("body") or item.get("snippet") or ""))
            source = _normalize_whitespace(str(item.get("source") or ""))
            published = _normalize_whitespace(str(item.get("date") or item.get("published") or ""))

            cleaned_item: Dict[str, Any] = {
                "rank": len(cleaned) + 1,
                "title": title,
                "url": url,
                "snippet": _trim_display_text(snippet, self.MAX_SNIPPET_CHARS) if snippet else "",
            }
            if source:
                cleaned_item["source"] = source
            if published:
                cleaned_item["published"] = published
            cleaned.append(cleaned_item)
        return cleaned

    @staticmethod
    def _is_no_results_error(exc: Exception) -> bool:
        return type(exc).__name__ == "DDGSException" and "no results" in str(exc).lower()

    @staticmethod
    def _map_search_error(exc: Exception) -> str:
        name = type(exc).__name__
        text = str(exc).lower()
        if isinstance(exc, TimeoutError) or name == "TimeoutException" or "timed out" in text:
            return ToolErrorCode.TIMEOUT
        if name == "RatelimitException":
            return ToolErrorCode.RATE_LIMIT
        return ToolErrorCode.NETWORK_ERROR


class WebFetchTool(_WebToolBase):
    """Fetch and extract readable text content from a web URL."""

    DEFAULT_MAX_LENGTH = 120_000
    MAX_LENGTH_LIMIT = 1_000_000
    DEFAULT_TIMEOUT_SECONDS = 20
    MAX_TIMEOUT_SECONDS = 120
    MAX_RESPONSE_BYTES = 5 * 1024 * 1024
    DEFAULT_FORMAT = "text"
    DEFAULT_EXTRACT_MODE = "auto"
    ALLOWED_FORMATS = {"text", "html"}
    ALLOWED_EXTRACT_MODES = {"auto", "main", "full"}
    OUTPUT_MAX_LINES = 1200
    OUTPUT_MAX_BYTES = 48_000
    ENABLE_ENV = "WEB_FETCH_ENABLED"
    TEXT_MIME_PREFIXES = ("text/",)
    TEXT_MIME_TYPES = {
        "",
        "application/json",
        "application/ld+json",
        "application/javascript",
        "application/x-javascript",
        "application/xml",
        "application/xhtml+xml",
        "text/xml",
    }

    @classmethod
    def is_enabled_by_default(cls) -> bool:
        return _env_enabled("WEB_TOOLS_ENABLED", True) and _env_enabled(cls.ENABLE_ENV, True)

    def __init__(
        self,
        name: str = "WebFetch",
        project_root: str = ".",
        enabled: Optional[bool] = None,
        output_dir: Optional[str] = None,
        output_max_lines: int = OUTPUT_MAX_LINES,
        output_max_bytes: int = OUTPUT_MAX_BYTES,
        truncate_direction: str = "head_tail",
        session_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(
            name=name,
            description=(
                "Fetch a web page and extract readable content by URL. "
                "Use this after WebSearch to inspect a specific page. "
                "Supports text extraction from HTML and bounded downloads."
            ),
            project_root=project_root,
            enabled=self.is_enabled_by_default() if enabled is None else bool(enabled),
            output_dir=output_dir,
            output_max_lines=output_max_lines,
            output_max_bytes=output_max_bytes,
            truncate_direction=truncate_direction,
            hint_message="Use Read on the saved output file for the full fetched content when needed.",
        )
        self._session_factory = session_factory
        self._session: Any = None

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="The URL to fetch (must include http:// or https://).",
                required=True,
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Return format: text or html (default: text).",
                required=False,
                default=self.DEFAULT_FORMAT,
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description=(
                    f"Maximum extracted text characters before output truncation "
                    f"(default: {self.DEFAULT_MAX_LENGTH}, max: {self.MAX_LENGTH_LIMIT})."
                ),
                required=False,
                default=self.DEFAULT_MAX_LENGTH,
            ),
            ToolParameter(
                name="timeout_seconds",
                type="integer",
                description=(
                    f"Request timeout in seconds (default: {self.DEFAULT_TIMEOUT_SECONDS}, "
                    f"max: {self.MAX_TIMEOUT_SECONDS})."
                ),
                required=False,
                default=self.DEFAULT_TIMEOUT_SECONDS,
            ),
            ToolParameter(
                name="extract_mode",
                type="string",
                description="HTML text extraction mode: auto, main, or full (default: auto). Only used when format=text for HTML pages.",
                required=False,
                default=self.DEFAULT_EXTRACT_MODE,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        disabled = self._check_enabled("webfetch")
        if disabled:
            return disabled

        url, error = self._validate_required_string(parameters, "url")
        if error:
            return error
        format_name, error = self._validate_choice(
            parameters,
            "format",
            default=self.DEFAULT_FORMAT,
            allowed=self.ALLOWED_FORMATS,
        )
        if error:
            return error
        max_length, error = self._validate_int(
            parameters,
            "max_length",
            default=self.DEFAULT_MAX_LENGTH,
            minimum=256,
            maximum=self.MAX_LENGTH_LIMIT,
        )
        if error:
            return error
        timeout_seconds, error = self._validate_int(
            parameters,
            "timeout_seconds",
            default=self.DEFAULT_TIMEOUT_SECONDS,
            minimum=1,
            maximum=self.MAX_TIMEOUT_SECONDS,
        )
        if error:
            return error
        extract_mode, error = self._validate_choice(
            parameters,
            "extract_mode",
            default=self.DEFAULT_EXTRACT_MODE,
            allowed=self.ALLOWED_EXTRACT_MODES,
        )
        if error:
            return error

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"Invalid parameter `url`: expected http:// or https:// URL, got {url!r}.",
            )

        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="requests package is not installed.",
            )

        response = None
        raw_bytes = b""
        try:
            try:
                session = self._get_session(requests)
                response = session.get(
                    url,
                    headers=self._build_headers(format_name or self.DEFAULT_FORMAT),
                    timeout=timeout_seconds,
                    allow_redirects=True,
                    stream=True,
                )
                response.raise_for_status()
                raw_bytes = self._read_response_bytes(response)
            except requests.exceptions.Timeout as exc:  # type: ignore[attr-defined]
                return ToolResponse.error(code=ToolErrorCode.TIMEOUT, message=f"Request timed out: {exc}")
            except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                code = ToolErrorCode.RATE_LIMIT if status_code == 429 else ToolErrorCode.API_ERROR
                return ToolResponse.error(code=code, message=f"Request failed: {exc}")
            except requests.exceptions.ConnectionError as exc:  # type: ignore[attr-defined]
                return ToolResponse.error(code=ToolErrorCode.NETWORK_ERROR, message=f"Request failed: {exc}")
            except requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
                return ToolResponse.error(code=ToolErrorCode.NETWORK_ERROR, message=f"Request failed: {exc}")
            except _ResponseTooLargeError as exc:
                return ToolResponse.error(code=ToolErrorCode.API_ERROR, message=str(exc))

            content_type = response.headers.get("Content-Type", "") if response is not None else ""
            mime = content_type.split(";", 1)[0].strip().lower()
            decoded = self._decode_response_body(response, raw_bytes)
            page_title = _extract_html_title(decoded) if mime in {"text/html", "application/xhtml+xml"} else ""
            extraction_info: Dict[str, Any] = {
                "strategy": "plain_text",
                "used_main_candidate": False,
            }

            if format_name == "html":
                extracted = decoded
                extraction_info = {
                    "strategy": "raw_html",
                    "used_main_candidate": False,
                }
            elif mime in {"text/html", "application/xhtml+xml"}:
                extracted, extraction_info = _extract_html_text(
                    decoded,
                    mode=extract_mode or self.DEFAULT_EXTRACT_MODE,
                )
            elif mime.startswith(self.TEXT_MIME_PREFIXES) or mime in self.TEXT_MIME_TYPES:
                extracted = decoded
            else:
                effective_url = getattr(response, "url", url) if response is not None else url
                return ToolResponse.partial(
                    text=(
                        f"Unsupported content type: {content_type or '[unknown]'}. "
                        "Only text-like responses are supported by WebFetch."
                    ),
                    data={
                        "requested_url": url,
                        "url": effective_url,
                        "effective_url": effective_url,
                        "content_type": content_type,
                        "format": format_name,
                        "extract_mode": extract_mode,
                        "response_bytes": len(raw_bytes),
                        "truncated": False,
                    },
                )

            content_clipped = len(extracted) > max_length
            if content_clipped:
                extracted = extracted[:max_length]

            effective_url = getattr(response, "url", url) if response is not None else url
            lines = [
                f"Content from: {effective_url}",
            ]
            if effective_url != url:
                lines.append(f"Requested URL: {url}")
            if page_title:
                lines.append(f"Page title: {page_title}")
            lines.extend(
                [
                    f"Content type: {content_type or '[unknown]'}",
                    f"Format: {format_name}",
                    f"Extract mode: {extract_mode}",
                    f"Extraction strategy: {extraction_info.get('strategy', 'plain_text')}",
                    f"Downloaded bytes: {len(raw_bytes)}",
                    f"Extracted length: {len(extracted)} chars" + (" (clipped)" if content_clipped else ""),
                    "",
                    extracted,
                ]
            )

            return self._build_text_response(
                tool_name="webfetch",
                output_text="\n".join(lines),
                metadata={
                    "requested_url": url,
                    "effective_url": effective_url,
                    "content_type": content_type,
                    "format": format_name,
                    "extract_mode": extract_mode,
                    "extraction_strategy": extraction_info.get("strategy", "plain_text"),
                    "used_main_candidate": bool(extraction_info.get("used_main_candidate")),
                    "page_title": page_title,
                    "response_bytes": len(raw_bytes),
                    "max_length": max_length,
                    "timeout_seconds": timeout_seconds,
                    "content_clipped": content_clipped,
                    "candidate_text_len": extraction_info.get("candidate_text_len"),
                    "full_text_len": extraction_info.get("full_text_len"),
                },
                data={
                    "requested_url": url,
                    "url": effective_url,
                    "effective_url": effective_url,
                    "content_type": content_type,
                    "format": format_name,
                    "extract_mode": extract_mode,
                    "extraction_strategy": extraction_info.get("strategy", "plain_text"),
                    "used_main_candidate": bool(extraction_info.get("used_main_candidate")),
                    "page_title": page_title,
                    "response_bytes": len(raw_bytes),
                    "length": len(extracted),
                    "content_clipped": content_clipped,
                    "max_length": max_length,
                    "timeout_seconds": timeout_seconds,
                    "candidate_text_len": extraction_info.get("candidate_text_len"),
                    "full_text_len": extraction_info.get("full_text_len"),
                },
                force_partial=content_clipped,
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    def _get_session(self, requests_module: Any) -> Any:
        if self._session is None:
            self._session = self._session_factory() if self._session_factory else requests_module.Session()
        return self._session

    def _build_headers(self, format_name: str) -> Dict[str, str]:
        if format_name == "html":
            accept = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
        else:
            accept = "text/html,application/xhtml+xml,text/plain,application/json,application/xml;q=0.9,*/*;q=0.1"
        return {
            "User-Agent": (
                "Mozilla/5.0 (compatible; WhaleCode/1.0; "
                "+https://github.com/hello-agents)"
            ),
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        }

    def _read_response_bytes(self, response: Any) -> bytes:
        content_length = response.headers.get("Content-Length") or response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.MAX_RESPONSE_BYTES:
                    raise _ResponseTooLargeError(
                        f"Response too large (exceeds {self.MAX_RESPONSE_BYTES} byte limit)."
                    )
            except ValueError:
                pass

        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.MAX_RESPONSE_BYTES:
                raise _ResponseTooLargeError(
                    f"Response too large (exceeds {self.MAX_RESPONSE_BYTES} byte limit)."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode_response_body(response: Any, payload: bytes) -> str:
        encoding = getattr(response, "encoding", None) or "utf-8"
        try:
            return payload.decode(encoding, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")
