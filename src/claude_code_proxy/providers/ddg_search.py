"""Free DuckDuckGo HTML search and page fetching.

DuckDuckGo's non-JavaScript endpoint (https://html.duckduckgo.com/html/)
serves result links and snippets without an API key. It challenges requests
whose headers do not look like a browser, so the client sends a full
browser header set and treats the challenge page ("anomaly") as a failure.
"""

import logging
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

_RESULT_LINK = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S
)
_RESULT_SNIPPET = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>', re.S
)
_TAG = re.compile(r"<[^>]+>")
_MAX_RESULTS = 10
_MAX_PAGE_CHARS = 3000
_TIMEOUT_SECONDS = 20.0

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
}


def strip_html(fragment: str) -> str:
    """Remove tags and decode entities, collapsing whitespace."""
    return " ".join(unescape(_TAG.sub("", fragment)).split())


def _unwrap(href: str) -> str:
    """Follow DuckDuckGo's redirect wrapper to the real target URL."""
    if href.startswith("//"):
        href = "https:" + href
    try:
        query = parse_qs(urlparse(href).query)
    except ValueError:
        return href
    return query.get("uddg", [href])[0]


def extract_results(page: str) -> list[dict[str, str]]:
    """Extract (title, url, snippet) results from a DDG HTML page."""
    links = _RESULT_LINK.findall(page)
    snippets = _RESULT_SNIPPET.findall(page)
    results = []
    for (href, title), snippet in zip(links, snippets + [""] * len(links)):
        clean_title = strip_html(title)
        if not clean_title:
            continue
        results.append({
            "title": clean_title,
            "url": _unwrap(unescape(href)),
            "snippet": strip_html(snippet),
        })
    return results[:_MAX_RESULTS]


def _format_results(results: list[dict[str, str]]) -> str:
    entries = []
    for result in results:
        entries.append(
            f"- {result['title']}\n  {result['url']}\n  {result['snippet']}"
        )
    return "\n".join(entries)


def fetch_page_text(response: httpx.Response) -> str:
    """Return visible page text, empty for non-HTML or oversized bodies."""
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return ""
    text = strip_html(response.text)
    return text[:_MAX_PAGE_CHARS]


class DuckDuckGoSearch:
    """Searches via DuckDuckGo's HTML endpoint; fetches pages on demand.

    Failures return error text rather than raise, so a search outage
    degrades to an empty tool result instead of a failed request.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._headers = headers or DEFAULT_HEADERS

    async def search(self, query: str) -> str:
        try:
            if self._client is not None:
                response = await self._client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers=self._headers,
                )
            else:
                async with httpx.AsyncClient(
                    timeout=_TIMEOUT_SECONDS, follow_redirects=True
                ) as client:
                    response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query},
                        headers=self._headers,
                    )
        except httpx.RequestError as error:
            logger.warning("duckduckgo search request error: %s", error)
            return f"Web search failed: {error}"
        results = extract_results(response.text)
        if not results and "anomaly" in response.text:
            return (
                "Web search failed: DuckDuckGo returned a bot challenge. "
                "Retry the search."
            )
        return _format_results(results)

    async def fetch_page(self, url: str) -> str:
        try:
            if self._client is not None:
                response = await self._client.get(url, headers=self._headers)
            else:
                async with httpx.AsyncClient(
                    timeout=_TIMEOUT_SECONDS, follow_redirects=True
                ) as client:
                    response = await client.get(url, headers=self._headers)
        except httpx.RequestError as error:
            logger.warning("page fetch error for %s: %s", url, error)
            return ""
        return fetch_page_text(response)
