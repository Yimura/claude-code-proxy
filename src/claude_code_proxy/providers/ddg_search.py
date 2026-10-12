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
_RETRY_DELAY_SECONDS = 0.25
_BLOCKED_STATUS = frozenset({401, 403, 429})
_CONSENT_KEYWORDS = ("cookie", "consent", "to continue", "zustimmen", "akzeptieren")
_CONSENT_MAX_CHARS = 600

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
}

# Alternate identity for one retry when a site blocks the first request.
_ALTERNATE_HEADERS = {
    **DEFAULT_HEADERS,
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
}

_READER_FALLBACK = "https://r.jina.ai/"


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


def _fetch_headers(url: str, base: dict[str, str]) -> dict[str, str]:
    """Browser-like headers for one page fetch, referring to the site itself."""
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
    return {
        **base,
        "Referer": origin,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def _www_variant(url: str) -> str:
    """Prepend www. to a bare-domain URL (or drop it if already present)."""
    parts = urlparse(url)
    host = parts.netloc
    if host.startswith("www."):
        new_host = host[4:]
    else:
        new_host = "www." + host
    return parts._replace(netloc=new_host).geturl()


def is_consent_wall(text: str) -> bool:
    """Whether a fetched page is only a cookie/consent interstitial.

    Only short bodies dominated by consent keywords count; real articles
    mentioning cookies stay untouched.
    """
    if len(text) > _CONSENT_MAX_CHARS:
        return False
    lowered = text.lower()
    return sum(keyword in lowered for keyword in _CONSENT_KEYWORDS) >= 2


def _usable(text: str) -> bool:
    return bool(text) and not is_consent_wall(text)


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
        """Fetch a page, escalating through fallbacks until one yields text.

        Tries the URL with browser headers, then an alternate user agent on
        a block status, then the www-variant of the host on a connect
        error, and finally the r.jina.ai reader when everything above
        fails or only a consent wall comes back.
        """
        for attempt in (url, _www_variant(url)):
            for headers in (self._headers, _ALTERNATE_HEADERS):
                try:
                    response = await self._get(attempt, headers)
                except httpx.RequestError as error:
                    logger.info("page fetch error for %s: %s", attempt, error)
                    break  # wrong host shape; try the www-variant next
                text = fetch_page_text(response)
                if response.status_code in _BLOCKED_STATUS:
                    continue  # blocked; retry with the alternate identity
                if _usable(text):
                    return text
                break  # reachable but unusable; try the www-variant next
        return await self._fetch_via_reader(url)

    async def _fetch_via_reader(self, url: str) -> str:
        try:
            response = await self._get(_READER_FALLBACK + url, DEFAULT_HEADERS)
        except httpx.RequestError as error:
            logger.warning("reader fallback error for %s: %s", url, error)
            return ""
        return fetch_page_text(response)

    async def _get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, headers=_fetch_headers(url, headers))
        async with httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS, follow_redirects=True
        ) as client:
            return await client.get(url, headers=_fetch_headers(url, headers))
