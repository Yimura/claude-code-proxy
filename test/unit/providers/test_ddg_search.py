"""Tests for the DuckDuckGo HTML search client and page fetcher."""

import httpx
import pytest

from claude_code_proxy.providers.ddg_search import (
    DuckDuckGoSearch,
    extract_results,
    fetch_page_text,
    strip_html,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip",
}

SAMPLE = """
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.idealo.de%2Fa57&amp;rut=abc">Samsung Galaxy A57 ab 319,90 &euro;</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/x">Bereits ab 319,90 &euro; Jetzt kaufen</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://example.com/direct">Direct link</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/y">Second snippet</a>
</div>
"""


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_extract_results_parses_title_url_snippet_and_unwraps_redirect():
    results = extract_results(SAMPLE)

    assert results[0]["title"] == "Samsung Galaxy A57 ab 319,90 €"
    assert results[0]["url"] == "https://www.idealo.de/a57"
    assert "319,90" in results[0]["snippet"]
    assert results[1]["url"] == "https://example.com/direct"


def test_extract_results_empty_page_is_empty_list():
    assert extract_results("<html><body>anomaly</body></html>") == []


def test_strip_html_removes_tags_and_entities():
    assert strip_html("<b>Hello &amp;   world</b>\n<span>!</span>") == "Hello & world !"


@pytest.mark.asyncio
async def test_search_formats_results_as_text():
    def handler(request):
        assert "galaxy+a57" in str(request.url) or "galaxy%20a57" in str(request.url)
        assert request.headers["accept-language"] == "en-US,en;q=0.5"
        return httpx.Response(200, text=SAMPLE)

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.search("galaxy a57")

    assert "Samsung Galaxy A57 ab 319,90 €" in text
    assert "https://www.idealo.de/a57" in text
    assert "Bereits ab 319,90 €" in text


@pytest.mark.asyncio
async def test_search_reports_anomaly_challenge_as_error_text():
    def handler(request):
        return httpx.Response(200, text="<html>anomaly challenge</html>")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.search("anything")

    assert "Web search failed" in text


@pytest.mark.asyncio
async def test_search_handles_network_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.search("anything")

    assert "Web search failed: boom" in text


@pytest.mark.asyncio
async def test_fetch_page_text_strips_html_and_truncates():
    def handler(request):
        return httpx.Response(200, text="<html><body>" + "word " * 5000 + "</body></html>")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://example.com/page")

    assert len(text) <= 3000
    assert text.startswith("word word")


@pytest.mark.asyncio
async def test_fetch_page_skips_non_html_content_type():
    def handler(request):
        return httpx.Response(200, content=b"\xff\xd8jpegdata", headers={"content-type": "image/jpeg"})

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://example.com/img.jpg")

    assert text == ""


@pytest.mark.asyncio
async def test_fetch_page_error_returns_empty_string():
    def handler(request):
        raise httpx.ConnectTimeout("timeout")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://example.com/slow")

    assert text == ""


@pytest.mark.asyncio
async def test_fetch_page_sends_full_browser_headers_and_site_referer():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, text="<html><body>content</body></html>")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    await search.fetch_page("https://www.coolblue.de/product")

    assert seen["referer"] == "https://www.coolblue.de/"
    assert seen["sec-fetch-mode"] == "navigate"
    assert seen["sec-fetch-dest"] == "document"
    assert seen["upgrade-insecure-requests"] == "1"


@pytest.mark.asyncio
async def test_fetch_page_retries_with_alternate_user_agent_on_block():
    agents = []

    def handler(request):
        agents.append(request.headers["user-agent"])
        if len(agents) == 1:
            return httpx.Response(403, text="forbidden")
        return httpx.Response(200, text="<html><body>content</body></html>")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://www.coolblue.de/product")

    assert text.startswith("content")
    assert len(agents) == 2
    assert agents[0] != agents[1]


@pytest.mark.asyncio
async def test_fetch_page_tries_www_variant_on_connect_error():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "example.com":
            raise httpx.ConnectError("[Errno -2] Name or service not known")
        return httpx.Response(200, text="<html><body>content</body></html>")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://example.com/product")

    assert text.startswith("content")
    assert hosts == ["example.com", "www.example.com"]


@pytest.mark.asyncio
async def test_fetch_page_falls_back_to_jina_on_consent_wall():
    def handler(request):
        if request.url.host == "r.jina.ai":
            assert str(request.url).startswith("https://r.jina.ai/https://tweakers.net/")
            return httpx.Response(200, text="real page body about prices")
        return httpx.Response(
            200, text="<html><body>We use cookies. Accept cookies to continue.</body></html>"
        )

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://tweakers.net/pricewatch")

    assert "real page body" in text


@pytest.mark.asyncio
async def test_fetch_page_falls_back_to_jina_when_direct_fetch_fails():
    def handler(request):
        if request.url.host == "r.jina.ai":
            return httpx.Response(200, text="fallback body")
        return httpx.Response(403, text="forbidden")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://www.coolblue.de/product")

    assert text == "fallback body"


@pytest.mark.asyncio
async def test_fetch_page_returns_empty_when_jina_also_fails():
    def handler(request):
        raise httpx.ConnectError("down")

    search = DuckDuckGoSearch(client=_client(handler), headers=HEADERS)
    text = await search.fetch_page("https://example.com/gone")

    assert text == ""


def test_is_consent_wall_flags_short_cookie_walls_only():
    from claude_code_proxy.providers.ddg_search import is_consent_wall

    assert is_consent_wall("Please accept cookies and consent to continue")
    assert not is_consent_wall(
        "Article about cookies in browsers. " * 200
    )
