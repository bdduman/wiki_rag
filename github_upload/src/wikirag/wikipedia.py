from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from urllib.parse import quote

import requests


API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "LocalWikipediaRAGAssistant/0.1 (student homework; local use)",
}


@dataclass(frozen=True)
class WikiPage:
    title: str
    text: str
    source_url: str


def fetch_wikipedia_page(title: str, timeout: int = 30, retries: int = 4) -> WikiPage:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    response = None
    for attempt in range(retries + 1):
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 5 + attempt * 5
        sleep(wait_seconds)

    if response is None:
        raise ValueError(f"No Wikipedia response for {title}")
    response.raise_for_status()
    payload = response.json()

    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No Wikipedia page found for {title}")

    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page is missing: {title}")

    resolved_title = page.get("title", title)
    text = (page.get("extract") or "").strip()
    if not text:
        raise ValueError(f"Wikipedia page has no extract text: {resolved_title}")

    url_title = quote(resolved_title.replace(" ", "_"))
    return WikiPage(
        title=resolved_title,
        text=text,
        source_url=f"https://en.wikipedia.org/wiki/{url_title}",
    )
