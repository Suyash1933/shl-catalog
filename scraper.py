"""
Scraper for the SHL Product Catalog — Individual Test Solutions.
Fetches all pages and optionally enriches with detail page descriptions.
"""

import json
import logging
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com/products/product-catalog/"
DETAIL_BASE = "https://www.shl.com"
OUTPUT_PATH = Path(__file__).parent / "shl_catalog.json"


def _get_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session

# Test type mapping from table header icons/classes
TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competency",
    "D": "Development",
    "E": "Assessment Experience",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulation",
}


def fetch_page(start: int = 0, session: requests.Session | None = None) -> str:
    """Fetch a single catalog page."""
    params = {"start": start, "type": 1}  # type=1 = Individual Test Solutions
    s = session or _get_session()
    resp = s.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_catalog_page(html: str) -> list[dict]:
    """Parse assessments from a catalog listing page."""
    soup = BeautifulSoup(html, "html.parser")
    items = []

    # Find the results table
    table = soup.find("table")
    if not table:
        return items

    rows = table.find_all("tr")
    for row in rows[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # First cell: name + link
        link_tag = cells[0].find("a")
        if not link_tag:
            continue
        name = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        if href and not href.startswith("http"):
            url = DETAIL_BASE + href
        else:
            url = href

        # Second cell: remote testing (check for icon/checkmark)
        remote = bool(cells[1].find("span", class_=re.compile(r"catalogue__circle--fill")))

        # Third cell: adaptive/IRT
        adaptive = bool(cells[2].find("span", class_=re.compile(r"catalogue__circle--fill")))

        # Fourth cell onwards: test types
        test_types = []
        for cell in cells[3:]:
            if cell.find("span", class_=re.compile(r"catalogue__circle--fill")):
                # Determine type from column header
                idx = cells.index(cell)
                # Map column index to type code
                col_types = ["A", "B", "C", "D", "E", "K", "P", "S"]
                if idx - 3 < len(col_types):
                    test_types.append(col_types[idx - 3])

        items.append({
            "name": name,
            "url": url,
            "test_type": test_types,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "description": "",
            "duration": "",
        })

    return items


def fetch_detail_page(url: str, session: requests.Session | None = None) -> dict:
    """Fetch a detail page and extract description + duration."""
    s = session or _get_session()
    try:
        resp = s.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to find description
        desc = ""
        desc_section = soup.find("div", class_=re.compile(r"product-catalogue-description|product-detail"))
        if desc_section:
            desc = desc_section.get_text(strip=True)
        else:
            # Fallback: look for main content paragraphs
            main = soup.find("main") or soup.find("article") or soup
            paragraphs = main.find_all("p")
            desc = " ".join(p.get_text(strip=True) for p in paragraphs[:3])

        # Try to find duration
        duration = ""
        duration_match = soup.find(string=re.compile(r"duration|minutes|mins", re.I))
        if duration_match:
            parent = duration_match.parent
            if parent:
                duration = parent.get_text(strip=True)

        return {"description": desc[:500], "duration": duration[:100]}
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return {"description": "", "duration": ""}


def scrape_all_pages() -> list[dict]:
    """Scrape all catalog pages."""
    all_items = []
    seen_urls = set()
    page = 0
    consecutive_errors = 0
    session = _get_session()

    while True:
        start = page * 12
        logger.info(f"Fetching page {page + 1} (start={start})...")
        try:
            html = fetch_page(start, session=session)
            items = parse_catalog_page(html)
            if not items:
                logger.info(f"  No items on page {page + 1}, stopping.")
                break
            # Deduplicate by URL
            new_items = []
            for item in items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    new_items.append(item)
            all_items.extend(new_items)
            logger.info(f"  Found {len(new_items)} new items (total: {len(all_items)})")
            page += 1
            consecutive_errors = 0
            time.sleep(0.5)  # Be polite
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"  Error on page {page + 1}: {e}")
            if consecutive_errors >= 3:
                logger.error("  Too many consecutive errors, stopping.")
                break
            time.sleep(2)  # Wait before retrying

    return all_items


def enrich_with_details(catalog: list[dict], max_workers: int = 5) -> list[dict]:
    """Fetch detail pages to add descriptions."""
    logger.info(f"Enriching {len(catalog)} items with detail page data...")
    session = _get_session()

    def _enrich(item):
        details = fetch_detail_page(item["url"], session=session)
        item["description"] = details["description"]
        item["duration"] = details["duration"]
        return item

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_enrich, item): i for i, item in enumerate(catalog)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 20 == 0:
                logger.info(f"  Enriched {done}/{len(catalog)}")

    return catalog


def main():
    catalog = scrape_all_pages()
    print(f"\nTotal assessments scraped: {len(catalog)}")

    if catalog:
        # Optionally enrich with detail descriptions
        catalog = enrich_with_details(catalog)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"Saved to {OUTPUT_PATH}")
    else:
        print("No items scraped! The HTML structure may have changed.")
        print("You may need to inspect the page manually and adjust the parser.")


if __name__ == "__main__":
    main()
