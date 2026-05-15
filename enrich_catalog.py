"""
Enrich the SHL catalog with descriptions and durations from detail pages.
"""

import json
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CATALOG_PATH = Path(__file__).parent / "shl_catalog.json"
OUTPUT_PATH = CATALOG_PATH  # overwrite


def fetch_detail(url: str) -> dict:
    """Fetch a detail page and extract description + duration + job levels."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract description - try multiple selectors
        desc = ""
        # Try the product description section
        for selector in [
            "div.product-catalogue-description",
            "div.product-detail",
            "div.product-catalogue__content",
            "section.product-detail",
            "div.richtext",
        ]:
            el = soup.select_one(selector)
            if el:
                desc = el.get_text(separator=" ", strip=True)
                break

        # Fallback: look for paragraphs in main content
        if not desc:
            main = soup.find("main") or soup.find("article") or soup
            paragraphs = main.find_all("p")
            desc_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 20 and not text.startswith("©"):
                    desc_parts.append(text)
                if len(desc_parts) >= 4:
                    break
            desc = " ".join(desc_parts)

        # Extract duration
        duration = ""
        # Look for duration in structured data
        for text in soup.stripped_strings:
            if re.search(r'\d+\s*min', text, re.I):
                duration = text.strip()
                break

        # Extract job levels
        job_levels = ""
        for heading in soup.find_all(["h3", "h4", "strong", "b"]):
            if "job level" in heading.get_text(strip=True).lower():
                sibling = heading.find_next_sibling()
                if sibling:
                    job_levels = sibling.get_text(strip=True)
                break

        # Also look for job levels in list items near "Job Level" text
        if not job_levels:
            for el in soup.find_all(string=re.compile(r"Job Level", re.I)):
                parent = el.parent
                if parent:
                    container = parent.parent
                    if container:
                        items = container.find_all("li")
                        if items:
                            job_levels = ", ".join(i.get_text(strip=True) for i in items)

        return {
            "description": desc[:600].strip(),
            "duration": duration[:100].strip(),
            "job_levels": job_levels[:200].strip(),
        }
    except Exception as e:
        return {"description": "", "duration": "", "job_levels": ""}


def enrich_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Enriching {len(catalog)} items...")

    # Only enrich items that don't already have descriptions
    to_enrich = [(i, item) for i, item in enumerate(catalog) if not item.get("description")]
    print(f"  {len(to_enrich)} items need enrichment")

    if not to_enrich:
        print("All items already have descriptions!")
        return

    done = 0
    errors = 0

    def process(args):
        idx, item = args
        details = fetch_detail(item["url"])
        return idx, details

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process, args): args for args in to_enrich}
        for future in as_completed(futures):
            idx, details = future.result()
            catalog[idx]["description"] = details["description"]
            catalog[idx]["duration"] = details["duration"]
            if details.get("job_levels"):
                catalog[idx]["job_levels"] = details["job_levels"]
            done += 1
            if not details["description"]:
                errors += 1
            if done % 25 == 0:
                print(f"  {done}/{len(to_enrich)} done ({errors} without description)")

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nDone! Enriched {done} items ({errors} without description)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    enrich_catalog()
