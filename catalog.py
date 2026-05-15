"""
Catalog data layer — loads the scraped SHL catalog and provides lookup utilities.
"""

import json
from pathlib import Path
from typing import Optional

CATALOG_PATH = Path(__file__).parent / "shl_catalog.json"

# Test type code descriptions
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competency",
    "D": "Development",
    "E": "Assessment Experience",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulation",
}


def load_catalog() -> list[dict]:
    """Load the full catalog from JSON."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return catalog


def catalog_to_text(item: dict) -> str:
    """Convert a catalog item to a searchable text string for embedding."""
    types = ", ".join(
        TEST_TYPE_LABELS.get(t, t) for t in item.get("test_type", [])
    )

    parts = [item["name"]]

    if types:
        parts.append(f"Test types: {types}")

    desc = item.get("description", "")
    if desc:
        # Use first 400 chars of description for embedding
        parts.append(f"Description: {desc[:400]}")

    remote = "Remote testing available" if item.get("remote_testing") else ""
    if remote:
        parts.append(remote)

    adaptive = "Adaptive/IRT scoring" if item.get("adaptive_irt") else ""
    if adaptive:
        parts.append(adaptive)

    duration = item.get("duration", "")
    if duration:
        parts.append(f"Duration: {duration}")

    return " | ".join(p for p in parts if p)


def find_by_name(catalog: list[dict], name: str) -> Optional[dict]:
    """Fuzzy-ish lookup by name (case-insensitive substring)."""
    name_lower = name.lower()
    for item in catalog:
        if name_lower in item["name"].lower():
            return item
    return None
