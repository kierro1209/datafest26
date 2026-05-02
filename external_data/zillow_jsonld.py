"""
DartMonkey-style Zillow enrichment: parse search-result HTML for JSON-LD
(SingleFamilyResidence / Residence) — same pattern as saved SRP dumps (input.txt).
Does not use Selenium; pass HTML strings or files from your own scrape.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup


def _loads_json_ld(text: str) -> Any | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Rare: multiple objects or trailing commas
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def iter_zillow_jsonld_properties(html: str) -> Iterator[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", type="application/ld+json"):
        data = _loads_json_ld(script.string or "")
        if not data:
            continue
        if isinstance(data, list):
            items = data
        else:
            items = [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            t = item.get("@type")
            if t in (
                "SingleFamilyResidence",
                "Residence",
                "Apartment",
                "House",
                "Place",
            ) or (isinstance(t, list) and any(x in str(t) for x in ("Residence", "Place"))):
                yield item


def property_to_row(item: dict[str, Any]) -> dict[str, Any]:
    addr = item.get("address") or {}
    geo = item.get("geo") or {}
    if isinstance(geo, list) and geo:
        geo = geo[0] if isinstance(geo[0], dict) else {}
    lat = geo.get("latitude")
    lon = geo.get("longitude")
    row = {
        "ZILLOW_JSONLD_STREET": addr.get("streetAddress"),
        "ZILLOW_JSONLD_CITY": addr.get("addressLocality"),
        "ZILLOW_JSONLD_STATE": addr.get("addressRegion"),
        "ZILLOW_JSONLD_ZIP": addr.get("postalCode"),
        "ZILLOW_JSONLD_LAT": float(lat) if lat is not None else None,
        "ZILLOW_JSONLD_LON": float(lon) if lon is not None else None,
        "ZILLOW_JSONLD_URL": item.get("url"),
    }
    return row


def parse_zillow_srp_html(html: str) -> list[dict[str, Any]]:
    return [property_to_row(p) for p in iter_zillow_jsonld_properties(html)]


def parse_zillow_srp_file(path: str | Path) -> list[dict[str, Any]]:
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_zillow_srp_html(html)

