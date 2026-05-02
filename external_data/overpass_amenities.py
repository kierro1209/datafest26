"""OSM Overpass: count grocery / supermarket / park / pharmacy / community centre within ~1 mile."""
from __future__ import annotations

import time
from typing import Iterable

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 1609  # ~1 mile


def _query(lat: float, lon: float) -> str:
    # amenity + shop tags used as in user spec
    return f"""
[out:json][timeout:60];
(
  node["shop"="supermarket"](around:{RADIUS_M},{lat},{lon});
  node["shop"="grocery"](around:{RADIUS_M},{lat},{lon});
  node["amenity"="pharmacy"](around:{RADIUS_M},{lat},{lon});
  node["leisure"="park"](around:{RADIUS_M},{lat},{lon});
  node["amenity"="community_centre"](around:{RADIUS_M},{lat},{lon});
  way["shop"="supermarket"](around:{RADIUS_M},{lat},{lon});
  way["shop"="grocery"](around:{RADIUS_M},{lat},{lon});
  way["amenity"="pharmacy"](around:{RADIUS_M},{lat},{lon});
  way["leisure"="park"](around:{RADIUS_M},{lat},{lon});
  way["amenity"="community_centre"](around:{RADIUS_M},{lat},{lon});
  relation["shop"="supermarket"](around:{RADIUS_M},{lat},{lon});
  relation["leisure"="park"](around:{RADIUS_M},{lat},{lon});
);
out count;
"""


def amenity_count_1mi(lat: float, lon: float, session: requests.Session, sleep_s: float = 1.0) -> int:
    time.sleep(sleep_s)
    r = session.post(
        OVERPASS_URL,
        data={"data": _query(lat, lon)},
        headers={"User-Agent": "datafest26-patient-enrichment/1.0"},
        timeout=90,
    )
    r.raise_for_status()
    j = r.json()
    # Overpass "out count" returns elements with "type":"count" in some versions;
    # fall back to len(elements)
    for el in j.get("elements", []):
        if el.get("type") == "count":
            t = el.get("tags") or {}
            if "total" in t:
                return int(float(t["total"]))
            n = int(float(t.get("nodes", 0)))
            w = int(float(t.get("ways", 0)))
            r = int(float(t.get("relations", 0)))
            return n + w + r
    return len(j.get("elements", []))


def map_unique_coords(
    coords: Iterable[tuple[float, float]],
    sleep_s: float = 1.0,
) -> dict[tuple[float, float], int]:
    sess = requests.Session()
    out: dict[tuple[float, float], int] = {}
    for lat, lon in coords:
        key = (round(lat, 5), round(lon, 5))
        if key in out:
            continue
        try:
            out[key] = amenity_count_1mi(lat, lon, sess, sleep_s=sleep_s)
        except Exception:
            out[key] = 0
    return out

