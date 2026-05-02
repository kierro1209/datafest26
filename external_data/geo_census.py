"""Census Geocoder: coordinates -> tract / block group FIPS."""
from __future__ import annotations

import time
from typing import Any

import requests

CENSUS_GEO_URL = (
    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
)


def geographies_from_lat_lon(
    lat: float,
    lon: float,
    benchmark: str = "Public_AR_Current",
    vintage: str = "Current_Current",
    sleep_s: float = 0.05,
) -> dict[str, Any]:
    """Return raw Census geographies dict for one point."""
    time.sleep(sleep_s)
    params = {
        "x": lon,
        "y": lat,
        "benchmark": benchmark,
        "vintage": vintage,
        "format": "json",
    }
    r = requests.get(CENSUS_GEO_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("result", {})


def parse_geographies(geo_blob: dict[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "CENSUS_TRACT": None,
        "BLOCK_GROUP": None,
        "COUNTY_FIPS": None,
        "STATE_FIPS": None,
    }
    geos = geo_blob.get("geographies") or {}
    tracts = geos.get("Census Tracts") or []
    if tracts:
        t = tracts[0]
        out["CENSUS_TRACT"] = t.get("GEOID") or t.get("OID")
        out["STATE_FIPS"] = t.get("STATE")
        out["COUNTY_FIPS"] = (t.get("STATE") or "") + (t.get("COUNTY") or "")
    bgs = geos.get("Census Block Groups") or []
    if bgs:
        bg = bgs[0]
        out["BLOCK_GROUP"] = bg.get("GEOID") or bg.get("OID")
    return out


def enrich_coords_batch(
    pairs: list[tuple[float, float]],
    sleep_s: float = 0.08,
) -> list[dict[str, str | None]]:
    results = []
    for lat, lon in pairs:
        try:
            blob = geographies_from_lat_lon(lat, lon, sleep_s=sleep_s)
            g = parse_geographies(blob)
            g["LAT"] = lat
            g["LON"] = lon
            results.append(g)
        except Exception:
            results.append(
                {
                    "CENSUS_TRACT": None,
                    "BLOCK_GROUP": None,
                    "COUNTY_FIPS": None,
                    "STATE_FIPS": None,
                    "LAT": lat,
                    "LON": lon,
                }
            )
    return results

