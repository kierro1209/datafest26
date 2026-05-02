"""ACS 5-year B28002: broadband subscription rates by census tract (Census API)."""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

ACS5_2020 = "https://api.census.gov/data/2020/acs/acs5"


def fetch_broadband_tracts(
    state_fips: str,
    county_fips: str,
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    """Return tract-level totals and broadband count for one county."""
    st = str(state_fips).zfill(2)
    co = str(county_fips).zfill(3)
    params = {
        "get": "NAME,B28002_001E,B28002_004E",
        "for": "tract:*",
        "in": f"state:{st} county:{co}",
    }
    time.sleep(sleep_s)
    r = requests.get(ACS5_2020, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    if len(rows) < 2:
        return pd.DataFrame(columns=["TRACT_GEOID", "ACS_HH_TOTAL", "ACS_HH_BROADBAND"])
    hdr, *data = rows
    df = pd.DataFrame(data, columns=hdr)
    df["TRACT_GEOID"] = df["state"] + df["county"] + df["tract"]
    df["ACS_HH_TOTAL"] = pd.to_numeric(df["B28002_001E"], errors="coerce")
    df["ACS_HH_BROADBAND"] = pd.to_numeric(df["B28002_004E"], errors="coerce")
    return df[["TRACT_GEOID", "ACS_HH_TOTAL", "ACS_HH_BROADBAND"]]


def broadband_for_counties(pairs: Iterable[tuple[str, str]], sleep_s: float = 0.12) -> pd.DataFrame:
    out = []
    seen = set()
    for st, co in pairs:
        key = (str(st).zfill(2), str(co).zfill(3))
        if key in seen:
            continue
        seen.add(key)
        try:
            out.append(fetch_broadband_tracts(key[0], key[1], sleep_s=sleep_s))
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=["TRACT_GEOID", "ACS_HH_TOTAL", "ACS_HH_BROADBAND"])
    return pd.concat(out, ignore_index=True)


def add_broadband_scores(tract_df: pd.DataFrame) -> pd.DataFrame:
    df = tract_df.copy()
    tot = df["ACS_HH_TOTAL"].replace(0, pd.NA)
    share = (df["ACS_HH_BROADBAND"] / tot).clip(lower=0, upper=1)
    df["BROADBAND_INACCESSIBILITY_SCORE"] = ((1 - share) * 100).astype(float)
    df["MAX_DOWNLOAD_SPEED"] = pd.NA
    df["NUM_BROADBAND_PROVIDERS"] = pd.NA
    df["BROADBAND_DATA_SOURCE"] = "acs_b28002_proxy"
    return df

