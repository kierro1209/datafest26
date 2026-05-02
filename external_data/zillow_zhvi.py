"""
Official Zillow Research ZHVI: ZIP-level All Homes (smoothed, seasonally adjusted).
https://www.zillow.com/research/data/
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

ZHVI_ZIP_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)


def _latest_month_column(columns: list[str]) -> str | None:
    date_cols = [c for c in columns if len(c) == 10 and c[4] == "-" and c[7] == "-"]
    if not date_cols:
        return None
    return sorted(date_cols)[-1]


def load_zhvi_for_zips(
    zips: set[str],
    cache_path: str | Path | None = None,
    chunk_rows: int = 50_000,
) -> pd.DataFrame:
    """
    Return DataFrame with columns ZIP_STR, ZHVI_HOME_VALUE (latest month).
    Streams the national file and keeps only needed ZIPs (memory-safe).
    """
    zips_norm = {str(z).strip().zfill(5) for z in zips if str(z).strip() and str(z) != "nan"}
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        raw = cache.read_bytes()
    else:
        r = requests.get(ZHVI_ZIP_URL, timeout=600)
        r.raise_for_status()
        raw = r.content
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)

    head = pd.read_csv(io.BytesIO(raw), nrows=0)
    latest = _latest_month_column(list(head.columns))
    if not latest:
        raise RuntimeError("Could not find a YYYY-MM-DD column in ZHVI file.")

    usecols = [c for c in ("RegionID", "SizeRank", "RegionName", "StateName") if c in head.columns]
    usecols.append(latest)

    out: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        io.BytesIO(raw),
        usecols=lambda c: c in set(usecols),
        chunksize=chunk_rows,
        low_memory=False,
    ):
        chunk["RegionName"] = chunk["RegionName"].astype(str).str.replace(".0", "", regex=False)
        chunk["ZIP_STR"] = chunk["RegionName"].str.zfill(5)
        sub = chunk[chunk["ZIP_STR"].isin(zips_norm)].copy()
        if len(sub):
            sub = sub.rename(columns={latest: "ZHVI_HOME_VALUE"})
            keep = ["ZIP_STR", "ZHVI_HOME_VALUE"]
            extra = [c for c in ("RegionID", "StateName") if c in sub.columns]
            out.append(sub[keep + extra])

    if not out:
        return pd.DataFrame(columns=["ZIP_STR", "ZHVI_HOME_VALUE"])

    df = pd.concat(out, ignore_index=True)
    df = df.drop_duplicates(subset=["ZIP_STR"], keep="first")
    return df

