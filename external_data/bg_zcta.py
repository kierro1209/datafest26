"""
2020 block group → ZCTA (ZIP) using **local** Census relationship files only (no HTTP).

Place manually downloaded files under ``external_data/census_rel2020_blkgrp/``:

1. **Required** — 2020 BG ↔ 2010 BG (pipe-delimited ``.txt``), from:
   https://www2.census.gov/geo/docs/maps-data/data/rel2020/blkgrp/

   - ``tab20_blkgrp20_blkgrp10_natl.txt`` (national), and/or
   - ``tab20_blkgrp20_blkgrp10_st01.txt`` … ``st78.txt`` (by state; st codes skip 03,07 per Census).

2. **Optional (for ZIP_CODE)** — tract ↔ 2020 ZCTA (same pipe format), from:
   https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/

   - ``tab20_zcta520_tract20_natl.txt``

   Tract GEOID is the first 11 characters of ``GEOID_BLKGRP_20``. If the ZCTA file is
   absent, ``ZIP_STR`` is left missing (NaN).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CENSUS_DATA_DIR = Path(__file__).resolve().parent / "census_rel2020_blkgrp"

BLKGRP_GLOB = "tab20_blkgrp20_blkgrp10_*.txt"
ZCTA_TRACT_FILE = "tab20_zcta520_tract20_natl.txt"


def _read_pipe_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig", low_memory=False)


def _load_blkgrp_bridge_tables(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob(BLKGRP_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No {BLKGRP_GLOB} files under {data_dir}. "
            "Download tab20_blkgrp20_blkgrp10_natl.txt and/or tab20_blkgrp20_blkgrp10_st*.txt "
            "from https://www2.census.gov/geo/docs/maps-data/data/rel2020/blkgrp/ "
            "and place them in that folder."
        )
    parts = [_read_pipe_txt(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    if "GEOID_BLKGRP_20" not in df.columns:
        raise ValueError(f"Expected column GEOID_BLKGRP_20 in {paths[0].name}, got {list(df.columns)}")
    df["BLOCK_GROUP_GEOID"] = df["GEOID_BLKGRP_20"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["BLOCK_GROUP_GEOID"], keep="first")
    return df


def _load_tract_to_zcta(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / ZCTA_TRACT_FILE
    if not path.exists():
        return None
    df = _read_pipe_txt(path)
    if "GEOID_TRACT_20" not in df.columns or "GEOID_ZCTA5_20" not in df.columns:
        raise ValueError(
            f"Expected GEOID_TRACT_20 and GEOID_ZCTA5_20 in {path.name}, got {list(df.columns)}"
        )
    out = df[["GEOID_TRACT_20", "GEOID_ZCTA5_20"]].copy()
    out["TRACT_GEOID"] = out["GEOID_TRACT_20"].astype(str).str.strip().str.zfill(11).str[-11:]
    out["ZIP_STR"] = out["GEOID_ZCTA5_20"].astype(str).str.strip().str.zfill(5).str[-5:]
    out = out[out["ZIP_STR"].str.len() == 5]
    out = out.drop_duplicates(subset=["TRACT_GEOID"], keep="first")
    return out[["TRACT_GEOID", "ZIP_STR"]]


def load_bg_zip_crosswalk(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Return columns ``BLOCK_GROUP_GEOID``, ``ZIP_STR`` (ZCTA5) where ZIP can be resolved.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_CENSUS_DATA_DIR
    bridge = _load_blkgrp_bridge_tables(root)
    bridge["TRACT_GEOID"] = bridge["BLOCK_GROUP_GEOID"].str[:11]

    zt = _load_tract_to_zcta(root)
    if zt is None or zt.empty:
        out = bridge[["BLOCK_GROUP_GEOID"]].copy()
        out["ZIP_STR"] = pd.NA
        return out

    merged = bridge.merge(zt, on="TRACT_GEOID", how="left")
    return merged[["BLOCK_GROUP_GEOID", "ZIP_STR"]].drop_duplicates(subset=["BLOCK_GROUP_GEOID"], keep="first")


def zip_for_block_groups(
    block_group_geoids: set[str],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Subset crosswalk to requested 2020 block group GEOIDs."""
    cw = load_bg_zip_crosswalk(data_dir)
    want = {str(g).strip()[:12] for g in block_group_geoids if str(g).strip().isdigit()}
    return cw[cw["BLOCK_GROUP_GEOID"].isin(want)]


def zip_for_block_groups_tract_zcta_only(
    block_group_geoids: set[str],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Resolve ZCTA using only ``tab20_zcta520_tract20_natl.txt``: treat the first 11
    characters of each 2020 block group GEOID as ``GEOID_TRACT_20``.

    Use when blkgrp bridge files (``tab20_blkgrp20_blkgrp10_*.txt``) are not present.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_CENSUS_DATA_DIR
    zt = _load_tract_to_zcta(root)
    if zt is None or zt.empty:
        raise FileNotFoundError(
            f"Tract→ZCTA file not found or empty: {root / ZCTA_TRACT_FILE}. "
            "Download tab20_zcta520_tract20_natl.txt from "
            "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
        )
    zt = zt.copy()
    zt["TRACT_GEOID"] = zt["TRACT_GEOID"].astype(str).str.zfill(11).str[-11:]
    lut = zt.set_index("TRACT_GEOID")["ZIP_STR"].to_dict()
    rows = []
    for g in block_group_geoids:
        raw = str(g).strip()
        digits = "".join(c for c in raw if c.isdigit())
        if len(digits) < 11:
            rows.append({"BLOCK_GROUP_GEOID": raw, "ZIP_STR": pd.NA})
            continue
        bg12 = digits.zfill(12)[-12:]
        tr = bg12[:11]
        zp = lut.get(tr, pd.NA)
        rows.append({"BLOCK_GROUP_GEOID": bg12, "ZIP_STR": zp})
    return pd.DataFrame(rows)


# Backwards-compatible name (no network; ``cache_dir`` is treated as data directory).
def load_bg_zcta_state(state_fips: str, cache_dir: Path) -> pd.DataFrame:
    _ = state_fips
    return load_bg_zip_crosswalk(cache_dir)
