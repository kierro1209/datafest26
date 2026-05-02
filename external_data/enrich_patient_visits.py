"""
End-to-end patient visit enrichment: Zillow ZHVI (research CSV) + external public sources.
DartMonkey-style Zillow listing parse lives in zillow_jsonld.py (optional HTML path).

Conda (use your ``datafest`` env)::

    conda activate datafest
    cd /path/to/datafest26/external_data
    python -m pip install -r requirements.txt
    python prepare_input.py --sample-n 2000
    python enrich_patient_visits.py --skip-overpass   # fast test (no OSM Overpass)
    python enrich_patient_visits.py                   # full run (Overpass ~1s per unique lat/lon)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import bg_zcta
import census_acs_broadband
import geo_census
import overpass_amenities
import zillow_zhvi

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"

NEW_NUMERIC_FEATURES = [
    "ZHVI_HOME_VALUE",
    "VIOLENT_CRIME_RATE",
    "PROPERTY_CRIME_RATE",
    "EVICTION_RATE",
    "EVICTION_FILINGS",
    "EJ_INDEX",
    "PM25",
    "TOXICS_SCORE",
    "AMENITY_COUNT_1MI",
    "MAX_DOWNLOAD_SPEED",
    "NUM_BROADBAND_PROVIDERS",
    "BROADBAND_INACCESSIBILITY_SCORE",
    "ADI_NATIONAL_RANK",
    "OPPORTUNITY_MOBILITY_SCORE",
    "SVI_RPL_THEMES",
]


def _zfill_zip(s: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x) or str(x).strip() == "":
            return pd.NA
        t = str(x).strip()
        if t.endswith(".0"):
            t = t[:-2]
        t = re.sub(r"\D", "", t)
        if not t:
            return pd.NA
        return t.zfill(5)[-5:]

    return s.map(one)


def _ensure_tract_county(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BLOCK_GROUP_GEOID" in out.columns:
        bg = out["BLOCK_GROUP_GEOID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        if "TRACT_GEOID" not in out.columns:
            out["TRACT_GEOID"] = bg.str[:11]
        else:
            out["TRACT_GEOID"] = out["TRACT_GEOID"].fillna(bg.str[:11])
        if "COUNTY_FIPS" not in out.columns:
            out["COUNTY_FIPS"] = bg.str[:5]
        else:
            out["COUNTY_FIPS"] = out["COUNTY_FIPS"].fillna(bg.str[:5])
    if "CENSUS_TRACT" in out.columns:
        ct = out["CENSUS_TRACT"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        if "TRACT_GEOID" not in out.columns:
            out["TRACT_GEOID"] = ct
        else:
            out["TRACT_GEOID"] = out["TRACT_GEOID"].fillna(ct)
    if "TRACT_GEOID" in out.columns:
        out["TRACT_GEOID"] = out["TRACT_GEOID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        out["TRACT_GEOID"] = out["TRACT_GEOID"].str.zfill(11).str[-11:]
    if "COUNTY_FIPS" in out.columns:
        out["COUNTY_FIPS"] = out["COUNTY_FIPS"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        out["COUNTY_FIPS"] = out["COUNTY_FIPS"].str.zfill(5).str[-5:]
    return out


def _normalize_lat_lon_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low = {c.lower(): c for c in out.columns}
    if "LAT" not in out.columns and "latitude" in low:
        out = out.rename(columns={low["latitude"]: "LAT"})
    if "LON" not in out.columns and "longitude" in low:
        out = out.rename(columns={low["longitude"]: "LON"})
    if "LAT" not in out.columns and "lat" in low:
        out = out.rename(columns={low["lat"]: "LAT"})
    if "LON" not in out.columns and "lon" in low:
        out = out.rename(columns={low["lon"]: "LON"})
    return out


def _fill_geographies_from_coordinates(
    df: pd.DataFrame,
    sleep: float,
    *,
    prefer_latlon: bool = False,
) -> pd.DataFrame:
    """
    Use Census /geographies/coordinates for (LAT, LON) → tract, block group, county.
    Deduplicates coordinates to limit API calls. When prefer_latlon is True, overwrites
    existing tract/BG/county from coordinates (ZIP is refreshed from BG crosswalk later).
    """
    out = df.copy()
    if "LAT" not in out.columns or "LON" not in out.columns:
        return out
    lat_ok = out["LAT"].notna() & out["LON"].notna()
    if not lat_ok.any():
        return out

    if prefer_latlon:
        need = lat_ok
    else:
        tr = out.get("TRACT_GEOID", pd.Series(pd.NA, index=out.index))
        bg = out.get("BLOCK_GROUP_GEOID", pd.Series(pd.NA, index=out.index))
        zp = out.get("ZIP_CODE", pd.Series(pd.NA, index=out.index))
        need = lat_ok & (tr.isna() | bg.isna() | zp.isna())

    if not need.any():
        return out

    sub = out.loc[need, ["LAT", "LON"]].astype(float)
    uniq = sub.drop_duplicates()
    coord_key = list(zip(uniq["LAT"], uniq["LON"]))
    geo_rows = geo_census.enrich_coords_batch(coord_key, sleep_s=sleep)
    lookup: dict[tuple[float, float], dict] = {}
    for (la, lo), g in zip(coord_key, geo_rows):
        lookup[(round(la, 6), round(lo, 6))] = g

    for row_ix in out.index[need]:
        la = float(out.at[row_ix, "LAT"])
        lo = float(out.at[row_ix, "LON"])
        g = lookup.get((round(la, 6), round(lo, 6)), {})
        if g.get("CENSUS_TRACT"):
            out.at[row_ix, "TRACT_GEOID"] = str(g["CENSUS_TRACT"]).replace(".0", "")
        if g.get("BLOCK_GROUP"):
            out.at[row_ix, "BLOCK_GROUP_GEOID"] = str(g["BLOCK_GROUP"]).replace(".0", "")
        if g.get("COUNTY_FIPS"):
            out.at[row_ix, "COUNTY_FIPS"] = str(g["COUNTY_FIPS"]).replace(".0", "").zfill(5)[-5:]
    return out


def _zip_from_block_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ZIP_CODE from BG→ZCTA where ZIP is still missing."""
    out = df.copy()
    if "BLOCK_GROUP_GEOID" not in out.columns:
        return out
    missing = out["ZIP_CODE"].isna() & out["BLOCK_GROUP_GEOID"].notna()
    if not missing.any():
        return out
    bgs = set(out.loc[missing, "BLOCK_GROUP_GEOID"].astype(str).str.strip().str[:12])
    zmap = bg_zcta.zip_for_block_groups(bgs, bg_zcta.DEFAULT_CENSUS_DATA_DIR)
    if "ZIP_STR" in out.columns:
        out = out.drop(columns=["ZIP_STR"])
    out = out.merge(zmap, on="BLOCK_GROUP_GEOID", how="left")
    out["ZIP_CODE"] = out["ZIP_CODE"].fillna(out["ZIP_STR"])
    out = out.drop(columns=["ZIP_STR"], errors="ignore")
    return out


def _merge_zhvi(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    zips = set(df["ZIP_CODE"].dropna().astype(str).unique())
    zh = zillow_zhvi.load_zhvi_for_zips(zips, cache_path=cache)
    if zh.empty:
        df["ZHVI_HOME_VALUE"] = np.nan
        return df
    out = df.merge(zh[["ZIP_STR", "ZHVI_HOME_VALUE"]], left_on="ZIP_CODE", right_on="ZIP_STR", how="left")
    out = out.drop(columns=["ZIP_STR"], errors="ignore")
    return out


def _merge_optional_county_crime(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        df["VIOLENT_CRIME_RATE"] = np.nan
        df["PROPERTY_CRIME_RATE"] = np.nan
        return df
    cr = pd.read_csv(path, dtype={"COUNTY_FIPS": str})
    cr["COUNTY_FIPS"] = cr["COUNTY_FIPS"].str.zfill(5)
    out = df.merge(cr, on="COUNTY_FIPS", how="left", suffixes=("", "_crime"))
    return out


def _merge_optional_eviction(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        df["EVICTION_RATE"] = np.nan
        df["EVICTION_FILINGS"] = np.nan
        return df
    ev = pd.read_csv(path, dtype=str)
    zip_col = next((c for c in ev.columns if "zip" in c.lower() or "zcta" in c.lower()), ev.columns[0])
    ev["ZIP_CODE"] = _zfill_zip(ev[zip_col])
    rate_col = next((c for c in ev.columns if "rate" in c.lower()), None)
    fil_col = next((c for c in ev.columns if "fil" in c.lower()), None)
    use = ev[["ZIP_CODE"]].copy()
    use["EVICTION_RATE"] = pd.to_numeric(ev[rate_col], errors="coerce") if rate_col else np.nan
    use["EVICTION_FILINGS"] = pd.to_numeric(ev[fil_col], errors="coerce") if fil_col else np.nan
    use = use.drop_duplicates(subset=["ZIP_CODE"])
    return df.merge(use, on="ZIP_CODE", how="left")


def _merge_optional_ejscreen(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    for c in ("EJ_INDEX", "PM25", "TOXICS_SCORE"):
        df[c] = np.nan
    if not path.exists():
        return df
    ej = pd.read_csv(path, low_memory=False, dtype=str, nrows=500_000)
    id_col = next(
        (c for c in ej.columns if c.upper() in ("ID", "TRACT", "GEOID", "FIPS", "STCNTR")),
        ej.columns[0],
    )
    ej["TRACT_GEOID"] = ej[id_col].astype(str).str.replace(r"\.0$", "", regex=False)
    ej["TRACT_GEOID"] = ej["TRACT_GEOID"].str.zfill(11).str[-11:]
    sub = ej[ej["TRACT_GEOID"].isin(df["TRACT_GEOID"].dropna().unique())].copy()
    if sub.empty:
        return df

    def pick(pat: str) -> str | None:
        for c in sub.columns:
            if pat.lower() in c.lower():
                return c
        return None

    pm = pick("pm25") or pick("pm2")
    ej_idx = pick("ej") and pick("pct")  # weak
    tox = pick("toxic") or pick("cancer") or pick("rsei")

    agg = sub.groupby("TRACT_GEOID", as_index=False).first()
    use = agg[["TRACT_GEOID"]].copy()
    if pm:
        use["PM25"] = pd.to_numeric(agg[pm], errors="coerce")
    else:
        use["PM25"] = np.nan
    if tox:
        use["TOXICS_SCORE"] = pd.to_numeric(agg[tox], errors="coerce")
    else:
        use["TOXICS_SCORE"] = np.nan
    ej_pick = pick("supplemental") or pick("demog") or pick("ej_index")
    if ej_pick:
        use["EJ_INDEX"] = pd.to_numeric(agg[ej_pick], errors="coerce")
    else:
        use["EJ_INDEX"] = np.nan
    return df.drop(columns=["EJ_INDEX", "PM25", "TOXICS_SCORE"], errors="ignore").merge(
        use, on="TRACT_GEOID", how="left"
    )


def _merge_svi(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    df["SVI_RPL_THEMES"] = np.nan
    tracts = df["TRACT_GEOID"].dropna().astype(str).str.zfill(11).unique()
    if len(tracts) == 0:
        return df
    url = "https://www.atsdr.cdc.gov/placehealth/data/svi2020/SVI2020_US.csv"
    import requests

    if not cache.exists():
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)

    head = pd.read_csv(cache, nrows=0)
    fips_col = "FIPS" if "FIPS" in head.columns else next(
        (c for c in head.columns if c.upper().endswith("FIPS")), head.columns[0]
    )
    rpl = next((c for c in head.columns if "RPL_THEMES" == c.upper()), None)
    if not rpl:
        rpl = next((c for c in head.columns if "RPL_THEMES" in c.upper()), None)
    if not rpl:
        rpl = next((c for c in head.columns if c.upper().startswith("RPL")), None)
    usecols = [fips_col] + ([rpl] if rpl else [])
    chunks = []
    tract_set = set(tracts)
    for chunk in pd.read_csv(
        cache,
        usecols=lambda c: c in set(usecols),
        chunksize=80_000,
        dtype=str,
        low_memory=False,
    ):
        chunk[fips_col] = chunk[fips_col].astype(str).str.replace(r"\.0$", "", regex=False)
        chunk[fips_col] = chunk[fips_col].str.replace(r"\D", "", regex=True).str.zfill(11).str[-11:]
        sub = chunk[chunk[fips_col].isin(tract_set)]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        return df
    svi = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=[fips_col])
    ren = {fips_col: "TRACT_GEOID"}
    if rpl:
        ren[rpl] = "SVI_RPL_THEMES"
    svi = svi.rename(columns=ren)
    if "SVI_RPL_THEMES" in svi.columns:
        svi["SVI_RPL_THEMES"] = pd.to_numeric(svi["SVI_RPL_THEMES"], errors="coerce")
    else:
        svi["SVI_RPL_THEMES"] = np.nan
    return df.merge(svi, on="TRACT_GEOID", how="left")


def _merge_opportunity_county(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    df["OPPORTUNITY_MOBILITY_SCORE"] = np.nan
    import requests

    url = "https://opportunityinsights.org/wp-content/uploads/2018/10/cty_covariates.csv"
    if not cache.exists():
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return df
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
    cty = pd.read_csv(cache, low_memory=False)
    # county id often 'cty' or first column
    id_col = next((c for c in cty.columns if "cty" in c.lower()), cty.columns[0])
    cty["COUNTY_FIPS"] = cty[id_col].astype(str).str.replace(r"\.0$", "", regex=False).str.zfill(5)
    mob_col = next(
        (c for c in cty.columns if re.search(r"mob|kfr|kir|pooled|mobility", c, re.I)),
        None,
    )
    if not mob_col:
        return df
    use = cty[["COUNTY_FIPS", mob_col]].drop_duplicates(subset=["COUNTY_FIPS"])
    use = use.rename(columns={mob_col: "OPPORTUNITY_MOBILITY_SCORE"})
    use["OPPORTUNITY_MOBILITY_SCORE"] = pd.to_numeric(use["OPPORTUNITY_MOBILITY_SCORE"], errors="coerce")
    return df.merge(use, on="COUNTY_FIPS", how="left")


def _add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}_MISSING"] = out[c].isna().astype(int)
    return out


def _fill_numeric(out: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in out.columns:
            continue
        med = out[c].median(skipna=True)
        fill = 0 if c in ("AMENITY_COUNT_1MI", "NUM_BROADBAND_PROVIDERS") else med
        if pd.isna(fill):
            fill = 0.0
        out[c] = out[c].fillna(fill)
    return out


def run(
    input_csv: Path,
    output_csv: Path,
    geocode_sleep: float = 0.1,
    overpass_sleep: float = 1.0,
    skip_overpass: bool = False,
    prefer_latlon_geographies: bool = False,
) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Normalize common column names
    if "ZIP_CODE" not in df.columns and "ZIP" in df.columns:
        df = df.rename(columns={"ZIP": "ZIP_CODE"})
    if "VISIT_DATE" not in df.columns and "ADMIT_DATE" in df.columns:
        df["VISIT_DATE"] = df["ADMIT_DATE"]

    df = _normalize_lat_lon_columns(df)

    if "ZIP_CODE" in df.columns:
        df["ZIP_CODE"] = _zfill_zip(df["ZIP_CODE"])
    else:
        df["ZIP_CODE"] = pd.Series(pd.NA, index=df.index, dtype=object)

    for c in ("LAT", "LON"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "BLOCK_GROUP_GEOID" not in df.columns and "CensusBlockGroupFipsCode" in df.columns:
        df = df.rename(columns={"CensusBlockGroupFipsCode": "BLOCK_GROUP_GEOID"})

    # ZIP from known block group (no coordinates required)
    df = _zip_from_block_groups(df)

    df = _ensure_tract_county(df)
    # Primary assumption: LAT/LON are available — derive / repair FIPS and ZIP via Census + crosswalk
    df = _fill_geographies_from_coordinates(
        df,
        sleep=geocode_sleep,
        prefer_latlon=prefer_latlon_geographies,
    )
    df = _ensure_tract_county(df)
    df = _zip_from_block_groups(df)
    df["ZIP_CODE"] = _zfill_zip(df["ZIP_CODE"])

    df = _merge_zhvi(df, RAW / "zillow_zhvi_zip_all_homes.csv")

    df = _merge_optional_county_crime(df, RAW / "crime_by_county_fips.csv")
    df = _merge_optional_eviction(df, RAW / "eviction_by_zip.csv")
    df = _merge_optional_ejscreen(df, RAW / "ejscreen_tract_subset.csv")

    # ADI: national Neighborhood Atlas file is restricted; column reserved
    df["ADI_NATIONAL_RANK"] = np.nan

    df = _merge_svi(df, RAW / "SVI2020_US.csv")
    df = _merge_opportunity_county(df, RAW / "cty_covariates.csv")

    # ACS broadband proxy (tract)
    pairs = []
    if "TRACT_GEOID" in df.columns:
        for t in df["TRACT_GEOID"].dropna().astype(str):
            t = t.zfill(11)
            if len(t) >= 5:
                pairs.append((t[:2], t[2:5]))
    bb = census_acs_broadband.broadband_for_counties(pairs)
    if not bb.empty:
        bb["TRACT_GEOID"] = bb["TRACT_GEOID"].astype(str).str.zfill(11).str[-11:]
        bb = census_acs_broadband.add_broadband_scores(bb)
        df = df.merge(bb, on="TRACT_GEOID", how="left")
    else:
        df["ACS_HH_TOTAL"] = np.nan
        df["ACS_HH_BROADBAND"] = np.nan
        df["BROADBAND_INACCESSIBILITY_SCORE"] = np.nan
        df["MAX_DOWNLOAD_SPEED"] = np.nan
        df["NUM_BROADBAND_PROVIDERS"] = np.nan
        df["BROADBAND_DATA_SOURCE"] = "missing"

    if not skip_overpass and "LAT" in df.columns and "LON" in df.columns:
        coords = (
            df[["LAT", "LON"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        cmap = overpass_amenities.map_unique_coords(coords, sleep_s=overpass_sleep)

        def lookup(row):
            if pd.isna(row["LAT"]) or pd.isna(row["LON"]):
                return 0
            k = (round(float(row["LAT"]), 5), round(float(row["LON"]), 5))
            return cmap.get(k, 0)

        df["AMENITY_COUNT_1MI"] = df.apply(lookup, axis=1)
    else:
        df["AMENITY_COUNT_1MI"] = 0

    flag_cols = [c for c in NEW_NUMERIC_FEATURES if c in df.columns]
    df = _add_missing_flags(df, flag_cols)
    df = _fill_numeric(df, flag_cols)

    df.to_csv(output_csv, index=False)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("shape:", df.shape)
    added = [c for c in NEW_NUMERIC_FEATURES if c in df.columns]
    print("new feature columns:", ", ".join(added))
    for c in added:
        s = pd.to_numeric(df[c], errors="coerce")
        print(
            f"  {c}: mean={s.mean():.4g} min={s.min():.4g} max={s.max():.4g} "
            f"na%={100 * s.isna().mean():.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich patient_visits.csv with external data.")
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=BASE / "patient_visits.csv",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BASE / "enriched_patient_visits.csv",
    )
    ap.add_argument("--geocode-sleep", type=float, default=0.1)
    ap.add_argument("--overpass-sleep", type=float, default=1.0)
    ap.add_argument("--skip-overpass", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}. Run prepare_input.py first.")
        raise SystemExit(1)

    df = run(
        args.input,
        args.output,
        geocode_sleep=args.geocode_sleep,
        overpass_sleep=args.overpass_sleep,
        skip_overpass=args.skip_overpass,
    )
    _print_summary(df)
    print("\nWrote:", args.output)
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
