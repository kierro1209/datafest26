#!/usr/bin/env python3
"""
Augment ``patient_sequences_encounter_only_with_sdoh_status_and_fips.pt`` with
numeric external features repeated for every token position (shape ``[T, F]``).

Reads local datasets only (no HTTP). Default paths match this repo layout.

Adds **all** of: census block group (decomposed FIPS parts + ZCTA), **Zillow** ZHVI by ZIP,
**ADI** by block group, FCC **4G/5G** mobile broadband (H3 from lat/lon), and **FBI** NIBRS
county rates. ZIP is resolved from local Census files: full blkgrp bridge if present, else
tract→ZCTA only (``tab20_zcta520_tract20_natl.txt``).

Dependencies::

    pip install dbfread h3 tqdm torch pandas

Example::

    python final_patient_tokens.py \\
        --input patient_sequences_encounter_only_with_sdoh_status_and_fips.pt \\
        --output patient_sequences_with_external_features.pt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Repo root (parent of token_sequence_model/)
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external_data"
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
DEFAULT_DATASETS = EXT / "datasets"
DEFAULT_CENSUS_BLKGRP = EXT / "census_rel2020_blkgrp"
DEFAULT_4G_DBF = DEFAULT_DATASETS / "4G_broadband" / "bdc_20_4GLTE_mobile_broadband_h3_J25_29apr2026.dbf"
DEFAULT_5G_DBF = DEFAULT_DATASETS / "5G_broadband" / "bdc_20_5GNR_mobile_broadband_h3_J25_29apr2026.dbf"
DEFAULT_FBI_DIR = DEFAULT_DATASETS / "FBI_Crime"


def _latest_month_column(columns: list[str]) -> str | None:
    date_cols = [c for c in columns if len(c) == 10 and c[4] == "-" and c[7] == "-"]
    if not date_cols:
        return None
    return sorted(date_cols)[-1]


def _zfill_zip(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if pd.isna(x):
            return None
        t = str(int(round(float(x))))
    else:
        t = re.sub(r"\D", "", str(x).strip())
    if not t:
        return None
    return t.zfill(5)[-5:]


def load_zhvi_by_zip(zillow_csv: Path) -> dict[str, float]:
    if not zillow_csv.exists():
        return {}
    df = pd.read_csv(zillow_csv, low_memory=False)
    col = _latest_month_column([str(c) for c in df.columns])
    if not col or "RegionName" not in df.columns:
        return {}
    z = df[["RegionName", col]].copy()
    z["ZIP_STR"] = z["RegionName"].map(_zfill_zip)
    z[col] = pd.to_numeric(z[col], errors="coerce")
    z = z.dropna(subset=["ZIP_STR"])
    return z.drop_duplicates(subset=["ZIP_STR"], keep="first").set_index("ZIP_STR")[col].astype(float).to_dict()


def load_adi_by_block_group(adi_csv: Path) -> dict[str, tuple[float, float]]:
    if not adi_csv.exists():
        return {}
    df = pd.read_csv(adi_csv, dtype=str, low_memory=False)
    if "FIPS" not in df.columns:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        fips = re.sub(r"\D", "", str(row["FIPS"]))
        if len(fips) < 12:
            fips = fips.zfill(12)[-12:]
        else:
            fips = fips[-12:]
        nat = float(pd.to_numeric(row.get("ADI_NATRANK"), errors="coerce"))
        st = float(pd.to_numeric(row.get("ADI_STATERNK"), errors="coerce"))
        out[fips] = (nat, st)
    return out


def _bg_to_zip_map(bg_fips_set: set[str], census_dir: Path) -> tuple[dict[str, str | None], str]:
    """
    Returns (block_group_geoid -> ZCTA5 or None, resolution_mode).

    Tries full blkgrp20→blkgrp10 bridge + tract→ZCTA; if bridge files are missing,
    falls back to tract→ZCTA using BG[:11] only (needs ``tab20_zcta520_tract20_natl.txt``).
    """
    if not bg_fips_set:
        return {}, "none"
    from bg_zcta import zip_for_block_groups, zip_for_block_groups_tract_zcta_only

    try:
        sub = zip_for_block_groups(bg_fips_set, census_dir)
        mode = "blkgrp_bridge"
    except FileNotFoundError:
        sub = zip_for_block_groups_tract_zcta_only(bg_fips_set, census_dir)
        mode = "tract_zcta_only"

    if sub.empty:
        return {str(g): None for g in bg_fips_set}, mode
    m: dict[str, str | None] = {}
    for _, r in sub.iterrows():
        key = str(r["BLOCK_GROUP_GEOID"]).strip()
        zp = r.get("ZIP_STR")
        if zp is not None and not (isinstance(zp, float) and pd.isna(zp)):
            m[key] = str(zp).strip().zfill(5)[-5:]
        else:
            m[key] = None
    for g in bg_fips_set:
        gd = "".join(c for c in str(g) if c.isdigit())
        if gd:
            gd12 = gd.zfill(12)[-12:]
            m.setdefault(gd12, None)
        m.setdefault(str(g).strip(), None)
    return m, mode


def _parse_bg_fips_parts(bg: str | None) -> tuple[float, float, float, float]:
    """State (2), county (3), tract (6), block group digit — float32-safe integers."""
    nan = float("nan")
    if not bg:
        return nan, nan, nan, nan
    digits = "".join(c for c in str(bg) if c.isdigit())
    if len(digits) < 12:
        digits = digits.zfill(12)
    digits = digits[-12:]
    try:
        st = float(int(digits[0:2]))
        co = float(int(digits[2:5]))
        tr = float(int(digits[5:11]))
        gd = float(int(digits[11]))
        return st, co, tr, gd
    except ValueError:
        return nan, nan, nan, nan


def load_broadband_h3_agg(dbf_path: Path, desc: str) -> dict[str, tuple[float, float, int]]:
    """h3_res9_id -> (max mindown Mbps, max minup Mbps, number of layer rows in hex)."""
    if not dbf_path.exists():
        return {}
    try:
        from dbfread import DBF
    except ImportError as e:
        raise SystemExit(
            "Broadband .dbf requires dbfread. Install: pip install dbfread"
        ) from e

    agg: dict[str, dict[str, float | int]] = {}
    for row in tqdm(DBF(str(dbf_path), encoding="latin1"), desc=desc, unit="row"):
        h = str(row.get("h3_res9_id") or "").strip()
        if not h:
            continue
        try:
            md = float(row.get("mindown") or 0)
        except (TypeError, ValueError):
            md = 0.0
        try:
            mu = float(row.get("minup") or 0)
        except (TypeError, ValueError):
            mu = 0.0
        if h not in agg:
            agg[h] = {"md": md, "mu": mu, "n": 1}
        else:
            a = agg[h]
            a["md"] = max(float(a["md"]), md)
            a["mu"] = max(float(a["mu"]), mu)
            a["n"] = int(a["n"]) + 1
    return {k: (float(v["md"]), float(v["mu"]), int(v["n"])) for k, v in agg.items()}


def lat_lon_to_h3_res9(lat: float, lon: float) -> str | None:
    if lat != lat or lon != lon:
        return None
    try:
        import h3
    except ImportError as e:
        raise SystemExit("H3 lookup requires: pip install h3") from e
    try:
        return h3.latlng_to_cell(lat, lon, 9)
    except AttributeError:
        try:
            return h3.geo_to_h3(lat, lon, 9)
        except Exception:
            return None


def _load_state_abbr_to_fips(ref_state_csv: Path) -> dict[str, str]:
    df = pd.read_csv(ref_state_csv, dtype=str)
    out = {}
    for _, r in df.iterrows():
        ab = str(r.get("state_postal_abbr") or "").strip().upper()
        sf = str(r.get("state_fips_code") or "").strip()
        if not ab or not sf:
            continue
        out[ab] = re.sub(r"\D", "", sf).zfill(2)[-2:]
    return out


def ori_to_county_fips(ori: str, abbr2fips: dict[str, str]) -> str | None:
    if not ori or len(ori) < 8:
        return None
    ab = ori[:2].upper()
    if ab not in abbr2fips:
        return None
    co = ori[2:5]
    if not co.isdigit():
        return None
    return abbr2fips[ab] + co


def load_fbi_nibrs_county_rates(fbi_dir: Path) -> dict[str, dict[str, float]]:
    """
    5-digit county FIPS -> rates per 100k (agency population denominator = max pop per county).
    Uses NIBRS incidents + offenses (Person vs Property) for violent/property splits.
    """
    inc_path = fbi_dir / "NIBRS_incident.csv"
    off_path = fbi_dir / "NIBRS_OFFENSE.csv"
    otype_path = fbi_dir / "NIBRS_OFFENSE_TYPE.csv"
    ag_path = fbi_dir / "agencies.csv"
    ref_path = fbi_dir / "REF_STATE.csv"
    if not all(p.exists() for p in (inc_path, off_path, otype_path, ag_path, ref_path)):
        return {}

    abbr2fips = _load_state_abbr_to_fips(ref_path)

    ag = pd.read_csv(ag_path, dtype=str, low_memory=False)
    ag["data_year"] = pd.to_numeric(ag["data_year"], errors="coerce")
    ag = ag.sort_values("data_year").drop_duplicates("agency_id", keep="last")
    ag["population"] = pd.to_numeric(ag["population"], errors="coerce").fillna(0)

    inc = pd.read_csv(inc_path, usecols=["incident_id", "agency_id"], dtype=str, low_memory=False)
    inc_cf = inc.merge(ag[["agency_id", "ori", "population"]], on="agency_id", how="left")
    inc_cf["cfips"] = inc_cf["ori"].map(lambda o: ori_to_county_fips(str(o), abbr2fips))
    inc_cf = inc_cf.dropna(subset=["cfips"])

    total_by_c = inc_cf.groupby("cfips")["incident_id"].nunique()
    pop_by_c = inc_cf.groupby("cfips")["population"].max()

    off = pd.read_csv(off_path, usecols=["incident_id", "offense_code"], dtype=str, low_memory=False)
    ot = pd.read_csv(otype_path, usecols=["offense_code", "crime_against"], dtype=str, low_memory=False)
    mo = off.merge(ot, on="offense_code", how="left")
    viol_ids = set(mo.loc[mo["crime_against"].eq("Person"), "incident_id"])
    prop_ids = set(mo.loc[mo["crime_against"].eq("Property"), "incident_id"])

    viol_by_c = (
        inc_cf[inc_cf["incident_id"].isin(viol_ids)].groupby("cfips")["incident_id"].nunique()
    )
    prop_by_c = (
        inc_cf[inc_cf["incident_id"].isin(prop_ids)].groupby("cfips")["incident_id"].nunique()
    )

    out: dict[str, dict[str, float]] = {}
    for cf in total_by_c.index:
        pop = float(pop_by_c.get(cf, 0) or 0)
        if pop <= 0:
            pop = 1.0
        out[str(cf)] = {
            "fbi_nibrs_incidents_per_100k": 100000.0 * float(total_by_c[cf]) / pop,
            "fbi_nibrs_violent_incidents_per_100k": 100000.0 * float(viol_by_c.get(cf, 0)) / pop,
            "fbi_nibrs_property_incidents_per_100k": 100000.0 * float(prop_by_c.get(cf, 0)) / pop,
        }
    return out


def _seq_len(seq: dict) -> int:
    g = seq.get("gap_ids")
    if isinstance(g, list):
        return len(g)
    return 0


def _patient_bg(seq: dict) -> str | None:
    v = seq.get("patient_context_values") or {}
    f = v.get("patient_census_block_group_fips")
    if f is None or (isinstance(f, float) and pd.isna(f)):
        return None
    s = re.sub(r"\D", "", str(f).strip())
    if len(s) < 12:
        s = s.zfill(12)
    return s[-12:] if len(s) >= 12 else None


def _county_fips5(bg: str | None) -> str | None:
    if not bg or len(bg) < 5:
        return None
    return bg[:5]


def _patient_lat_lon(seq: dict) -> tuple[float, float]:
    v = seq.get("patient_context_values") or {}
    lat, lon = v.get("patient_lat"), v.get("patient_lon")
    try:
        la = float(lat) if lat is not None and not (isinstance(lat, float) and pd.isna(lat)) else float("nan")
    except (TypeError, ValueError):
        la = float("nan")
    try:
        lo = float(lon) if lon is not None and not (isinstance(lon, float) and pd.isna(lon)) else float("nan")
    except (TypeError, ValueError):
        lo = float("nan")
    return la, lo


def build_feature_vector(
    bg: str | None,
    bg_to_zip: dict[str, str | None],
    zhvi_by_zip: dict[str, float],
    adi_by_bg: dict[str, tuple[float, float]],
    lat: float,
    lon: float,
    h9: str | None,
    agg_4g: dict[str, tuple[float, float, int]],
    agg_5g: dict[str, tuple[float, float, int]],
    fbi_by_county: dict[str, dict[str, float]],
) -> tuple[list[float], list[str]]:
    names = [
        "ext_patient_lat",
        "ext_patient_lon",
        "ext_has_block_group",
        "ext_bg_state_fips",
        "ext_bg_county_fips",
        "ext_bg_tract_code",
        "ext_bg_group_digit",
        "ext_zcta_zip_code",
        "ext_zhvi_home_value",
        "ext_adi_natrank",
        "ext_adi_staternk",
        "ext_cell_4g_max_mindown_mbps",
        "ext_cell_4g_max_minup_mbps",
        "ext_cell_4g_layer_row_count",
        "ext_cell_5g_max_mindown_mbps",
        "ext_cell_5g_max_minup_mbps",
        "ext_cell_5g_layer_row_count",
        "ext_fbi_nibrs_incidents_per_100k",
        "ext_fbi_nibrs_violent_incidents_per_100k",
        "ext_fbi_nibrs_property_incidents_per_100k",
    ]
    has_bg = 1.0 if bg else 0.0
    st_f, co_f, tr_f, gdig = _parse_bg_fips_parts(bg)
    bg_key = None
    if bg:
        d = "".join(c for c in str(bg) if c.isdigit())
        if d:
            bg_key = d.zfill(12)[-12:]
    zip_s = bg_to_zip.get(bg_key) if bg_key else bg_to_zip.get(bg) if bg else None
    if zip_s is None and bg:
        zip_s = bg_to_zip.get(str(bg).strip())
    zip_float = float(int(zip_s)) if zip_s and str(zip_s).isdigit() else float("nan")
    zhvi = float(zhvi_by_zip.get(zip_s, float("nan"))) if zip_s else float("nan")
    adi_nat, adi_st = (float("nan"), float("nan"))
    if bg_key and bg_key in adi_by_bg:
        adi_nat, adi_st = adi_by_bg[bg_key]
        adi_nat, adi_st = float(adi_nat), float(adi_st)
    elif bg and str(bg).strip() in adi_by_bg:
        adi_nat, adi_st = adi_by_bg[str(bg).strip()]
        adi_nat, adi_st = float(adi_nat), float(adi_st)

    if h9 and agg_4g:
        g4 = agg_4g.get(h9)
    else:
        g4 = None
    if h9 and agg_5g:
        g5 = agg_5g.get(h9)
    else:
        g5 = None

    md4, mu4, n4 = (float("nan"), float("nan"), float("nan"))
    if g4 is not None:
        md4, mu4, n4 = float(g4[0]), float(g4[1]), float(g4[2])
    md5, mu5, n5 = (float("nan"), float("nan"), float("nan"))
    if g5 is not None:
        md5, mu5, n5 = float(g5[0]), float(g5[1]), float(g5[2])

    cf = _county_fips5(bg_key) if bg_key else _county_fips5(bg)
    fr = fbi_by_county.get(cf, {}) if cf else {}
    fbi_t = float(fr.get("fbi_nibrs_incidents_per_100k", float("nan")))
    fbi_v = float(fr.get("fbi_nibrs_violent_incidents_per_100k", float("nan")))
    fbi_p = float(fr.get("fbi_nibrs_property_incidents_per_100k", float("nan")))

    vec = [
        lat,
        lon,
        has_bg,
        st_f,
        co_f,
        tr_f,
        gdig,
        zip_float,
        zhvi,
        adi_nat,
        adi_st,
        md4,
        mu4,
        n4,
        md5,
        mu5,
        n5,
        fbi_t,
        fbi_v,
        fbi_p,
    ]
    return vec, names


def augment_sequences(
    bundle: dict,
    zhvi_by_zip: dict[str, float],
    adi_by_bg: dict[str, tuple[float, float]],
    bg_to_zip: dict[str, str | None],
    agg_4g: dict[str, tuple[float, float, int]],
    agg_5g: dict[str, tuple[float, float, int]],
    fbi_by_county: dict[str, dict[str, float]],
) -> list[dict]:
    seqs = bundle["sequences"]
    out_seqs: list[dict] = []
    h3_cache: dict[tuple[float, float], str | None] = {}

    def h9_for(lat: float, lon: float) -> str | None:
        if not agg_4g and not agg_5g:
            return None
        if lat != lat or lon != lon:
            return None
        key = (round(lat, 5), round(lon, 5))
        if key not in h3_cache:
            h3_cache[key] = lat_lon_to_h3_res9(lat, lon)
        return h3_cache[key]

    for seq in tqdm(seqs, desc="sequences", unit="seq"):
        row = dict(seq)
        bg = _patient_bg(seq)
        lat, lon = _patient_lat_lon(seq)
        vec, names = build_feature_vector(
            bg,
            bg_to_zip,
            zhvi_by_zip,
            adi_by_bg,
            lat,
            lon,
            h9_for(lat, lon),
            agg_4g,
            agg_5g,
            fbi_by_county,
        )
        t = _seq_len(seq)
        v = torch.tensor(vec, dtype=torch.float32)
        if t == 0:
            feat = torch.zeros((0, len(vec)), dtype=torch.float32)
        else:
            feat = v.unsqueeze(0).expand(t, -1).clone()
        row["external_feature_tensor"] = feat
        row["external_feature_names"] = names
        out_seqs.append(row)
    return out_seqs


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Attach external numeric features to each token row.")
    ap.add_argument(
        "--input",
        type=Path,
        default=here / "patient_sequences_encounter_only_with_sdoh_status_and_fips.pt",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=here / "patient_sequences_with_external_features.pt",
    )
    ap.add_argument(
        "--zillow-csv",
        type=Path,
        default=DEFAULT_DATASETS / "Zillow.csv",
        help="ZIP-level ZHVI (uses latest YYYY-MM-DD column as value).",
    )
    ap.add_argument(
        "--adi-csv",
        type=Path,
        default=DEFAULT_DATASETS / "ADI" / "KS_2023_ADI_Census_Block_Group_v4_0_1.csv",
        help="Block-group ADI (expects FIPS, ADI_NATRANK, ADI_STATERNK).",
    )
    ap.add_argument(
        "--census-blkgrp-dir",
        type=Path,
        default=DEFAULT_CENSUS_BLKGRP,
        help="Folder with tab20_blkgrp20_blkgrp10_*.txt and optional tab20_zcta520_tract20_natl.txt.",
    )
    ap.add_argument("--fourg-dbf", type=Path, default=DEFAULT_4G_DBF)
    ap.add_argument("--fiveg-dbf", type=Path, default=DEFAULT_5G_DBF)
    ap.add_argument("--fbi-dir", type=Path, default=DEFAULT_FBI_DIR)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N sequences (debug / smoke test).",
    )
    ap.add_argument(
        "--skip-census-zip",
        action="store_true",
        help="Do not resolve ZCTA; ext_zcta_zip_code and ext_zhvi_home_value will be NaN (block group parts + ADI still filled).",
    )
    ap.add_argument(
        "--skip-broadband",
        action="store_true",
        help="Skip FCC 4G/5G .dbf aggregation (cell columns will be NaN).",
    )
    ap.add_argument(
        "--skip-fbi",
        action="store_true",
        help="Skip NIBRS county rate tables (FBI columns will be NaN).",
    )
    args = ap.parse_args()

    print("Loading", args.input, flush=True)
    bundle = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "sequences" not in bundle:
        raise SystemExit("Input must be a dict with key 'sequences'.")

    seqs: list[dict] = bundle["sequences"]
    if args.limit and args.limit > 0:
        seqs = seqs[: args.limit]
        bundle = {**bundle, "sequences": seqs}

    print("Collecting block group FIPS…", flush=True)
    uniq_bg: set[str] = set()
    for seq in tqdm(seqs, desc="scan BG", unit="seq"):
        bg = _patient_bg(seq)
        if bg:
            uniq_bg.add(bg)

    zip_mode = "skipped"
    if args.skip_census_zip:
        print("Skipping Census ZCTA resolution (--skip-census-zip).", flush=True)
        bg_to_zip = {}
        for g in uniq_bg:
            d = "".join(c for c in str(g) if c.isdigit())
            if d:
                bg_to_zip[d.zfill(12)[-12:]] = None
            bg_to_zip[str(g).strip()] = None
    else:
        print("Resolving ZCTA from local Census files (blkgrp bridge or tract→ZCTA fallback)…", flush=True)
        try:
            bg_to_zip, zip_mode = _bg_to_zip_map(uniq_bg, args.census_blkgrp_dir)
        except FileNotFoundError as e:
            raise SystemExit(
                f"{e}\n\n"
                "Place either:\n"
                "  • tab20_blkgrp20_blkgrp10_*.txt + tab20_zcta520_tract20_natl.txt, or\n"
                "  • at least tab20_zcta520_tract20_natl.txt (tract→ZIP using BG[:11])\n"
                "under external_data/census_rel2020_blkgrp/, or use --skip-census-zip.\n"
            ) from e

    n_zip = sum(1 for v in bg_to_zip.values() if v)
    print(f"ZIP resolution mode: {zip_mode}; non-null ZIPs in map: {n_zip}", flush=True)

    print("Loading Zillow (ZHVI) and ADI lookups…", flush=True)
    zhvi_by_zip = load_zhvi_by_zip(args.zillow_csv)
    adi_by_bg = load_adi_by_block_group(args.adi_csv)
    print(
        f"  Zillow ZHVI: {len(zhvi_by_zip)} ZIP codes; "
        f"ADI: {len(adi_by_bg)} block groups in {args.adi_csv.name}",
        flush=True,
    )

    if args.skip_broadband:
        print("Skipping 4G/5G broadband (--skip-broadband).", flush=True)
        agg_4g: dict = {}
        agg_5g: dict = {}
    else:
        print("Aggregating FCC 4G H3 layer (dbf)…", flush=True)
        agg_4g = load_broadband_h3_agg(args.fourg_dbf, desc="4G dbf")
        print("Aggregating FCC 5G H3 layer (dbf)…", flush=True)
        agg_5g = load_broadband_h3_agg(args.fiveg_dbf, desc="5G dbf")

    if args.skip_fbi:
        print("Skipping FBI NIBRS (--skip-fbi).", flush=True)
        fbi_by_county = {}
    else:
        print("Building FBI NIBRS county rates…", flush=True)
        fbi_by_county = load_fbi_nibrs_county_rates(args.fbi_dir)

    out_bundle = dict(bundle)
    out_bundle["sequences"] = augment_sequences(
        bundle,
        zhvi_by_zip,
        adi_by_bg,
        bg_to_zip,
        agg_4g,
        agg_5g,
        fbi_by_county,
    )

    feat_names: list[str] = []
    for s in out_bundle["sequences"]:
        fn = s.get("external_feature_names")
        if isinstance(fn, list) and fn:
            feat_names = fn
            break

    out_bundle["external_feature_schema"] = {
        "description": "Per-token tensor external_feature_tensor[t, :] aligns with external_feature_names.",
        "features": feat_names,
        "zillow_csv": str(args.zillow_csv),
        "adi_csv": str(args.adi_csv),
        "census_blkgrp_dir": str(args.census_blkgrp_dir),
        "census_zip_skipped": bool(args.skip_census_zip),
        "zip_resolution_mode": zip_mode if not args.skip_census_zip else "skipped",
        "fourg_dbf": str(args.fourg_dbf),
        "fiveg_dbf": str(args.fiveg_dbf),
        "fbi_dir": str(args.fbi_dir),
        "block_group_note": "ext_bg_* are Census 2020 BG GEOID parts; ext_zcta_zip_code is ZCTA5 when resolved.",
        "zillow_note": "ext_zhvi_home_value joins Zillow ZIP ZHVI (latest month column) on ext_zcta_zip_code.",
        "adi_note": "ADI joins on 12-digit block group FIPS; add national ADI CSV for coverage outside default KS file.",
        "broadband_note": "FCC mobile coverage H3 res-9; patient lat/lon mapped to hex; max mindown/minup across layer rows.",
        "fbi_note": "County = first 5 chars of block group FIPS; rates use NIBRS incidents / max agency population in county (approximate).",
    }

    print("Saving", args.output, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_bundle, args.output)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
