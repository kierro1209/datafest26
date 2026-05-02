"""
Build patient_visits.csv from DataFest tables when the user file is missing:
encounters + patients + block-group centroids (tigercensuscodes) + ZIP from **local**
Census relationship files under ``external_data/census_rel2020_blkgrp/`` (no HTTP).

Download manually from Census (then run this script):

- **Required:** ``tab20_blkgrp20_blkgrp10_natl.txt`` and/or ``tab20_blkgrp20_blkgrp10_st*.txt``  
  https://www2.census.gov/geo/docs/maps-data/data/rel2020/blkgrp/

- **Optional (needed for ZIP_CODE):** ``tab20_zcta520_tract20_natl.txt``  
  https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/

Conda::

    conda activate datafest
    cd /path/to/datafest26/external_data
    python -m pip install -r requirements.txt
    python prepare_input.py --sample-n 2000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import bg_zcta

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BASE = Path(__file__).resolve().parent
CENSUS_REL_DIR = BASE / "census_rel2020_blkgrp"

_BG_PAT = re.compile(r"^\d{12}$")


def load_patients_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df.rename(columns={"DurableKey": "PATIENT_ID", "CensusBlockGroupFipsCode": "BLOCK_GROUP_GEOID"})
    df["PATIENT_ID"] = df["PATIENT_ID"].str.strip()
    df = df[df["BLOCK_GROUP_GEOID"].notna() & df["BLOCK_GROUP_GEOID"].str.match(_BG_PAT)]
    return df[["PATIENT_ID", "BLOCK_GROUP_GEOID"]].drop_duplicates(subset=["PATIENT_ID"])


def load_tiger(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, dtype=str)
    t["GEOID"] = t["GEOID"].astype(str).str.strip()
    t["CENTLAT"] = pd.to_numeric(t["CENTLAT"], errors="coerce")
    t["CENTLON"] = pd.to_numeric(t["CENTLON"], errors="coerce")
    return t


def sample_encounters(
    path: Path,
    patient_ids: set[str],
    n: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    rows = []
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk["PatientDurableKey"] = chunk["PatientDurableKey"].astype(str).str.strip()
        sub = chunk[chunk["PatientDurableKey"].isin(patient_ids)]
        if len(sub) == 0:
            continue
        sub = sub.rename(
            columns={
                "PatientDurableKey": "PATIENT_ID",
                "AdmissionInstant": "VISIT_DATE",
            }
        )
        use = sub[["PATIENT_ID", "VISIT_DATE", "EncounterKey"]].copy()
        rows.append(use)
        seen += len(use)
        if seen >= n * 4:
            break
    if not rows:
        return pd.DataFrame(columns=["PATIENT_ID", "VISIT_DATE", "EncounterKey"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["EncounterKey"])
    return out.head(n)


def build_patient_visits(
    sample_n: int = 2000,
    out_path: Path | None = None,
    census_rel_dir: Path | None = None,
) -> pd.DataFrame:
    patients = load_patients_map(DATA_DIR / "patients.csv")
    patient_ids = set(patients["PATIENT_ID"].tolist())
    enc = sample_encounters(DATA_DIR / "encounters.csv", patient_ids, sample_n)
    if enc.empty:
        raise RuntimeError("No encounters matched patients with valid block groups.")

    df = enc.merge(patients, on="PATIENT_ID", how="inner")
    tiger = load_tiger(DATA_DIR / "tigercensuscodes.csv")
    df = df.merge(
        tiger,
        left_on="BLOCK_GROUP_GEOID",
        right_on="GEOID",
        how="left",
    )
    df = df.rename(columns={"CENTLAT": "LAT", "CENTLON": "LON"})

    cdir = census_rel_dir or CENSUS_REL_DIR
    zmap = bg_zcta.zip_for_block_groups(set(df["BLOCK_GROUP_GEOID"].unique()), cdir)
    df = df.merge(zmap, on="BLOCK_GROUP_GEOID", how="left")

    df["ADDRESS"] = df.apply(
        lambda r: f"{r['LAT']}, {r['LON']} (block group centroid)",
        axis=1,
    )
    df["ZIP_CODE"] = df["ZIP_STR"]
    df["ADMIT_DATE"] = df["VISIT_DATE"]

    cols = [
        "PATIENT_ID",
        "EncounterKey",
        "ADDRESS",
        "ZIP_CODE",
        "ADMIT_DATE",
        "VISIT_DATE",
        "BLOCK_GROUP_GEOID",
        "LAT",
        "LON",
    ]
    df = df[[c for c in cols if c in df.columns]]

    out_path = out_path or (Path(__file__).resolve().parent / "patient_visits.csv")
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-n", type=int, default=2000)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--census-dir",
        type=Path,
        default=None,
        help="Folder with tab20_blkgrp20_blkgrp10_*.txt (and optional tab20_zcta520_tract20_natl.txt).",
    )
    args = p.parse_args()
    df = build_patient_visits(
        sample_n=args.sample_n,
        out_path=args.output,
        census_rel_dir=args.census_dir,
    )
    print(f"Wrote {len(df)} rows to {args.output or Path(__file__).resolve().parent / 'patient_visits.csv'}")


if __name__ == "__main__":
    main()

