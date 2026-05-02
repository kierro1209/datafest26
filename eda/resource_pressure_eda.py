#!/usr/bin/env python3
"""
Resource pressure / demand vs observed coverage EDA.

Builds proxies for where encounter demand may strain observed provider coverage
(department specialty and attending provider specialty). Does **not** reflect true
staffing or schedules — only patterns in the operational data.

Reads the same encounter export as journey_eda (`event_enriched.csv.gz`, ENCOUNTER rows).

Memory-conscious defaults: pyarrow CSV engine when available; high-cardinality strings as
categories after sort; lightweight in-place next-visit fields (no full transition matrix);
drops unused columns before aggregates; keeps only a slim column subset for repeat-rate
summaries; deletes the wide encounter frame before plotting.

Example:
  python eda/resource_pressure_eda.py --input data/processed/event_enriched.csv.gz \\
    --output-dir visuals/eda_resource --max-rows 500000

Requires: pandas, matplotlib; optional seaborn for heatmaps/scatter styling; optional pyarrow for faster one-shot CSV parsing; optional tqdm for a proper progress bar while reading chunks (otherwise INFO logs every few chunks).
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

try:
    from tqdm.auto import tqdm as _tqdm_bar

    _HAS_TQDM = True
except ImportError:
    _tqdm_bar = None
    _HAS_TQDM = False

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eda.journey_eda import (
    FINANCE_COL,
    FOOD_COL,
    HOUSING_COL,
    TRANSPORT_COL,
    configure_logging,
    filter_eda_rows,
)

logger = logging.getLogger(__name__)

# Slim slice for repeat/journey summaries — avoids retaining the full encounter-wide row.
EDA_SLIM_COLS = [
    "EncounterKey",
    "PatientDurableKey",
    "event_datetime",
    "DepartmentSpecialty",
    "GroupName",
    "next_department_specialty",
    "days_to_next",
    "days_to_next_int",
    TRANSPORT_COL,
]

# Compress repeated strings after sort (safe: tie-break sort uses EncounterKey before categorization).
_CAT_COLS = [
    "PatientDurableKey",
    "EncounterKey",
    "DepartmentSpecialty",
    "GroupCode",
    "GroupName",
    "AttendingProviderDurableKey",
    "attending_provider_PrimarySpecialty",
]

# Safe to drop after lightweight next-event columns exist (not used by aggregates or EDA_SLIM_COLS).
_DROP_AFTER_NEXT_EVENT = (
    "PatientBirthYearBin",
    "SexAssignedAtBirth",
    "IsEdVisit",
    "IsHospitalAdmission",
    "IsOutpatientFaceToFaceVisit",
    "sdoh_any_observed",
    FOOD_COL,
    HOUSING_COL,
    FINANCE_COL,
    "Type",
    "VisitTypeDescription",
    "DiagnosisValue",
    "DepartmentKey",
    "DepartmentType",
    "attending_provider_PrimarySpecialty",
)


def _add_next_event_resource_inplace(df: pd.DataFrame) -> None:
    """Minimal shift columns for resource EDA only — no copy; skips unused transition flags."""
    g = df.groupby("PatientDurableKey", sort=False, observed=False)
    df["next_event_datetime"] = g["event_datetime"].shift(-1)
    df["next_department_specialty"] = g["DepartmentSpecialty"].shift(-1)
    delta = df["next_event_datetime"] - df["event_datetime"]
    df["days_to_next"] = delta.dt.total_seconds() / 86400.0
    df["days_to_next_int"] = np.floor(df["days_to_next"]).astype("Int64")


def _categorize_high_cardinality(df: pd.DataFrame) -> None:
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _drop_optional_columns(df: pd.DataFrame, names: tuple[str, ...]) -> None:
    existing = [c for c in names if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)


def _as_missing_str(series: pd.Series) -> pd.Series:
    """Replace NA with (missing); safe when ``series`` is Categorical (fillna cannot add categories)."""
    return series.astype("object").fillna("(missing)").astype(str)


# Labels treated as unknown / placeholder for resource EDA (excluded from aggregates & plots).
_BAD_DIMENSION_LABELS = frozenset(
    {
        "",
        "(missing)",
        "missing",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
        "unknown",
        "unk",
    }
)


def _normalized_lower_token(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "(missing)"
    if pd.isna(val):
        return "(missing)"
    t = str(val).strip().lower()
    if t in {"", "nan", "<na>", "nat"}:
        return "(missing)"
    return t


def _mask_known_resource_dimensions(df: pd.DataFrame) -> pd.Series:
    """True for rows with known department specialty, provider specialty, and diagnosis group name."""
    ds = df["DepartmentSpecialty_disp"].map(_normalized_lower_token)
    ps = df["ProviderSpecialty"].map(_normalized_lower_token)
    gn = _as_missing_str(df["GroupName"]).map(_normalized_lower_token)
    ok = ~(ds.isin(_BAD_DIMENSION_LABELS) | ps.isin(_BAD_DIMENSION_LABELS) | gn.isin(_BAD_DIMENSION_LABELS))
    return ok


def _filter_unknown_dimension_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop encounters whose specialty or diagnosis label is missing/placeholder."""
    m = _mask_known_resource_dimensions(df)
    n_drop = int((~m).sum())
    if n_drop:
        logger.info(
            "Excluding %s encounters with unknown dept/provider specialty or diagnosis group "
            "(missing or placeholder labels)",
            f"{n_drop:,}",
        )
    out = df.loc[m].reset_index(drop=True)
    return out


def _uncoded_attending_specialty_mask(provider_specialty: pd.Series) -> pd.Series:
    """
    True when attending provider specialty from source is a placeholder.

    The export often uses the literal string '*Unspecified*' (or variants) instead of null,
    which would otherwise dominate diagnosis × specialty aggregates.
    """
    s = provider_specialty.astype(str).str.strip()
    core = s.str.lower().str.replace("*", "", regex=False).str.strip()
    return (
        core.eq("unspecified")
        | core.eq("(missing)")
        | core.isin(["", "nan", "none", "null", "<na>", "nat"])
        | core.eq("missing")
    )


def _effective_specialty_for_bottleneck(df: pd.DataFrame) -> pd.Series:
    """Attending specialty when coded; otherwise fall back to department specialty for the pair label."""
    uncoded = _uncoded_attending_specialty_mask(df["ProviderSpecialty"])
    dept = df["DepartmentSpecialty_disp"].astype(str)
    att = df["ProviderSpecialty"].astype(str)
    picked = np.where(uncoded, dept, att)
    return pd.Series(picked, index=df.index, dtype=object).map(
        lambda x: str(x).strip().strip("*").strip()
    )


def _usecols_resource_event_enriched() -> list[str]:
    base = [
        "event_grain",
        "EncounterKey",
        "PatientDurableKey",
        "Date",
        "AdmissionInstant",
        "Type",
        "VisitTypeDescription",
        "DepartmentKey",
        "DepartmentType",
        "DepartmentSpecialty",
        "DiagnosisValue",
        "GroupCode",
        "GroupName",
        "PatientBirthYearBin",
        "SexAssignedAtBirth",
        "IsEdVisit",
        "IsHospitalAdmission",
        "IsOutpatientFaceToFaceVisit",
        "sdoh_any_observed",
        TRANSPORT_COL,
        FOOD_COL,
        HOUSING_COL,
        FINANCE_COL,
        "AttendingProviderDurableKey",
        "attending_provider_PrimarySpecialty",
    ]
    return base


def _usecols_resource_encounter_enriched() -> list[str]:
    return [c for c in _usecols_resource_event_enriched() if c != "event_grain"]


def _read_csv_build_kw(
    path: Path,
    table_kind: str,
    *,
    nrows: int | None,
) -> dict:
    if table_kind == "event_enriched":
        usecols = _usecols_resource_event_enriched()
    else:
        usecols = _usecols_resource_encounter_enriched()

    read_kw: dict = {
        "filepath_or_buffer": path,
        "usecols": usecols,
        "dtype": {
            "EncounterKey": str,
            "PatientDurableKey": str,
            "DepartmentKey": str,
            "AttendingProviderDurableKey": str,
        },
        "low_memory": False,
    }
    if nrows is not None:
        read_kw["nrows"] = nrows
    return read_kw


def _load_csv_chunked(
    path: Path,
    *,
    table_kind: str,
    max_rows: int | None,
    chunksize: int,
    show_progress: bool,
) -> pd.DataFrame:
    """Stream the gzip/CSV in chunks (shows tqdm or periodic INFO). Chunked path uses engine='c'."""
    read_kw = _read_csv_build_kw(path, table_kind, nrows=None)
    read_kw["chunksize"] = chunksize
    read_kw["engine"] = "c"

    iterator = pd.read_csv(**read_kw)
    chunks: list[pd.DataFrame] = []
    raw_rows = 0
    enc_rows = 0
    n_chunk = 0

    use_tqdm = show_progress and _HAS_TQDM and _tqdm_bar is not None
    pbar = (
        _tqdm_bar(
            desc="Reading CSV",
            unit="rows",
            unit_scale=True,
            dynamic_ncols=True,
        )
        if use_tqdm
        else None
    )

    for chunk in iterator:
        n_chunk += 1
        raw_rows += len(chunk)

        if table_kind == "event_enriched":
            chunk = chunk.loc[
                chunk["event_grain"].astype(str).str.upper().eq("ENCOUNTER")
            ].drop(columns=["event_grain"], errors="ignore")

        if max_rows is not None:
            need = max_rows - enc_rows
            if need <= 0:
                break
            if len(chunk) > need:
                chunk = chunk.iloc[:need].copy()

        if len(chunk) == 0:
            continue

        enc_rows += len(chunk)
        chunks.append(chunk)

        if pbar is not None:
            pbar.update(len(chunk))
            pbar.set_postfix(enc=f"{enc_rows:,}", n_chunks=n_chunk)
        elif show_progress and (n_chunk <= 3 or n_chunk % 5 == 0):
            logger.info(
                "CSV progress: chunk %s (~%s raw rows read, %s encounter rows kept)",
                n_chunk,
                f"{raw_rows:,}",
                f"{enc_rows:,}",
            )

        if max_rows is not None and enc_rows >= max_rows:
            break

    if pbar is not None:
        pbar.close()

    if not chunks:
        return pd.DataFrame()

    logger.info("Concatenating %s chunks (%s encounter rows)...", len(chunks), f"{enc_rows:,}")
    try:
        return pd.concat(chunks, ignore_index=True, copy=False)
    except TypeError:
        return pd.concat(chunks, ignore_index=True)


def _load_csv_single_shot(
    path: Path,
    *,
    table_kind: str,
    max_rows: int | None,
) -> pd.DataFrame:
    nrows = None
    if max_rows is not None:
        nrows = max_rows * 3 if table_kind == "event_enriched" else max_rows

    read_kw = _read_csv_build_kw(path, table_kind, nrows=nrows)

    try:
        df = pd.read_csv(**read_kw, engine="pyarrow")
    except Exception as exc:
        logger.debug("read_csv(pyarrow) failed (%s); retrying with default engine", exc)
        df = pd.read_csv(**read_kw)

    if table_kind == "event_enriched":
        df = df.loc[df["event_grain"].astype(str).str.upper().eq("ENCOUNTER")].drop(
            columns=["event_grain"]
        )
        if max_rows is not None and len(df) > max_rows:
            df = df.iloc[:max_rows].copy()

    return df


def _enrich_loaded_encounter_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse times, sort, categorize, add calendar features."""
    if len(df) == 0:
        return df

    df["EncounterKey"] = df["EncounterKey"].astype(str)
    df["PatientDurableKey"] = df["PatientDurableKey"].astype(str)

    inst = pd.to_datetime(df["AdmissionInstant"], errors="coerce")
    day = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    if day.isna().all():
        day = pd.to_datetime(df["Date"], errors="coerce")
    df["event_datetime"] = inst.fillna(day)
    _drop_optional_columns(df, ("Date", "AdmissionInstant"))

    logger.info("Sorting %s rows by patient and time...", f"{len(df):,}")
    df.sort_values(
        ["PatientDurableKey", "event_datetime", "EncounterKey"],
        kind="mergesort",
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    # Must run before _categorize_high_cardinality.
    df["DepartmentSpecialty_disp"] = _as_missing_str(df["DepartmentSpecialty"])
    df["ProviderSpecialty"] = _as_missing_str(df["attending_provider_PrimarySpecialty"])

    logger.info("Compressing key columns to category dtype...")
    _categorize_high_cardinality(df)

    df["month"] = df["event_datetime"].dt.to_period("M").dt.to_timestamp()
    df["week"] = df["event_datetime"] - pd.to_timedelta(
        df["event_datetime"].dt.dayofweek, unit="D"
    )
    df["week"] = df["week"].dt.normalize()

    n_pat = df["PatientDurableKey"].nunique()
    logger.info(
        "Loaded %s encounter rows (%s patients); span %s → %s",
        f"{len(df):,}",
        f"{n_pat:,}",
        df["event_datetime"].min(),
        df["event_datetime"].max(),
    )
    return df


def load_encounters_resource(
    path: Path,
    *,
    max_rows: int | None,
    table_kind: str,
    chunksize: int | None = 200_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load encounter rows including attending provider keys/specialty.

    By default reads the CSV in chunks so progress is visible (tqdm bar if installed,
    else periodic INFO logs). Use ``chunksize=0`` for one-shot ``read_csv`` (may look stuck).
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        sz_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Reading %s (%.1f MiB); table_kind=%s", path, sz_mb, table_kind)
    except OSError:
        logger.info("Reading %s; table_kind=%s", path, table_kind)

    if chunksize and chunksize > 0:
        if show_progress and not _HAS_TQDM:
            logger.info(
                "Install tqdm (`pip install tqdm`) for a progress bar; logging chunk progress every ~25M raw rows."
            )
        df = _load_csv_chunked(
            path,
            table_kind=table_kind,
            max_rows=max_rows,
            chunksize=int(chunksize),
            show_progress=show_progress,
        )
    else:
        if show_progress:
            logger.info("Reading entire file in one pass (no chunk progress)...")
        df = _load_csv_single_shot(path, table_kind=table_kind, max_rows=max_rows)

    return _enrich_loaded_encounter_frame(df)


def _safe_div_num_den(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype(float) / den.replace(0, np.nan).astype(float)
    return out


def build_monthly_specialty_pressure(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["month", "DepartmentSpecialty_disp"], observed=False).agg(
        encounters=("EncounterKey", "nunique"),
        patients=("PatientDurableKey", "nunique"),
        providers=("AttendingProviderDurableKey", "nunique"),
        diagnosis_groups=("GroupCode", "nunique"),
    )
    out = g.reset_index()
    out["encounters_per_provider"] = _safe_div_num_den(out["encounters"], out["providers"])
    return out


def merge_monthly_return_rates(
    monthly: pd.DataFrame,
    eda: pd.DataFrame,
) -> pd.DataFrame:
    """Mean P(return ≤30d) by month × department specialty from linked pairs."""
    if len(eda) == 0:
        monthly = monthly.copy()
        monthly["return_30d"] = np.nan
        monthly["return_7d"] = np.nan
        return monthly

    dtn = eda["days_to_next"].astype(float)
    tmp = pd.DataFrame(
        {
            "month": eda["event_datetime"].dt.to_period("M").dt.to_timestamp(),
            "DepartmentSpecialty_disp": _as_missing_str(eda["DepartmentSpecialty"]),
            "_r30": (dtn <= 30).astype(np.float64),
            "_r7": (dtn <= 7).astype(np.float64),
        }
    )
    r = tmp.groupby(["month", "DepartmentSpecialty_disp"], observed=False).agg(
        return_30d=("_r30", "mean"),
        return_7d=("_r7", "mean"),
    ).reset_index()
    del tmp
    return monthly.merge(r, on=["month", "DepartmentSpecialty_disp"], how="left")


def add_stress_score(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    cols_rank = ["encounters_per_provider", "patients", "return_30d", "diagnosis_groups"]
    for c in cols_rank:
        if c not in m.columns:
            m[c] = np.nan
    # Missing repeat-rate for a specialty-month (no linked next visit in slice) → rank at bottom.
    fill_map = {
        "encounters_per_provider": m["encounters_per_provider"].fillna(0),
        "patients": m["patients"],
        "return_30d": m["return_30d"].fillna(0),
        "diagnosis_groups": m["diagnosis_groups"],
    }
    for c in cols_rank:
        m[f"{c}_pct"] = fill_map[c].rank(pct=True, method="average")
    m["stress_score"] = m[[f"{c}_pct" for c in cols_rank]].sum(axis=1, skipna=False)
    return m


def build_diag_pressure(eda: pd.DataFrame) -> pd.DataFrame:
    if len(eda) == 0:
        return pd.DataFrame()
    gn = _as_missing_str(eda["GroupName"])
    ns = _as_missing_str(eda["next_department_specialty"])
    dtn = eda["days_to_next"].astype(float)
    sub = pd.DataFrame(
        {
            "GroupName_disp": gn,
            "next_department_specialty_disp": ns,
            "EncounterKey": eda["EncounterKey"],
            "PatientDurableKey": eda["PatientDurableKey"],
            "days_to_next": eda["days_to_next"],
            "return_7d": dtn <= 7,
            "return_30d": dtn <= 30,
        }
    )
    g = sub.groupby("GroupName_disp", observed=False).agg(
        encounters=("EncounterKey", "nunique"),
        patients=("PatientDurableKey", "nunique"),
        median_gap=("days_to_next", "median"),
        return_7d=("return_7d", "mean"),
        return_30d=("return_30d", "mean"),
        next_specialties=("next_department_specialty_disp", "nunique"),
    )
    return g.reset_index()


def build_diag_provider_pressure(df: pd.DataFrame) -> pd.DataFrame:
    uncoded = _uncoded_attending_specialty_mask(df["ProviderSpecialty"])
    if uncoded.any():
        logger.info(
            "Diagnosis × specialty pairs: %s / %s rows had uncoded attending specialty (*Unspecified*, etc.); "
            "using department specialty for that side of the pair.",
            f"{int(uncoded.sum()):,}",
            f"{len(df):,}",
        )
    eff_spec = _effective_specialty_for_bottleneck(df)
    sub = pd.DataFrame(
        {
            "GroupName_disp": _as_missing_str(df["GroupName"]),
            "ProviderSpecialty": eff_spec,
            "EncounterKey": df["EncounterKey"],
            "AttendingProviderDurableKey": df["AttendingProviderDurableKey"],
        }
    )
    g = sub.groupby(["GroupName_disp", "ProviderSpecialty"], observed=False).agg(
        encounters=("EncounterKey", "nunique"),
        providers=("AttendingProviderDurableKey", "nunique"),
    )
    out = g.reset_index()
    out["encounters_per_provider"] = _safe_div_num_den(out["encounters"], out["providers"])
    return out


def build_weekly_specialty(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["week", "DepartmentSpecialty_disp"], observed=False).agg(
        encounters=("EncounterKey", "nunique"),
        providers=("AttendingProviderDurableKey", "nunique"),
    )
    out = g.reset_index()
    out["encounters_per_provider"] = _safe_div_num_den(out["encounters"], out["providers"])

    def z_within_group(s: pd.Series) -> pd.Series:
        m = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or pd.isna(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - m) / sd

    out["demand_z"] = out.groupby("DepartmentSpecialty_disp", observed=False)["encounters"].transform(
        z_within_group
    )
    out["stress_week"] = out["demand_z"] >= 2.0
    return out


def build_sdoh_pressure(eda: pd.DataFrame) -> pd.DataFrame:
    if TRANSPORT_COL not in eda.columns or len(eda) == 0:
        return pd.DataFrame()
    tr = eda[TRANSPORT_COL].fillna(0).astype(int)
    sub = pd.DataFrame(
        {
            "GroupName_disp": _as_missing_str(eda["GroupName"]),
            "transportation_need_flag": np.where(tr == 1, "flagged", "not_flagged"),
            "EncounterKey": eda["EncounterKey"],
            "PatientDurableKey": eda["PatientDurableKey"],
            "days_to_next": eda["days_to_next"],
        }
    )
    sub["_r30"] = sub["days_to_next"].astype(float) <= 30
    g = sub.groupby(["transportation_need_flag", "GroupName_disp"], observed=False).agg(
        encounters=("EncounterKey", "nunique"),
        patients=("PatientDurableKey", "nunique"),
        median_gap=("days_to_next", "median"),
        return_30d=("_r30", "mean"),
    )
    return g.reset_index()


# --- plots ---


def plot_monthly_encounters_per_provider(
    monthly: pd.DataFrame,
    out: Path,
    *,
    top_n_specialties: int = 15,
) -> None:
    vc = (
        monthly.groupby("DepartmentSpecialty_disp", observed=False)["encounters"]
        .sum()
        .sort_values(ascending=False)
    )
    keep = set(vc.head(top_n_specialties).index)
    sub = monthly[monthly["DepartmentSpecialty_disp"].isin(keep)]

    fig, ax = plt.subplots(figsize=(11, 5))
    for spec in sorted(keep, key=lambda s: vc.get(s, 0), reverse=True):
        m = sub.loc[sub["DepartmentSpecialty_disp"] == spec].sort_values("month")
        ax.plot(m["month"], m["encounters_per_provider"], marker=".", ms=3, label=spec[:40])

    ax.set_xlabel("Month")
    ax.set_ylabel("Encounters per distinct attending provider (observed)")
    ax.set_title(
        "Monthly encounters per observed attending provider by department specialty "
        f"(top {top_n_specialties} specialties by volume)"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_encounters_vs_providers_scatter(
    monthly: pd.DataFrame,
    out: Path,
    *,
    top_n_specialties: int = 12,
) -> None:
    vc = (
        monthly.groupby("DepartmentSpecialty_disp", observed=False)["encounters"]
        .sum()
        .sort_values(ascending=False)
    )
    keep = set(vc.head(top_n_specialties).index)
    sub = monthly[monthly["DepartmentSpecialty_disp"].isin(keep)]

    fig, ax = plt.subplots(figsize=(9, 6))
    if _HAS_SNS:
        sns.scatterplot(
            data=sub,
            x="encounters",
            y="providers",
            hue="DepartmentSpecialty_disp",
            size="patients",
            sizes=(20, 400),
            alpha=0.65,
            ax=ax,
            legend="brief",
        )
    else:
        for spec in keep:
            ms = sub.loc[sub["DepartmentSpecialty_disp"] == spec]
            ax.scatter(ms["encounters"], ms["providers"], alpha=0.5, label=spec[:30])
        ax.legend(fontsize=6)

    ax.set_xlabel("Monthly encounters (distinct EncounterKey)")
    ax.set_ylabel("Distinct attending providers (observed)")
    ax.set_title("Monthly demand vs observed provider count by specialty")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diag_volume_vs_return(
    diag_pressure: pd.DataFrame,
    out: Path,
    *,
    label_top_n: int = 25,
    max_groups_plot: int = 80,
) -> None:
    if len(diag_pressure) == 0:
        logger.warning("Skipping diagnosis volume vs return plot: empty diag_pressure")
        return

    dp = diag_pressure.sort_values("encounters", ascending=False).head(max_groups_plot)

    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = (dp["patients"] / dp["patients"].max() * 350 + 15).clip(15, 350)
    ax.scatter(dp["return_30d"], dp["encounters"], s=sizes, alpha=0.55, c="steelblue")

    top_labels = dp.nlargest(label_top_n, "encounters")
    for _, row in top_labels.iterrows():
        ax.annotate(
            str(row["GroupName_disp"])[:35],
            (row["return_30d"], row["encounters"]),
            fontsize=6,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Share of encounters with next visit within 30 days")
    ax.set_ylabel("Encounter volume (distinct EncounterKey)")
    ax.set_title(
        "Diagnosis groups: near-term return rate vs volume "
        f"(bubble size ∝ patients; top {label_top_n} labeled)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_stress_heatmap(
    monthly_stress: pd.DataFrame,
    out: Path,
    *,
    max_specialties: int = 35,
) -> None:
    if len(monthly_stress) == 0:
        return
    vol = monthly_stress.groupby("DepartmentSpecialty_disp", observed=False)["encounters"].sum()
    top_specs = vol.nlargest(max_specialties).index
    sub = monthly_stress[monthly_stress["DepartmentSpecialty_disp"].isin(top_specs)]
    pivot = sub.pivot_table(
        index="DepartmentSpecialty_disp",
        columns="month",
        values="stress_score",
        aggfunc="mean",
    )
    pivot = pivot.reindex(vol.loc[top_specs].index)

    fig, ax = plt.subplots(figsize=(max(10, 0.22 * pivot.shape[1]), max(6, 0.18 * pivot.shape[0])))
    if _HAS_SNS:
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Stress score (rank sum)"})
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x)[:7] for x in pivot.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="Stress score")

    ax.set_xlabel("Month")
    ax.set_ylabel("Department specialty")
    ax.set_title(
        "Specialty × month stress score (encounters/provider, patients, return≤30d, diagnosis breadth)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diag_provider_bottlenecks(
    diag_provider: pd.DataFrame,
    out: Path,
    *,
    top_n: int = 20,
) -> None:
    if len(diag_provider) == 0:
        return
    dp = diag_provider.dropna(subset=["encounters_per_provider"])
    dp = dp.nlargest(top_n, "encounters_per_provider")

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.28)))
    labels = (
        dp["GroupName_disp"].astype(str).str[:28]
        + " → "
        + dp["ProviderSpecialty"].astype(str).str[:34]
    )
    y = np.arange(len(dp))
    ax.barh(y, dp["encounters_per_provider"].astype(float), color="darkslateblue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Encounters per distinct attending provider (observed)")
    ax.set_title(
        f"Top {top_n} diagnosis × specialty pairs by encounters per provider\n"
        "(right-hand side: attending specialty when present in data; otherwise department specialty — "
        "many rows use literal *Unspecified* instead of null)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_weekly_demand_z(
    weekly: pd.DataFrame,
    out: Path,
    *,
    top_n_specialties: int = 8,
) -> None:
    if len(weekly) == 0:
        return
    vol = weekly.groupby("DepartmentSpecialty_disp", observed=False)["encounters"].sum()
    keep = set(vol.nlargest(top_n_specialties).index)
    sub = weekly[weekly["DepartmentSpecialty_disp"].isin(keep)].sort_values(["DepartmentSpecialty_disp", "week"])

    fig, ax = plt.subplots(figsize=(11, 5))
    for spec in sorted(keep, key=lambda s: vol.get(s, 0), reverse=True):
        w = sub.loc[sub["DepartmentSpecialty_disp"] == spec]
        ax.plot(w["week"], w["demand_z"], alpha=0.75, linewidth=1.2, label=spec[:35])

    ax.axhline(2.0, color="crimson", linestyle="--", linewidth=1, label="z = 2 (stress week)")
    ax.set_xlabel("Week (Monday start)")
    ax.set_ylabel("Encounter volume z-score within specialty")
    ax.set_title(
        "Weekly demand spikes (z-score of encounter counts) — interpret as workload variability"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_all(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="notebook")

    df = load_encounters_resource(
        Path(args.input),
        max_rows=args.max_rows,
        table_kind=args.table_kind,
        chunksize=args.read_chunksize,
        show_progress=not args.no_progress,
    )

    df = _filter_unknown_dimension_rows(df)

    _add_next_event_resource_inplace(df)
    _drop_optional_columns(df, _DROP_AFTER_NEXT_EVENT)
    gc.collect()

    monthly = build_monthly_specialty_pressure(df)
    weekly = build_weekly_specialty(df)
    diag_provider = build_diag_provider_pressure(df)

    eda = filter_eda_rows(df, args.max_gap_days, columns=EDA_SLIM_COLS)

    del df
    gc.collect()
    logger.info(
        "Released wide encounter frame; slim EDA slice has %s linked rows (%s cols)",
        f"{len(eda):,}",
        len(eda.columns),
    )

    monthly = merge_monthly_return_rates(monthly, eda)
    monthly_stress = add_stress_score(monthly)

    diag_pressure = build_diag_pressure(eda)
    sdoh_pressure = build_sdoh_pressure(eda)

    monthly.to_csv(out_dir / "monthly_specialty_pressure.csv", index=False)
    monthly_stress.to_csv(out_dir / "monthly_specialty_stress.csv", index=False)
    if len(diag_pressure):
        diag_pressure.to_csv(out_dir / "diag_group_repeat_pressure.csv", index=False)
    diag_provider.to_csv(out_dir / "diag_by_provider_specialty.csv", index=False)
    weekly.to_csv(out_dir / "weekly_specialty_demand_z.csv", index=False)
    if len(sdoh_pressure):
        sdoh_pressure.to_csv(out_dir / "sdoh_transport_pressure_by_diag.csv", index=False)

    top_tbl = monthly_stress.sort_values("stress_score", ascending=False).head(40)
    top_tbl.to_csv(out_dir / "top_stressed_specialty_month.csv", index=False)

    plot_monthly_encounters_per_provider(
        monthly,
        out_dir / "01_monthly_encounters_per_provider_by_specialty.png",
        top_n_specialties=args.top_specialties_plots,
    )
    plot_encounters_vs_providers_scatter(
        monthly,
        out_dir / "02_monthly_encounters_vs_provider_count.png",
        top_n_specialties=min(12, args.top_specialties_plots),
    )
    plot_diag_volume_vs_return(
        diag_pressure,
        out_dir / "03_diag_volume_vs_return_30d.png",
    )
    plot_stress_heatmap(
        monthly_stress,
        out_dir / "04_specialty_month_stress_heatmap.png",
        max_specialties=args.heatmap_specialties,
    )
    plot_diag_provider_bottlenecks(
        diag_provider,
        out_dir / "05_top_diag_provider_specialty_bottlenecks.png",
        top_n=args.top_bottlenecks,
    )
    plot_weekly_demand_z(
        weekly,
        out_dir / "06_weekly_demand_zscore_by_specialty.png",
        top_n_specialties=min(8, args.top_specialties_plots),
    )

    # Simple SDOH comparison: return_30d for top diagnosis groups
    if len(sdoh_pressure) and len(eda):
        vc = _as_missing_str(eda["GroupName"]).value_counts()
        top_g = list(vc.head(15).index)
        sp = sdoh_pressure[sdoh_pressure["GroupName_disp"].isin(top_g)]
        if len(sp) > 0:
            pivot = sp.pivot_table(
                index="GroupName_disp",
                columns="transportation_need_flag",
                values="return_30d",
                aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(9, 6))
            pivot.plot(kind="bar", ax=ax, rot=35)
            ax.set_ylabel("P(next visit within 30 days)")
            ax.set_xlabel("Diagnosis group (top 15 by volume)")
            ax.set_title("Near-term return rate: transportation SDOH flag vs not (linked pairs)")
            ax.legend(title="Transport flag")
            fig.tight_layout()
            fig.savefig(out_dir / "07_sdoh_transport_return30d_top_diagnoses.png", dpi=150)
            plt.close(fig)

    logger.info(
        "Resource pressure EDA finished in %.2fs — outputs under %s",
        time.perf_counter() - t0,
        out_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demand vs observed provider coverage proxies (resource pressure EDA)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data/processed/event_enriched.csv.gz",
        help="Encounter-level CSV.gz.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "visuals/eda_resource",
        help="Directory for PNG + CSV outputs.",
    )
    p.add_argument(
        "--table-kind",
        choices=("event_enriched", "encounter_enriched"),
        default="event_enriched",
        help="event_enriched: filter ENCOUNTER rows.",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional row cap after filters.")
    p.add_argument(
        "--read-chunksize",
        type=int,
        default=200_000,
        metavar="N",
        help="CSV rows per chunk while loading (default 200000); enables tqdm or periodic INFO progress. "
        "Use 0 for one-shot read_csv (no chunk progress).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bar and chunk progress logs during CSV read.",
    )
    p.add_argument(
        "--max-gap-days",
        type=int,
        default=365,
        help="Upper bound on days_to_next for repeat-rate features.",
    )
    p.add_argument(
        "--no-gap-cap",
        action="store_true",
        help="Do not cap days_to_next for the EDA slice.",
    )
    p.add_argument(
        "--top-specialties-plots",
        type=int,
        default=15,
        help="How many department specialties to show in line/z-score plots.",
    )
    p.add_argument(
        "--heatmap-specialties",
        type=int,
        default=35,
        help="Number of rows (specialties) in the stress heatmap.",
    )
    p.add_argument(
        "--top-bottlenecks",
        type=int,
        default=20,
        help="Top N diagnosis × provider-specialty pairs in bottleneck chart.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level.upper()))
    if args.no_gap_cap:
        args.max_gap_days = None
    elif args.max_gap_days is not None and args.max_gap_days < 0:
        raise SystemExit("--max-gap-days must be non-negative")
    run_all(args)


if __name__ == "__main__":
    main()
