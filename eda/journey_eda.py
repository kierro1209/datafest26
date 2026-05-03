#!/usr/bin/env python3
"""
Exploratory analysis for patient journey / next-event framing.

Reads encounter-level rows (default: data/processed/event_enriched.csv.gz with
event_grain == ENCOUNTER), builds consecutive-encounter features, and writes
figures + CSV summaries under data/processed/eda/ by default.

Example:
  python eda/journey_eda.py --input data/processed/event_enriched.csv.gz \\
    --output-dir data/processed/eda --max-rows 500000

  python eda/journey_eda.py --log-level DEBUG   # verbose timings + parse diagnostics

Requires: pandas, matplotlib; optional seaborn (styling), scikit-learn (clustering).
"""
from __future__ import annotations

import argparse
import logging
import time
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


def configure_logging(level: int) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        for h in root.handlers:
            h.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    # Keep third-party chatter down even when root is DEBUG (font matching, backends, …).
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def sdoh_observed_col(domain: str) -> str:
    return "sdoh_" + "".join(ch for ch in domain if ch.isalnum()) + "_observed"


TRANSPORT_COL = sdoh_observed_col("Transportation Needs")
FOOD_COL = sdoh_observed_col("Food Insecurity")
HOUSING_COL = sdoh_observed_col("Housing Stability")
FINANCE_COL = sdoh_observed_col("Financial Resource Strain")


def _usecols_event_enriched() -> list[str]:
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
    ]
    return base


def _usecols_encounter_enriched() -> list[str]:
    return [c for c in _usecols_event_enriched() if c != "event_grain"]


def load_encounters(
    path: Path,
    *,
    max_rows: int | None,
    table_kind: str,
) -> pd.DataFrame:
    """Load encounter rows from gzipped CSV."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        sz_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Reading %s (%.1f MiB); table_kind=%s", path, sz_mb, table_kind)
    except OSError:
        logger.info("Reading %s; table_kind=%s", path, table_kind)

    if max_rows is not None:
        logger.info("Row cap: max_rows=%s (after ENCOUNTER filter for event_enriched)", max_rows)

    if table_kind == "event_enriched":
        usecols = _usecols_event_enriched()
    else:
        usecols = _usecols_encounter_enriched()

    read_kw: dict = {
        "filepath_or_buffer": path,
        "usecols": usecols,
        "dtype": {
            "EncounterKey": str,
            "PatientDurableKey": str,
            "DepartmentKey": str,
        },
        "low_memory": False,
    }
    if max_rows is not None:
        read_kw["nrows"] = max_rows * 3 if table_kind == "event_enriched" else max_rows

    chunks = []
    seen = 0
    for chunk in pd.read_csv(**read_kw, chunksize=500_000):
        chunks.append(chunk)
        seen += len(chunk)
        if max_rows and seen >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True) 

    if table_kind == "event_enriched":
        df = df.loc[df["event_grain"].astype(str).str.upper().eq("ENCOUNTER")].drop(
            columns=["event_grain"]
        )
        if max_rows is not None and len(df) > max_rows:
            df = df.iloc[:max_rows].copy()

    df["EncounterKey"] = df["EncounterKey"].astype(str)
    df["PatientDurableKey"] = df["PatientDurableKey"].astype(str)

    # Parse datetimes: combined ordering time prefers AdmissionInstant when present.
    inst = pd.to_datetime(df["AdmissionInstant"], errors="coerce")
    day = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    if day.isna().all():
        day = pd.to_datetime(df["Date"], errors="coerce")
    df["event_datetime"] = inst.fillna(day)
    df["calendar_date"] = df["event_datetime"].dt.normalize()
    n_time_fallback = int(inst.isna().sum())

    df.sort_values(
        ["PatientDurableKey", "event_datetime", "EncounterKey"],
        kind="mergesort",
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    n_pat = df["PatientDurableKey"].nunique()
    t_min = df["event_datetime"].min()
    t_max = df["event_datetime"].max()
    logger.info(
        "Loaded %s encounter rows (%s patients); calendar span %s → %s",
        f"{len(df):,}",
        f"{n_pat:,}",
        t_min,
        t_max,
    )
    logger.debug(
        "Time parsing: %s rows used Date-only fallback (no parseable AdmissionInstant)",
        f"{n_time_fallback:,}",
    )
    return df


def add_next_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per patient, shift core fields to obtain next encounter."""
    out = df.copy()
    g = out.groupby("PatientDurableKey", sort=False)
    shift_cols = [
        ("event_datetime", "next_event_datetime"),
        ("calendar_date", "next_calendar_date"),
        ("DepartmentKey", "next_department_key"),
        ("DepartmentType", "next_department_type"),
        ("DepartmentSpecialty", "next_department_specialty"),
        ("DiagnosisValue", "next_diagnosis_value"),
        ("GroupCode", "next_group_code"),
        ("GroupName", "next_group_name"),
        ("Type", "next_type"),
        ("VisitTypeDescription", "next_visit_type_description"),
    ]
    for cur, nxt in shift_cols:
        out[nxt] = g[cur].shift(-1)

    out["days_to_next"] = (
        out["next_event_datetime"] - out["event_datetime"]
    ).dt.total_seconds() / 86400.0
    # Integer days for alignment with challenge wording (also buckets short gaps).
    out["days_to_next_int"] = np.floor(out["days_to_next"]).astype("Int64")

    out["same_diag_value_next"] = out["DiagnosisValue"].eq(out["next_diagnosis_value"])
    out["same_group_next"] = out["GroupCode"].eq(out["next_group_code"])
    out["same_department_next"] = out["DepartmentKey"].eq(out["next_department_key"])
    out["same_department_type_next"] = out["DepartmentType"].eq(out["next_department_type"])
    out["same_specialty_next"] = out["DepartmentSpecialty"].eq(out["next_department_specialty"])

    out["GroupName_disp"] = out["GroupName"].fillna("(missing)").astype(str)
    out["DepartmentType_disp"] = out["DepartmentType"].fillna("(missing)").astype(str)
    out["next_department_type_disp"] = out["next_department_type"].fillna("(missing)").astype(str)
    return out


def filter_eda_rows(
    df: pd.DataFrame, max_gap_days: int | None
) -> pd.DataFrame:
    eda = df[df["next_event_datetime"].notna()].copy()
    eda = eda[eda["days_to_next_int"].notna()]
    eda = eda[eda["days_to_next_int"].astype(float) >= 0]
    if max_gap_days is not None:
        eda = eda[eda["days_to_next_int"].astype(float) <= max_gap_days]
    return eda


def plot_histogram_days_to_next(eda: pd.DataFrame, out: Path) -> None:
    x = eda["days_to_next_int"].astype(float)
    x = x[(x >= 0) & (x <= 365)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=60, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Days until next encounter")
    axes[0].set_ylabel("Encounters (with a known next)")
    axes[0].set_title("Time to next encounter (≤365d)")

    x_pos = x[x > 0]
    if len(x_pos):
        axes[1].hist(np.log10(x_pos.clip(lower=1)), bins=50, color="darkorange", edgecolor="white", linewidth=0.5)
        axes[1].set_xlabel("log10(days to next), days ≥1")
        axes[1].set_ylabel("Encounters")
        axes[1].set_title("Same distribution on log scale (days≥1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_median_gap_by_group(eda: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    vc = eda["GroupName_disp"].value_counts()
    top = vc.head(top_n).index
    sub = eda[eda["GroupName_disp"].isin(top)]
    gap_med = sub.groupby("GroupName_disp", observed=False)["days_to_next_int"].median().sort_values()

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    gap_med.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Median days until next encounter")
    ax.set_ylabel("Diagnosis group (top by volume)")
    ax.set_title(f"Median gap by diagnosis group (top {top_n} by encounter count)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_survival_style_returns(eda: pd.DataFrame, out: Path, top_n: int = 10) -> None:
    eda = eda[
        ~eda["GroupName_disp"].astype(str).str.lower().str.contains("missing")
    ]
    vc = eda["GroupName_disp"].value_counts()
    top = vc.head(top_n).index
    sub = eda[eda["GroupName_disp"].isin(top)]

    horizons = [7, 14, 30, 90]
    rates = []
    for gname in top:
        rows = sub.loc[sub["GroupName_disp"] == gname, "days_to_next_int"].astype(float)
        row = {"GroupName_disp": gname}
        for h in horizons:
            row[f"pct_within_{h}d"] = float((rows <= h).mean()) if len(rows) else np.nan
        rates.append(row)
    rate_df = pd.DataFrame(rates)
    rate_df["GroupName_disp"] = rate_df["GroupName_disp"].apply(lambda x: str(x).split(",")[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rate_df))
    w = 0.15
    for i, h in enumerate(horizons):
        ax.bar(x + i * w, rate_df[f"pct_within_{h}d"], width=w, label=f"≤{h}d")

    ax.set_xticks(x + w * (len(horizons) / 2 - 0.5))
    ax.set_xticklabels(rate_df["GroupName_disp"], rotation=35, ha="right")
    ax.set_ylabel("Share of encounters")
    ax.set_title("Short-horizon return rates by diagnosis group (top by volume)")
    ax.legend(
        title="Return window",
        loc="center right",
        bbox_to_anchor=(-0.25, 0.5)
    )
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_department_transition_heatmap(eda: pd.DataFrame, out: Path) -> None:
    trans = eda.dropna(subset=["DepartmentType_disp", "next_department_type_disp"])
    if len(trans) < 100:
        logger.warning(
            "Skipping department-type heatmap: only %s transition rows after dropna (need ≥100)",
            len(trans),
        )
        return
    mat = pd.crosstab(
        trans["DepartmentType_disp"],
        trans["next_department_type_disp"],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    if _HAS_SNS:
        sns.heatmap(mat, cmap="Blues", ax=ax, cbar_kws={"label": "P(next | current)"})
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="P(next | current)")
    ax.set_xlabel("Next department type")
    ax.set_ylabel("Current department type")
    ax.set_title("Department type transition probabilities (row-normalized)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_diagnosis_transition_heatmap(eda: pd.DataFrame, out: Path, top_n: int = 35) -> None:
    vc = eda["GroupName_disp"].value_counts()
    top = vc.head(top_n).index
    trans = eda[
        eda["GroupName_disp"].isin(top) & eda["next_group_name"].notna()
    ].copy()
    trans["next_group_disp"] = trans["next_group_name"].fillna("(missing)").astype(str)
    trans = trans[trans["next_group_disp"].isin(top)]
    if len(trans) < 100:
        logger.warning(
            "Skipping diagnosis-group heatmap: only %s transition rows in top groups (need ≥100)",
            len(trans),
        )
        return

    mat = pd.crosstab(trans["GroupName_disp"], trans["next_group_disp"], normalize="index")
    fig, ax = plt.subplots(figsize=(14, 12))
    if _HAS_SNS:
        sns.heatmap(mat, cmap="Purples", ax=ax, cbar_kws={"label": "P(next group | current)"})
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap="Purples")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=6)
        fig.colorbar(im, ax=ax, label="P(next | current)")
    ax.set_xlabel("Next diagnosis group")
    ax.set_ylabel("Current diagnosis group")
    ax.set_title(f"Diagnosis group transitions (top {top_n} groups, row-normalized)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_diagnosis_persistence(eda: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    vc = eda["GroupName_disp"].value_counts()
    top = vc.head(top_n).index
    sub = eda[eda["GroupName_disp"].isin(top)]
    persist = (
        sub.groupby("GroupName_disp", observed=False)["same_group_next"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    persist.plot(kind="barh", ax=ax, color="slateblue")
    ax.set_xlabel("P(same GroupCode at next encounter)")
    ax.set_ylabel("Current diagnosis group")
    ax.set_title("Diagnosis group persistence (consecutive encounters)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sdoh_transport_comparison(eda: pd.DataFrame, out: Path) -> None:
    if TRANSPORT_COL not in eda.columns:
        logger.warning("Skipping SDOH transport plot: column %s not in frame", TRANSPORT_COL)
        return
    sub = eda.copy()
    sub["_tr"] = sub[TRANSPORT_COL].fillna(0).astype(int)
    # Compare only rows where SDOH question was observed vs not — optional secondary split
    parts = []
    for label, mask in [
        ("Transport SDOH observed", sub["_tr"] == 1),
        ("Not flagged transport (incl. no SDOH)", sub["_tr"] == 0),
    ]:
        s = sub.loc[mask, "days_to_next_int"].astype(float)
        s = s[(s >= 0) & (s <= 365)]
        parts.append((label, s))

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, s in parts:
        if len(s) > 0:
            ax.hist(
                s,
                bins=45,
                alpha=0.45,
                label=f"{label} (n={len(s):,})",
                density=True,
            )
    ax.set_xlabel("Days to next encounter (≤365)")
    ax.set_ylabel("Density")
    ax.set_title("Visit gaps: transportation needs SDOH flag vs not")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_patient_clusters(
    df_enc: pd.DataFrame,
    eda: pd.DataFrame,
    out: Path,
    random_state: int = 42,
) -> None:
    if not _HAS_SKLEARN:
        return

    try:
        _plot_patient_clusters_impl(df_enc, eda, out, random_state=random_state)
    except Exception as exc:  # noqa: BLE001 — sklearn/threadpool can fail in odd environments
        logger.warning("Skipping patient clusters: %s", exc)


def _plot_patient_clusters_impl(
    df_enc: pd.DataFrame,
    eda: pd.DataFrame,
    out: Path,
    random_state: int = 42,
) -> None:
    same_dt = eda.groupby("PatientDurableKey", observed=False)["same_department_type_next"].mean()
    same_dt.name = "pct_same_dept_type_next"

    enc = df_enc.copy()
    enc.sort_values(["PatientDurableKey", "event_datetime", "EncounterKey"], inplace=True)
    enc["gap_days"] = enc.groupby("PatientDurableKey")["event_datetime"].diff().dt.total_seconds() / 86400.0

    def _flag(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower().str.strip()
        return s.isin(["1", "true", "t", "yes", "y"])

    agg_kw: dict = dict(
        n_encounters=("EncounterKey", "count"),
        median_gap_days=("gap_days", "median"),
        max_gap_days=("gap_days", "max"),
        n_unique_groups=("GroupName", pd.Series.nunique),
        n_unique_dept_types=("DepartmentType", pd.Series.nunique),
        pct_ed=("IsEdVisit", lambda s: float(_flag(s).mean()) if len(s) else 0.0),
        pct_hosp=("IsHospitalAdmission", lambda s: float(_flag(s).mean()) if len(s) else 0.0),
    )
    if TRANSPORT_COL in enc.columns:
        agg_kw["sdoh_transport_max"] = (TRANSPORT_COL, "max")
    agg = enc.groupby("PatientDurableKey").agg(**agg_kw)
    if TRANSPORT_COL not in enc.columns:
        agg["sdoh_transport_max"] = 0.0

    feat = agg.join(same_dt, how="left")
    feat["pct_same_dept_type_next"] = feat["pct_same_dept_type_next"].fillna(0)

    feat = feat.dropna(subset=["median_gap_days", "max_gap_days"])
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    if len(feat) < 200:
        logger.warning(
            "Skipping patient clusters: only %s patients with gap features (need ≥200)",
            len(feat),
        )
        return

    use_cols = [
        "n_encounters",
        "median_gap_days",
        "max_gap_days",
        "n_unique_groups",
        "n_unique_dept_types",
        "pct_ed",
        "pct_hosp",
        "pct_same_dept_type_next",
        "sdoh_transport_max",
    ]
    X = feat[use_cols].astype(float).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    k = max(2, min(6, len(feat) // 500))
    k = min(k, len(feat) - 1)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(Xs)
    feat["cluster"] = labels

    xy = PCA(n_components=2, random_state=random_state).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(9, 6))
    for cl in sorted(feat["cluster"].unique()):
        m = feat["cluster"].values == cl
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            label=str(int(cl)),
            alpha=0.35,
            s=18,
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Patient journey clusters (PCA of engineered features; point size fixed)")
    ax.legend(title="cluster", loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    feat_out = out.parent / f"{out.stem}_cluster_assignments.csv"
    feat.to_csv(feat_out, float_format="%.4f")
    logger.info(
        "Patient clustering: k=%s clusters, %s patients → %s + %s",
        k,
        f"{len(feat):,}",
        out,
        feat_out,
    )


def write_repeat_location_summary(eda: pd.DataFrame, out_csv: Path) -> None:
    rows = [
        ("same_department_next", float(eda["same_department_next"].mean())),
        ("same_department_type_next", float(eda["same_department_type_next"].mean())),
        ("same_specialty_next", float(eda["same_specialty_next"].mean())),
    ]
    pd.DataFrame(rows, columns=["metric", "rate"]).to_csv(out_csv, index=False)


def write_gap_by_group_csv(eda: pd.DataFrame, out_csv: Path, top_n: int = 50) -> None:
    vc = eda["GroupName_disp"].value_counts()
    top = vc.head(top_n).index
    sub = eda[eda["GroupName_disp"].isin(top)]
    g = sub.groupby("GroupName_disp", observed=False)["days_to_next_int"].agg(["median", "mean", "count"])
    g.to_csv(out_csv)


def run_all(args: argparse.Namespace) -> None:
    t_run = time.perf_counter()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting journey EDA → output_dir=%s input=%s table_kind=%s max_rows=%s max_gap_days=%s",
        out_dir,
        Path(args.input).resolve(),
        args.table_kind,
        args.max_rows,
        args.max_gap_days,
    )
    logger.debug(
        "Optional libs: seaborn=%s scikit-learn=%s",
        _HAS_SNS,
        _HAS_SKLEARN,
    )

    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="notebook")

    t0 = time.perf_counter()
    df = load_encounters(
        Path(args.input),
        max_rows=args.max_rows,
        table_kind=args.table_kind,
    )
    logger.info("CSV load + sort finished in %.2fs", time.perf_counter() - t0)

    t1 = time.perf_counter()
    df = add_next_event_columns(df)
    logger.info(
        "Built next-event columns in %.2fs (with-next density=%.4f)",
        time.perf_counter() - t1,
        df["next_event_datetime"].notna().mean(),
    )

    t2 = time.perf_counter()
    eda = filter_eda_rows(df, args.max_gap_days)
    logger.info(
        "Filtered EDA frame in %.2fs → %s linked pairs (%s patients)",
        time.perf_counter() - t2,
        f"{len(eda):,}",
        f"{eda['PatientDurableKey'].nunique():,}",
    )

    if args.max_gap_days is not None:
        logger.info("Gap filter: retaining pairs with 0 ≤ days_to_next ≤ %s", args.max_gap_days)
    else:
        logger.info("Gap filter: upper bound disabled (--no-gap-cap)")

    dropped_last = len(df) - len(eda)
    logger.debug(
        "Rows excluded from EDA slice vs loaded encounters: %s (typically last visit per patient + invalid gaps)",
        f"{dropped_last:,}",
    )

    if len(eda) > 0:
        gtd = eda["days_to_next_int"].astype(float)
        logger.info(
            "Gap distribution (days_to_next): min=%s median=%s mean=%s p90=%s max=%s",
            int(gtd.min()),
            gtd.median(),
            round(float(gtd.mean()), 2),
            gtd.quantile(0.9),
            int(gtd.max()),
        )

    steps = [
        ("01_days_to_next_histogram.png", lambda: plot_histogram_days_to_next(eda, out_dir / "01_days_to_next_histogram.png")),
        ("02_median_gap_top_diagnosis_groups.png", lambda: plot_median_gap_by_group(eda, out_dir / "02_median_gap_top_diagnosis_groups.png")),
        ("03_return_within_7_14_30_90_by_group.png", lambda: plot_survival_style_returns(eda, out_dir / "03_return_within_7_14_30_90_by_group.png")),
        ("04_department_type_transition_heatmap.png", lambda: plot_department_transition_heatmap(eda, out_dir / "04_department_type_transition_heatmap.png")),
        ("05_diagnosis_group_transition_heatmap.png", lambda: plot_diagnosis_transition_heatmap(eda, out_dir / "05_diagnosis_group_transition_heatmap.png")),
        ("06_diagnosis_group_persistence.png", lambda: plot_diagnosis_persistence(eda, out_dir / "06_diagnosis_group_persistence.png")),
        ("07_sdoh_transport_vs_visit_gap.png", lambda: plot_sdoh_transport_comparison(eda, out_dir / "07_sdoh_transport_vs_visit_gap.png")),
    ]

    for name, fn in steps:
        t_step = time.perf_counter()
        path = out_dir / name
        fn()
        elapsed = time.perf_counter() - t_step
        if path.is_file():
            logger.info("Wrote %s (%.2fs)", name, elapsed)
        else:
            logger.warning("Did not produce %s — plot skipped or failed (%.2fs)", name, elapsed)

    t_csv = time.perf_counter()
    write_repeat_location_summary(eda, out_dir / "summary_repeat_location_rates.csv")
    logger.info("Wrote %-45s %.3fs", "summary_repeat_location_rates.csv", time.perf_counter() - t_csv)
    t_csv = time.perf_counter()
    write_gap_by_group_csv(eda, out_dir / "summary_median_gap_by_diagnosis_group.csv")
    logger.info("Wrote %-45s %.3fs", "summary_median_gap_by_diagnosis_group.csv", time.perf_counter() - t_csv)

    cluster_out = out_dir / "08_patient_journey_clusters.png"
    t_cl = time.perf_counter()
    plot_patient_clusters(df, eda, cluster_out)
    logger.info("Patient clustering step finished in %.2fs", time.perf_counter() - t_cl)

    # Stratified quick views (Encounter Type)
    t_facet = time.perf_counter()
    type_counts = eda["Type"].fillna("(missing)").astype(str).value_counts().head(6)
    logger.debug("Top encounter Types for facet plot: %s", dict(type_counts.head(6)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (etype, _) in zip(axes, type_counts.items()):
        sub = eda.loc[eda["Type"].fillna("(missing)").astype(str).eq(etype), "days_to_next_int"].astype(float)
        sub = sub[(sub >= 0) & (sub <= 365)]
        ax.hist(sub, bins=40, color="gray", edgecolor="white", linewidth=0.3)
        ax.set_title(str(etype)[:40])
        ax.set_xlabel("Days")
    fig.suptitle("Days to next encounter by encounter Type (top categories)")
    fig.tight_layout()
    facet_path = out_dir / "09_gap_by_encounter_type_facets.png"
    fig.savefig(facet_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %-45s %.2fs", facet_path.name, time.perf_counter() - t_facet)

    logger.info(
        "Done in %.2fs total — artifacts under %s",
        time.perf_counter() - t_run,
        out_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patient journey EDA for next-event motivation.")
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data/processed/event_enriched.csv",
        help="Encounter-level CSV.gz (event_enriched or encounter_enriched export).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "visuals/eda",
        help="Directory for PNG + CSV outputs.",
    )
    p.add_argument(
        "--table-kind",
        choices=("event_enriched", "encounter_enriched"),
        default="event_enriched",
        help="event_enriched: filter ENCOUNTER rows; encounter_enriched: one row per encounter already.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on encounter rows after filtering (for faster iteration).",
    )
    p.add_argument(
        "--max-gap-days",
        type=int,
        default=365,
        help="Clip linked pairs to this gap for plots (ignored if --no-gap-cap).",
    )
    p.add_argument(
        "--no-gap-cap",
        action="store_true",
        help="Do not upper-bound days_to_next (use full observed spacing).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console logging verbosity (default INFO).",
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
