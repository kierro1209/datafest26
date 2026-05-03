#!/usr/bin/env python3
"""
Regenerate resource-pressure figures from existing CSVs only (no encounter reload).

Reads summaries written by resource_pressure_eda.py under --csv-dir and writes PNGs to --output-dir.

Rows whose specialty or diagnosis label is ``(missing)``, blank, or other placeholders are dropped
before plotting; non-finite metrics are removed where relevant.

Example:
  python eda/resource_pressure_redisplay.py \\
    --csv-dir visuals/eda_resource --output-dir visuals/eda_resource
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Kept in sync with resource_pressure_eda.py (standalone import to avoid loading full pipeline).
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


def _csv_dimension_ok(series: pd.Series) -> pd.Series:
    """Exclude (missing), blank, nan-string, and other placeholder labels."""
    tok = series.map(_normalized_lower_token)
    return ~tok.isin(_BAD_DIMENSION_LABELS)


def _clean_monthly_stress(df: pd.DataFrame) -> pd.DataFrame:
    if "DepartmentSpecialty_disp" not in df.columns:
        return df
    out = df.loc[_csv_dimension_ok(df["DepartmentSpecialty_disp"])].copy()
    if "encounters" in out.columns:
        out = out[out["encounters"].notna() & (out["encounters"] >= 0)]
    if "providers" in out.columns:
        out = out[out["providers"].notna() & (out["providers"] >= 0)]
    return out


def _clean_diag(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or "GroupName_disp" not in df.columns:
        return df
    out = df.loc[_csv_dimension_ok(df["GroupName_disp"])].copy()
    r = pd.to_numeric(out.get("return_30d"), errors="coerce")
    e = pd.to_numeric(out.get("encounters"), errors="coerce")
    ok = r.notna() & e.notna() & np.isfinite(r.to_numpy()) & np.isfinite(e.to_numpy())
    return out.loc[ok].copy()


def _clean_diag_provider(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    ok = _csv_dimension_ok(df["GroupName_disp"]) & _csv_dimension_ok(df["ProviderSpecialty"])
    out = df.loc[ok].copy()
    if "encounters_per_provider" in out.columns:
        out = out[np.isfinite(pd.to_numeric(out["encounters_per_provider"], errors="coerce"))]
    return out


def _clean_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if "DepartmentSpecialty_disp" not in df.columns:
        return df
    out = df.loc[_csv_dimension_ok(df["DepartmentSpecialty_disp"])].copy()
    if "demand_z" in out.columns:
        out = out[np.isfinite(pd.to_numeric(out["demand_z"], errors="coerce"))]
    return out


def _clean_sdoh(df: pd.DataFrame) -> pd.DataFrame:
    if "GroupName_disp" not in df.columns:
        return df
    out = df.loc[_csv_dimension_ok(df["GroupName_disp"])].copy()
    if "return_30d" in out.columns:
        out = out[np.isfinite(pd.to_numeric(out["return_30d"], errors="coerce"))]
    return out


def _setup_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "axes.labelweight": "medium",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.title_fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def _comma_int(x: float, pos: int | None = None) -> str:
    if np.isnan(x):
        return ""
    return f"{x:,.0f}"


def _pct_axis(ax, axis: str = "x") -> None:
    fmt = mticker.PercentFormatter(xmax=1.0, decimals=0)
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)


def _truncate(s: str, n: int = 36) -> str:
    s = str(s).strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def plot_01_monthly_lines(monthly: pd.DataFrame, out: Path, *, top_n: int = 15) -> None:
    monthly = monthly.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    vc = monthly.groupby("DepartmentSpecialty_disp")["encounters"].sum().sort_values(ascending=False)
    keep = list(vc.head(top_n).index)
    sub = monthly[monthly["DepartmentSpecialty_disp"].isin(keep)]

    fig, ax = plt.subplots(figsize=(12, 6))
    n_lines = max(len(keep), 1)
    cmap = plt.cm.get_cmap("tab20", n_lines)
    for i, spec in enumerate(sorted(keep, key=lambda s: vc.get(s, 0), reverse=True)):
        m = sub.loc[sub["DepartmentSpecialty_disp"] == spec].sort_values("month")
        ax.plot(
            m["month"],
            m["encounters_per_provider"],
            marker="o",
            ms=2.5,
            lw=1.4,
            color=cmap(i % n_lines),
            label=_truncate(spec, 42),
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Encounters per distinct attending provider")
    ax.set_title(
        "Observed load: monthly encounters per distinct attending provider\n"
        f"(top {top_n} department specialties by total encounter volume)"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_int))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    leg = ax.legend(
        title="Department specialty",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        ncol=1,
    )
    leg.get_title().set_fontweight("semibold")
    fig.text(
        0.01,
        0.01,
        "Proxy only: distinct providers in data, not staffing capacity. "
        "Rows with unknown department specialty omitted.",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.12, right=0.72)
    fig.savefig(out)
    plt.close(fig)


def plot_02_scatter(monthly: pd.DataFrame, out: Path, *, top_n: int = 12) -> None:
    monthly = monthly.copy()
    vc = monthly.groupby("DepartmentSpecialty_disp")["encounters"].sum().sort_values(ascending=False)
    keep = list(vc.head(top_n).index)
    sub = monthly[monthly["DepartmentSpecialty_disp"].isin(keep)].copy()
    sub["Specialty (short)"] = sub["DepartmentSpecialty_disp"].map(lambda x: _truncate(x, 34))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    if _HAS_SNS:
        sns.scatterplot(
            data=sub,
            x="encounters",
            y="providers",
            hue="Specialty (short)",
            size="patients",
            sizes=(30, 420),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
            palette="tab10",
        )
        leg = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
        )
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(8)
        for t in leg.get_texts():
            t.set_fontsize(7)
    else:
        for i, spec in enumerate(keep):
            ms = sub.loc[sub["DepartmentSpecialty_disp"] == spec]
            ax.scatter(ms["encounters"], ms["providers"], alpha=0.65, label=_truncate(spec, 30))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

    ax.set_xlabel("Encounters this month (distinct encounter IDs)")
    ax.set_ylabel("Distinct attending providers (observed)")
    ax.set_title("Monthly demand vs observed provider count by specialty")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_comma_int))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_int))
    fig.subplots_adjust(right=0.68)
    fig.savefig(out)
    plt.close(fig)


def plot_03_diag_scatter(diag: pd.DataFrame, out: Path, *, label_top: int = 20, max_points: int = 80) -> None:
    if len(diag) == 0:
        return
    dp = diag.sort_values("encounters", ascending=False).head(max_points).copy()
    sizes = (dp["patients"] / max(dp["patients"].max(), 1) * 320 + 18).clip(18, 320)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        dp["return_30d"],
        dp["encounters"],
        s=sizes,
        alpha=0.55,
        c="#2c5282",
        edgecolors="white",
        linewidths=0.35,
    )
    for _, row in dp.nlargest(label_top, "encounters").iterrows():
        ax.annotate(
            _truncate(row["GroupName_disp"], 40),
            (row["return_30d"], row["encounters"]),
            fontsize=7,
            alpha=0.9,
            xytext=(4, 3),
            textcoords="offset points",
            clip_on=True,
        )

    _pct_axis(ax, "x")
    ax.set_xlabel("Share of linked encounters with a next visit within 30 days")
    ax.set_ylabel("Encounter volume (distinct encounter IDs)")
    ax.set_title("Diagnosis groups: 30-day return rate vs volume\n(bubble area scales with distinct patients; labels on highest-volume groups)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_int))
    fig.savefig(out)
    plt.close(fig)


def plot_04_heatmap(stress: pd.DataFrame, out: Path, *, max_specialties: int = 35) -> None:
    if len(stress) == 0:
        return
    stress = stress.copy()
    stress["month"] = pd.to_datetime(stress["month"])
    vol = stress.groupby("DepartmentSpecialty_disp")["encounters"].sum()
    top_specs = vol.nlargest(max_specialties).index
    sub = stress[
        stress["DepartmentSpecialty_disp"].isin(top_specs)
        & ~stress["DepartmentSpecialty_disp"].astype(str).str.lower().str.contains("unspecified")
    ]

    pivot = sub.pivot_table(
        index="DepartmentSpecialty_disp",
        columns="month",
        values="stress_score",
        aggfunc="mean",
    )

    pivot = pivot[~pivot.index.str.lower().str.contains("unspecified")]

    w = max(11, min(28, 0.18 * pivot.shape[1] + 7))
    h = max(7, min(22, 0.22 * pivot.shape[0] + 4))
    fig, ax = plt.subplots(figsize=(w, h))
    x_labels = [pd.Timestamp(c).strftime("%b\n%Y") for c in pivot.columns]
    y_labels = [_truncate(str(i), 48) for i in pivot.index]

    if _HAS_SNS:
        sns.heatmap(
            pivot,
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.15,
            linecolor="#f5f5f5",
            cbar_kws={
                "label": "Stress score (rank-sum)\n↑ higher vs peers in grid",
                "shrink": 0.55,
            },
        )

        ax.set_xticklabels(x_labels, rotation=45, ha="center", fontsize=5)
        ax.set_yticklabels(y_labels, fontsize=7)
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(y_labels, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.55, label="Stress score")

    ax.set_xlabel("Month")
    ax.set_ylabel("Department specialty")
    ax.set_title(
        "Relative resource pressure by specialty and month\n"
        "(composite rank score; descriptive only; unknown specialties omitted)"
    )
    fig.savefig(out)
    plt.close(fig)


def plot_05_bottlenecks(diagp: pd.DataFrame, out: Path, *, top_n: int = 20) -> None:
    if len(diagp) == 0:
        return
    dp = diagp.dropna(subset=["encounters_per_provider"]).nlargest(top_n, "encounters_per_provider").copy()
    labels = [
        _truncate(str(gn), 30) + "  →  " + _truncate(str(ps), 30)
        for gn, ps in zip(dp["GroupName_disp"].astype(str), dp["ProviderSpecialty"].astype(str))
    ]

    fig, ax = plt.subplots(figsize=(11, max(5.5, top_n * 0.32)))
    y = np.arange(len(dp))
    ax.barh(y, dp["encounters_per_provider"].astype(float), color="#39587f", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Encounters per distinct attending provider (observed)")
    ax.set_title(
        f"Where diagnosis volume concentrates on few observed providers\n"
        f"(top {top_n} diagnosis × specialty pairs; CSV from pipeline uses department specialty "
        f"when attending was *Unspecified*)"
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.1f}"))
    fig.text(0.01, 0.01, "Higher values suggest a small observed provider pool per unit demand.", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.38, bottom=0.1)
    fig.savefig(out)
    plt.close(fig)


def plot_06_weekly_z(weekly: pd.DataFrame, out: Path, *, top_n: int = 8) -> None:
    if len(weekly) == 0:
        return
    weekly = weekly.copy()
    weekly["week"] = pd.to_datetime(weekly["week"])
    vol = weekly.groupby("DepartmentSpecialty_disp")["encounters"].sum()
    keep = list(vol.nlargest(top_n).index)
    sub = weekly[weekly["DepartmentSpecialty_disp"].isin(keep)].sort_values(
        ["DepartmentSpecialty_disp", "week"]
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    n_lines = max(len(keep), 1)
    cmap = plt.cm.get_cmap("tab10", n_lines)
    for i, spec in enumerate(sorted(keep, key=lambda s: vol.get(s, 0), reverse=True)):
        w = sub.loc[sub["DepartmentSpecialty_disp"] == spec]
        ax.plot(
            w["week"],
            w["demand_z"],
            lw=1.5,
            alpha=0.88,
            color=cmap(i % n_lines),
            label=_truncate(spec, 40),
        )

    ax.axhline(2.0, color="#c0392b", ls="--", lw=1.2, label="z = 2 (high vs own history)")
    ax.set_xlabel("Week (Monday week start)")
    ax.set_ylabel("Encounter count z-score within specialty")
    ax.set_title("Weekly demand variability: standardized encounter counts within each specialty")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        frameon=True,
        title="Specialty",
        title_fontsize=8,
    )
    fig.subplots_adjust(right=0.74)
    fig.savefig(out)
    plt.close(fig)


def plot_07_sdoh(sdoh: pd.DataFrame, out: Path, *, top_diagnoses: int = 15) -> None:
    if len(sdoh) == 0:
        return
    vol = sdoh.groupby("GroupName_disp")["encounters"].sum().nlargest(top_diagnoses)
    sub = sdoh[sdoh["GroupName_disp"].isin(vol.index)].copy()
    pivot = sub.pivot_table(
        index="GroupName_disp",
        columns="transportation_need_flag",
        values="return_30d",
        aggfunc="mean",
    )
    pivot = pivot.reindex(vol.index)

    col_map = {"flagged": "Transport need (flagged)", "not_flagged": "No transport flag"}
    pivot = pivot.rename(columns={k: col_map.get(k, k) for k in pivot.columns})

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#b8453b", "#2e6f95"][: pivot.shape[1]]
    pivot.plot(kind="bar", ax=ax, width=0.82, rot=28, color=colors, edgecolor="white")
    _pct_axis(ax, "y")
    ax.set_ylabel("Share with next visit within 30 days (linked pairs)")
    ax.set_xlabel("Diagnosis group (top by encounter volume among plotted)")
    ax.set_title("Near-term return rate: transportation SDOH screen vs not\n(same patient linked-encounter pairs)")
    ax.legend(title="SDOH transport screen", frameon=True, loc="upper right")
    ax.set_xticklabels([_truncate(l.get_text(), 28) for l in ax.get_xticklabels()])
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Redraw resource EDA figures from CSV summaries.")
    p.add_argument("--csv-dir", type=Path, default=_ROOT / "visuals/eda_resource")
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to --csv-dir.")
    p.add_argument("--top-specialties-lines", type=int, default=15)
    p.add_argument("--top-specialties-scatter", type=int, default=12)
    p.add_argument("--heatmap-rows", type=int, default=35)
    p.add_argument("--top-bottlenecks", type=int, default=20)
    p.add_argument("--top-weekly-lines", type=int, default=8)
    args = p.parse_args()

    csv_dir = args.csv_dir.expanduser().resolve()
    out_dir = (args.output_dir or csv_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_matplotlib_style()
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="notebook")

    monthly_path = csv_dir / "monthly_specialty_pressure.csv"
    stress_path = csv_dir / "monthly_specialty_stress.csv"
    diag_path = csv_dir / "diag_group_repeat_pressure.csv"
    diagp_path = csv_dir / "diag_by_provider_specialty.csv"
    weekly_path = csv_dir / "weekly_specialty_demand_z.csv"
    sdoh_path = csv_dir / "sdoh_transport_pressure_by_diag.csv"

    if monthly_path.is_file():
        monthly = _clean_monthly_stress(pd.read_csv(monthly_path))
        if len(monthly) > 0:
            plot_01_monthly_lines(
                monthly,
                out_dir / "01_monthly_encounters_per_provider_by_specialty.png",
                top_n=args.top_specialties_lines,
            )
            plot_02_scatter(
                monthly,
                out_dir / "02_monthly_encounters_vs_provider_count.png",
                top_n=args.top_specialties_scatter,
            )

    if stress_path.is_file():
        stress = _clean_monthly_stress(pd.read_csv(stress_path))
        if len(stress) > 0:
            plot_04_heatmap(
                stress,
                out_dir / "04_specialty_month_stress_heatmap.png",
                max_specialties=args.heatmap_rows,
            )

    if diag_path.is_file():
        diag = _clean_diag(pd.read_csv(diag_path))
        if len(diag) > 0:
            plot_03_diag_scatter(diag, out_dir / "03_diag_volume_vs_return_30d.png")

    if diagp_path.is_file():
        diagp = _clean_diag_provider(pd.read_csv(diagp_path))
        if len(diagp) > 0:
            plot_05_bottlenecks(
                diagp,
                out_dir / "05_top_diag_provider_specialty_bottlenecks.png",
                top_n=args.top_bottlenecks,
            )

    if weekly_path.is_file():
        weekly = _clean_weekly(pd.read_csv(weekly_path))
        if len(weekly) > 0:
            plot_06_weekly_z(
                weekly,
                out_dir / "06_weekly_demand_zscore_by_specialty.png",
                top_n=args.top_weekly_lines,
            )

    if sdoh_path.is_file():
        sdoh = _clean_sdoh(pd.read_csv(sdoh_path))
        if len(sdoh) > 0:
            plot_07_sdoh(sdoh, out_dir / "07_sdoh_transport_return30d_top_diagnoses.png")


if __name__ == "__main__":
    main()
