#!/usr/bin/env python3
"""
Empirical top-1 / top-5 baseline accuracy for next-event labels (linked pairs only).

Uses the same encounter pipeline as journey_eda: current row predicts next_* targets.

Denominator: linked encounters only (has observed next in extract).

Example:
  python eda/next_event_baselines.py --input data/processed/event_enriched.csv.gz \\
    --output-dir visuals/eda_story
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eda.gap_bins import days_to_gap_label
from eda.journey_eda import (
    add_next_event_columns,
    configure_logging,
    filter_eda_rows,
    load_encounters,
)

logger = logging.getLogger(__name__)


CAPTION = (
    "Denominator: linked encounters only (counts n under each label — pairs with non-missing "
    "current and next for that field). Metrics are empirical conditional frequencies, not model accuracy."
)


def _topk_accuracy(y_true: pd.Series, y_pred_matrix: np.ndarray, k: int) -> float:
    """y_pred_matrix shape (n_samples, k) with candidate labels per row."""
    ok = 0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i, yt in enumerate(y_true.astype(str).values):
        preds = [str(y_pred_matrix[i, j]) for j in range(min(k, y_pred_matrix.shape[1]))]
        if str(yt) in preds:
            ok += 1
    return ok / n


def _baseline_global(series: pd.Series) -> str:
    vc = series.astype(str).value_counts()
    return str(vc.index[0]) if len(vc) else ""


def _baseline_conditional(
    cur: pd.Series,
    nxt: pd.Series,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Per current value: mode and top-5 list for next."""
    df = pd.DataFrame({"c": cur.astype(str), "n": nxt.astype(str)})
    mode_map: dict[str, str] = {}
    top5_map: dict[str, list[str]] = {}
    for g, sub in df.groupby("c"):
        vc = sub["n"].value_counts()
        if len(vc) == 0:
            continue
        mode_map[g] = str(vc.index[0])
        top5_map[g] = [str(x) for x in vc.head(5).index.tolist()]
    return mode_map, top5_map


def _norm_token(v: object, max_len: int = 80) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "UNKNOWN"
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return "UNKNOWN"
    return s[:max_len]


def _trueish(v: object) -> bool:
    t = _norm_token(v, 40).upper()
    return t in {"1", "TRUE", "T", "YES", "Y"}


def _volume_bin(n: object) -> str:
    try:
        x = int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else None
    except (TypeError, ValueError):
        return "UNKNOWN"
    if x is None:
        return "UNKNOWN"
    if x < 100:
        return "VERY_LOW"
    if x < 1000:
        return "LOW"
    if x < 10000:
        return "MID"
    if x < 100000:
        return "HIGH"
    return "VERY_HIGH"


def _region_labels(eda: pd.DataFrame) -> pd.Series:
    county_raw = pd.Series(np.nan, index=eda.index, dtype=object)
    if "department_County" in eda.columns:
        county_raw = eda["department_County"]
    if "County" in eda.columns:
        county_raw = county_raw.fillna(eda["County"])
    county_s = county_raw.map(lambda x: _norm_token(x, 60))
    city_s = (
        eda["department_City"].map(lambda x: _norm_token(x, 60))
        if "department_City" in eda.columns
        else pd.Series("UNKNOWN", index=eda.index)
    )
    postal_s = (
        eda["department_PostalCode"].map(lambda x: _norm_token(x, 20))
        if "department_PostalCode" in eda.columns
        else pd.Series("UNKNOWN", index=eda.index)
    )
    result = pd.Series("UNKNOWN", index=eda.index, dtype=object)
    m_county = county_s != "UNKNOWN"
    result[m_county] = "COUNTY_" + county_s[m_county].astype(str)
    m_city = (result == "UNKNOWN") & (city_s != "UNKNOWN")
    result[m_city] = "CITY_" + city_s[m_city].astype(str)
    m_zip = (result == "UNKNOWN") & (postal_s != "UNKNOWN")
    result[m_zip] = "ZIP_" + postal_s[m_zip].astype(str)
    return result


def _setting_labels(eda: pd.DataFrame) -> pd.Series:
    def col(name: str) -> pd.Series:
        return eda[name] if name in eda.columns else pd.Series(False, index=eda.index)

    ed_vis = col("IsEdVisit").map(_trueish)
    inp = col("IsInpatientAdmission").map(_trueish)
    hadm = col("IsHospitalAdmission").map(_trueish)
    obs = col("IsObservation").map(_trueish)
    hop = col("IsHospitalOutpatientVisit").map(_trueish)
    opf = col("IsOutpatientFaceToFaceVisit").map(_trueish)
    return pd.Series(
        np.select(
            [ed_vis, inp, hadm, obs, hop, opf],
            ["ED", "INPATIENT", "HOSP_ADMIT", "OBS", "HOSP_OP", "OP_FACE"],
            default="NONE",
        ),
        index=eda.index,
    )


def _event_description_labels(eda: pd.DataFrame) -> pd.Series:
    vt = (
        eda["VisitTypeDescription"].fillna("(missing)").astype(str)
        if "VisitTypeDescription" in eda.columns
        else pd.Series("(missing)", index=eda.index)
    )
    if "event_description" not in eda.columns:
        return vt
    ed = eda["event_description"]
    mask = ed.notna() & (ed.astype(str).str.strip() != "") & (ed.astype(str).str.lower() != "nan")
    return pd.Series(np.where(mask, ed.astype(str), vt), index=eda.index)


def _incoming_gap_labels(eda: pd.DataFrame) -> pd.Series:
    g = eda.groupby("PatientDurableKey", sort=False)
    prev_dt = g["event_datetime"].shift(1)
    floor_days = np.floor((eda["event_datetime"] - prev_dt).dt.total_seconds() / 86400.0)
    enc_idx = g.cumcount()
    out: list[str] = []
    for i in range(len(eda)):
        if enc_idx.iloc[i] == 0 or pd.isna(prev_dt.iloc[i]):
            out.append("START")
        else:
            fd = floor_days.iloc[i]
            out.append(days_to_gap_label(int(fd)) if pd.notna(fd) else "UNKNOWN")
    return pd.Series(out, index=eda.index, dtype=object)


def evaluate_task(
    name: str,
    cur: pd.Series,
    nxt: pd.Series,
) -> dict[str, float]:
    mask = nxt.notna() & cur.notna()
    cur = cur[mask].astype(str)
    nxt = nxt[mask].astype(str)
    n = len(cur)
    if n == 0:
        return {"task": name, "n": 0, "top1_conditional": np.nan, "top5_conditional": np.nan}

    global_mode = _baseline_global(nxt)

    mode_map, top5_map = _baseline_conditional(cur, nxt)
    pred1 = cur.map(lambda x: mode_map.get(x, global_mode))
    top1_c = float((nxt.values == pred1.values).mean())

    top5_rows = []
    for i in range(n):
        c = cur.iloc[i]
        top5 = top5_map.get(c, [global_mode])
        while len(top5) < 5:
            top5 = top5 + [top5[-1]]
        top5_rows.append(top5[:5])
    mat = np.array(top5_rows)
    top5_c = _topk_accuracy(nxt.reset_index(drop=True), mat, 5)

    return {
        "task": name,
        "n": n,
        "top1_conditional": top1_c,
        "top5_conditional": top5_c,
    }


def _disp(s: pd.Series) -> pd.Series:
    return s.fillna("(missing)").astype(str)


def main() -> None:
    p = argparse.ArgumentParser(description="Empirical next-event baseline accuracies.")
    p.add_argument("--input", type=Path, default=_ROOT / "data/processed/event_enriched.csv")
    p.add_argument("--output-dir", type=Path, default=_ROOT / "visuals/eda_story")
    p.add_argument("--table-kind", choices=("event_enriched", "encounter_enriched"), default="event_enriched")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-gap-days", type=int, default=365)
    p.add_argument("--no-gap-cap", action="store_true")
    args = p.parse_args()
    configure_logging(logging.INFO)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_encounters(args.input, max_rows=args.max_rows, table_kind=args.table_kind)
    df = add_next_event_columns(df)
    max_gap = None if args.no_gap_cap else args.max_gap_days
    eda = filter_eda_rows(df, max_gap)

    eda = eda.copy()
    eda["gap_label"] = eda["days_to_next_int"].map(days_to_gap_label)

    # Align with factorized heads: coalesce event_description with VisitTypeDescription when needed.
    eda["event_desc_cur"] = _event_description_labels(eda)
    gpat = eda.groupby("PatientDurableKey", sort=False)
    eda["event_desc_next"] = gpat["event_desc_cur"].shift(-1)

    eda["incoming_gap_label"] = _incoming_gap_labels(eda)

    counts = eda.groupby("DepartmentKey", observed=False).size()
    eda["facility_size_cur"] = eda["DepartmentKey"].map(lambda k: _volume_bin(counts.get(k, np.nan)))
    eda["facility_size_next"] = eda.groupby("PatientDurableKey", sort=False)["facility_size_cur"].shift(-1)

    eda["region_cur"] = _region_labels(eda)
    eda["region_next"] = eda.groupby("PatientDurableKey", sort=False)["region_cur"].shift(-1)

    eda["setting_cur"] = _setting_labels(eda)
    eda["setting_next"] = eda.groupby("PatientDurableKey", sort=False)["setting_cur"].shift(-1)

    eda["DepartmentSpecialty_s"] = eda["DepartmentSpecialty"].fillna("(missing)").astype(str)
    eda["next_department_specialty_s"] = eda["next_department_specialty"].fillna("(missing)").astype(str)

    rows = [
        evaluate_task("dept_specialty", eda["DepartmentSpecialty_s"], eda["next_department_specialty_s"]),
        evaluate_task("dept_type", _disp(eda["DepartmentType"]), _disp(eda["next_department_type"])),
        evaluate_task("diagnosis_value", _disp(eda["DiagnosisValue"]), _disp(eda["next_diagnosis_value"])),
        evaluate_task("event_description", eda["event_desc_cur"], eda["event_desc_next"]),
        evaluate_task("facility_size", eda["facility_size_cur"], eda["facility_size_next"]),
        evaluate_task("gap", eda["incoming_gap_label"].astype(str), eda["gap_label"].astype(str)),
        evaluate_task("group_code", _disp(eda["GroupCode"]), _disp(eda["next_group_code"])),
        evaluate_task("region", eda["region_cur"].astype(str), eda["region_next"].astype(str)),
        evaluate_task("setting", eda["setting_cur"].astype(str), eda["setting_next"].astype(str)),
        evaluate_task("type", _disp(eda["Type"]), _disp(eda["next_type"])),
    ]

    out_csv = out_dir / "baseline_topk_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    try:
        import matplotlib.pyplot as plt

        n_tasks = len(rows)
        fig_w = max(10.0, 0.72 * n_tasks)
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        tasks = [r["task"] for r in rows]
        x = np.arange(n_tasks)
        w = 0.35
        ax.bar(x - w / 2, [r["top1_conditional"] for r in rows], width=w, label="top-1 given current")
        ax.bar(x + w / 2, [r["top5_conditional"] for r in rows], width=w, label="top-5 given current")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=22, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Empirical baseline predictability (conditional frequencies)")
        ax.legend(fontsize=9)
        fig.text(0.5, 0.01, CAPTION, ha="center", fontsize=8)
        fig.subplots_adjust(bottom=0.28)
        fig_path = out_dir / "baseline_topk_accuracy.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s", fig_path)
    except Exception as exc:
        logger.warning("Could not render baseline chart: %s", exc)


if __name__ == "__main__":
    main()
