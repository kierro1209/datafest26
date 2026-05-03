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
    "Denominator: linked encounters only. Metrics are empirical conditional frequencies, "
    "not model accuracy."
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
        return {"task": name, "n": 0, "top1_global": np.nan, "top1_conditional": np.nan, "top5_conditional": np.nan}

    global_mode = _baseline_global(nxt)
    top1_g = float((nxt == global_mode).mean())

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
        "top1_global": top1_g,
        "top1_conditional": top1_c,
        "top5_conditional": top5_c,
    }


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
    eda["next_group_name_s"] = eda["next_group_name"].fillna("(missing)").astype(str)
    eda["DepartmentSpecialty_s"] = eda["DepartmentSpecialty"].fillna("(missing)").astype(str)
    eda["next_department_specialty_s"] = eda["next_department_specialty"].fillna("(missing)").astype(str)
    eda["gap_label"] = eda["days_to_next_int"].map(days_to_gap_label)

    rows = []
    rows.append(evaluate_task("next_diagnosis_group", eda["GroupName_disp"], eda["next_group_name_s"]))
    rows.append(
        evaluate_task(
            "next_department_specialty",
            eda["DepartmentSpecialty_s"],
            eda["next_department_specialty_s"],
        )
    )
    rows.append(
        evaluate_task(
            "next_gap_bin_given_group",
            eda["GroupName_disp"],
            eda["gap_label"],
        )
    )

    out_csv = out_dir / "baseline_topk_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        tasks = [r["task"] for r in rows]
        x = np.arange(len(tasks))
        w = 0.25
        ax.bar(x - w, [r["top1_global"] for r in rows], width=w, label="top-1 global marginal")
        ax.bar(x, [r["top1_conditional"] for r in rows], width=w, label="top-1 given current")
        ax.bar(x + w, [r["top5_conditional"] for r in rows], width=w, label="top-5 given current")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Empirical baseline predictability (conditional frequencies)")
        ax.legend(fontsize=8)
        fig.text(0.5, 0.01, CAPTION, ha="center", fontsize=8)
        fig.subplots_adjust(bottom=0.22)
        fig_path = out_dir / "baseline_topk_accuracy.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s", fig_path)
    except Exception as exc:
        logger.warning("Could not render baseline chart: %s", exc)


if __name__ == "__main__":
    main()
