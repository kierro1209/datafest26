#!/usr/bin/env python3
"""
Model vs empirical baseline visuals (test set, final epoch).

Reads:
  - ``baseline_topk_metrics.csv`` from ``eda/next_event_baselines.py`` (tasks + top1_conditional + top5_conditional)
  - ``training_history_snapshot.json`` from train_patient_event_model (per-epoch valid/test metrics)

Outputs:
  - Dumbbell chart: baseline vs final-epoch **test Top-1** accuracy, sorted by improvement (model − baseline).
  - Optional appendix: small multiples of **test** accuracy vs epoch with dashed conditional-frequency baselines.

Example:
  python modelling/plot_model_vs_baseline.py \\
    --baseline visuals/eda_story/baseline_topk_metrics.csv \\
    --snapshot data/processed/patient_event_model/training_history_snapshot.json \\
    --output-dir visuals/gpt_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Storytelling split (optional ``--heads-subset`` presets).
HEADS_HARD = (
    "dept_specialty",
    "diagnosis_value",
    "event_description",
    "group_code",
    "type",
)
HEADS_EASY = ("dept_type", "facility_size", "gap", "region", "setting")


def load_baseline_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "top1_conditional" not in df.columns:
        raise SystemExit(f"{path}: expected column top1_conditional")
    if "top5_conditional" not in df.columns:
        raise SystemExit(f"{path}: expected column top5_conditional")
    if "task" not in df.columns:
        raise SystemExit(f"{path}: expected column task")
    return df


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list of epoch records")
    return data


def pick_final_epoch_record(snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        r
        for r in snapshot
        if isinstance(r.get("test"), dict) and any(k.startswith("acc_") for k in r["test"])
    ]
    if not rows:
        raise SystemExit("Snapshot has no rows with test.acc_* metrics.")
    return max(rows, key=lambda r: int(r.get("epoch", -1)))


def _sorted_snapshot_rows(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in snapshot if isinstance(r.get("test"), dict)]
    return sorted(rows, key=lambda r: int(r.get("epoch", 0)))


def aligned_test_metrics(snapshot: list[dict[str, Any]]) -> tuple[list[int], dict[str, list[float | None]]]:
    """Epoch list parallel to each metric series (only rows with test dict)."""
    rows = _sorted_snapshot_rows(snapshot)
    eps = [int(r["epoch"]) for r in rows]
    keys: set[str] = set()
    for r in rows:
        for k, v in r["test"].items():
            if isinstance(v, (int, float)) and (k.startswith("acc_") or k.startswith("top5_")):
                keys.add(k)
    out: dict[str, list[float | None]] = {}
    for k in sorted(keys):
        out[k] = []
        for r in rows:
            t = r["test"]
            if isinstance(t, dict) and k in t and isinstance(t[k], (int, float)):
                out[k].append(float(t[k]))
            else:
                out[k].append(None)
    return eps, out


def build_comparison_table(
    baseline: pd.DataFrame,
    final_test: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for _, br in baseline.iterrows():
        task = str(br["task"])
        b1 = float(br["top1_conditional"])
        b5 = float(br["top5_conditional"])
        k1 = f"acc_{task}"
        k5 = f"top5_{task}"
        m1 = final_test.get(k1)
        m5 = final_test.get(k5)
        if m1 is None or m5 is None:
            continue
        m1f = float(m1)
        m5f = float(m5)
        rows.append(
            {
                "task": task,
                "baseline_top1": b1,
                "model_top1": m1f,
                "delta_top1": m1f - b1,
                "baseline_top5": b5,
                "model_top5": m5f,
                "delta_top5": m5f - b5,
            }
        )
    return pd.DataFrame(rows)


def plot_dumbbell(
    cmp_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    sort_by: str,
) -> None:
    if cmp_df.empty:
        raise SystemExit("No overlapping baseline/model tasks to plot.")

    if sort_by != "top1":
        raise SystemExit("plot_dumbbell now supports only sort_by='top1'.")
    cmp_df = cmp_df.sort_values("delta_top1", ascending=True)

    tasks = cmp_df["task"].tolist()
    y = np.arange(len(tasks))
    fig, ax1 = plt.subplots(1, 1, figsize=(7.2, max(4.0, 0.38 * len(tasks) + 1.2)))

    def _panel(ax: Any, bcol: str, mcol: str, dcol: str, ptitle: str) -> None:
        for i, r in enumerate(cmp_df.itertuples(index=False)):
            b = getattr(r, bcol)
            m = getattr(r, mcol)
            d = getattr(r, dcol)
            color = "#15803d" if d >= 0 else "#b91c1c"
            ax.plot([b, m], [i, i], color=color, lw=2.2, solid_capstyle="round", zorder=1)
            ax.scatter([b], [i], s=52, color="#64748b", edgecolors="white", linewidths=0.8, zorder=3, label="baseline" if i == 0 else "")
            ax.scatter([m], [i], s=52, color="#2563eb", edgecolors="white", linewidths=0.8, zorder=3, label="model (final)" if i == 0 else "")
        ax.set_yticks(y)
        ax.set_yticklabels(tasks, fontsize=9)
        ax.set_xlabel("accuracy", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_title(ptitle, fontsize=11, fontweight="600", pad=10)
        ax.grid(True, axis="x", alpha=0.35)

    _panel(ax1, "baseline_top1", "model_top1", "delta_top1", "Top-1 accuracy")
    ax1.invert_yaxis()

    # Legend below panel (avoids covering low-y tasks like gap).
    h0, l0 = ax1.get_legend_handles_labels()
    if h0:
        fig.legend(
            h0,
            l0,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=8,
            frameon=True,
            fancybox=True,
            edgecolor="#e2e8f0",
        )
        leg_ax = ax1.get_legend()
        if leg_ax is not None:
            leg_ax.remove()

    # Low rect top (~0.74) clears space above axes so subtitle does not overlap panel title.
    plt.tight_layout(rect=[0, 0.09, 1, 0.74])
    fig.suptitle(title, fontsize=12, fontweight="600", y=0.975)
    fig.text(0.5, 0.888, subtitle, ha="center", fontsize=9, color="#475569")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_learning_curves_appendix(
    snapshot: list[dict[str, Any]],
    baseline_df: pd.DataFrame,
    heads: list[str],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    eps, bundle = aligned_test_metrics(snapshot)
    if not eps:
        raise SystemExit("No epochs with test metrics in snapshot.")

    n = len(heads)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 2.35 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])

    def _baseline_vals(task: str) -> tuple[float, float]:
        sub = baseline_df[baseline_df["task"] == task]
        if len(sub) == 0:
            return float("nan"), float("nan")
        return float(sub.iloc[0]["top1_conditional"]), float(sub.iloc[0]["top5_conditional"])

    for i, h in enumerate(heads):
        k1 = f"acc_{h}"
        k5 = f"top5_{h}"
        s1 = bundle.get(k1)
        s5 = bundle.get(k5)
        b1, b5 = _baseline_vals(h)

        ax_a = axes[i, 0]
        ax_t = axes[i, 1]
        if s1:
            yy1 = [x if x is not None and np.isfinite(x) else np.nan for x in s1]
            ax_a.plot(eps, yy1, "o-", color="#2563eb", lw=1.4, markersize=5)
        if np.isfinite(b1):
            ax_a.axhline(b1, color="#64748b", ls="--", lw=1.5, label="empirical baseline")
        ax_a.set_ylabel(h, fontsize=10, fontweight="600")
        ax_a.set_ylim(0, 1.02)
        ax_a.grid(True, alpha=0.3)
        if i == 0:
            ax_a.set_title("Test — top-1", fontsize=10)
        if i == n - 1:
            ax_a.set_xlabel("epoch")

        if s5:
            yy5 = [x if x is not None and np.isfinite(x) else np.nan for x in s5]
            ax_t.plot(eps, yy5, "o-", color="#2563eb", lw=1.4, markersize=5)
        if np.isfinite(b5):
            ax_t.axhline(b5, color="#64748b", ls="--", lw=1.5)
        ax_t.set_ylim(0, 1.02)
        ax_t.grid(True, alpha=0.3)
        if i == 0:
            ax_t.set_title("Test — top-5", fontsize=10)
        if i == n - 1:
            ax_t.set_xlabel("epoch")
        if i == 0 and any(np.isfinite([b1, b5])):
            ax_a.legend(loc="lower right", fontsize=7)

    # Match plot_dumbbell: leave headroom so subtitle does not overlap first-row subplot titles.
    plt.tight_layout(rect=[0, 0.02, 1, 0.74])
    fig.suptitle(title, fontsize=12, fontweight="600", y=0.975)
    fig.text(0.5, 0.888, subtitle, ha="center", fontsize=9, color="#475569")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-1 dumbbell + appendix plots: model vs empirical baseline.")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=_REPO_ROOT / "visuals/eda_story/baseline_topk_metrics.csv",
        help="baseline_topk_metrics.csv from next_event_baselines.py",
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=_REPO_ROOT / "data/processed/patient_event_model/training_history_snapshot.json",
        help="training_history_snapshot.json next to train.log",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "visuals/gpt_model",
        help="Directory for PNG outputs",
    )
    ap.add_argument(
        "--heads-subset",
        choices=("all", "hard", "easy"),
        default="all",
        help="Restrict dumbbell to preset head groups (hard=easy-to-narrate wins).",
    )
    ap.add_argument(
        "--sort-by",
        choices=("top1",),
        default="top1",
        help="Sort tasks by Top-1 accuracy delta.",
    )
    ap.add_argument(
        "--learning-curves",
        action="store_true",
        help="Also write appendix small-multiples PNG for selected heads.",
    )
    ap.add_argument(
        "--appendix-heads",
        type=str,
        default=",".join(HEADS_HARD),
        help="Comma-separated heads for appendix figure (default: hard heads).",
    )
    args = ap.parse_args()

    baseline_path = args.baseline.expanduser().resolve()
    snap_path = args.snapshot.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not baseline_path.is_file():
        raise SystemExit(f"Baseline CSV not found: {baseline_path}")
    if not snap_path.is_file():
        raise SystemExit(f"Snapshot not found: {snap_path}")

    baseline_df = load_baseline_csv(baseline_path)
    snapshot = load_snapshot(snap_path)
    final = pick_final_epoch_record(snapshot)
    test_block = final.get("test")
    if not isinstance(test_block, dict):
        raise SystemExit("Final epoch record has no test metrics dict.")

    cmp_all = build_comparison_table(baseline_df, test_block)
    if args.heads_subset == "hard":
        cmp = cmp_all[cmp_all["task"].isin(HEADS_HARD)]
    elif args.heads_subset == "easy":
        cmp = cmp_all[cmp_all["task"].isin(HEADS_EASY)]
    else:
        cmp = cmp_all

    stem = "model_vs_baseline_dumbbell"
    if args.heads_subset != "all":
        stem += f"_{args.heads_subset}"

    title = "Model vs empirical baseline by prediction task (test set, final epoch, Top-1)"
    subtitle = (
        "Baseline = conditional frequency of next label given current label (linked pairs). "
        f"Final epoch = {int(final.get('epoch', 0))}. "
        "Green connector = model ≥ baseline; red = below baseline."
    )

    plot_dumbbell(
        cmp,
        out_dir / f"{stem}.png",
        title=title,
        subtitle=subtitle,
        sort_by=args.sort_by,
    )
    print(f"Wrote {out_dir / f'{stem}.png'}")

    if args.learning_curves:
        heads = [h.strip() for h in args.appendix_heads.split(",") if h.strip()]
        plot_learning_curves_appendix(
            snapshot,
            baseline_df,
            heads,
            out_dir / "model_vs_baseline_test_curves_appendix.png",
            title="Test accuracy by epoch, with empirical baseline",
            subtitle="Solid = model (test); dashed = conditional-frequency baseline from next_event_baselines.py",
        )
        print(f"Wrote {out_dir / 'model_vs_baseline_test_curves_appendix.png'}")


if __name__ == "__main__":
    main()
