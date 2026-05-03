#!/usr/bin/env python3
"""
Parse patient_event_model train.log (from --log-file) and plot training curves.

Extracts:
  - Epoch N/M done | train_loss=... valid_loss=... test_loss=...
  - per-head mean CE lines (train / valid / test)
  - Optimizer LR after epoch ...
  - Batch-interval running means (train / valid+test) plus final batch= lines.
  - Plots batch metrics across the full run (global batch index).
  - From training_history_snapshot.json: 2x2 Val/Test x top-1 / top-5 accuracy ({stem}_accuracy_level.png).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

# Repo root = parent of modelling/; default PNG dir is visuals/gpt_model/
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "visuals" / "gpt_model"

# test_loss may follow as " test_loss=..." (space) or " | test_loss=..." (pipe)
RE_EPOCH_DONE = re.compile(
    r"Epoch (\d+)/(\d+) done \| train_loss=([\d.eE+-]+) valid_loss=([\w.eE+-/]+)"
    r"(?:\s*(?:\|\s*)?test_loss=([\d.eE+-]+))?"
)
RE_PER_HEAD = re.compile(
    r"Epoch (\d+)/(\d+) (train|valid|test) \| per-head mean CE: (.+?) \| total=([\d.eE+-]+)"
)
RE_LR = re.compile(
    r"Optimizer LR after epoch (\d+) .*?: ([\d.eE+-]+)"
)
RE_TRAIN_BATCH = re.compile(
    r"\[(\d+)/(\d+)\] train \| batches (\d+)/(\d+) \| running_mean_total_loss=([\d.eE+-]+)"
)
RE_VALID_BATCH = re.compile(
    r"\[(\d+)/(\d+)\] valid\+test \| batches (\d+)/(\d+) \| "
    r"running_mean_val_total=([\d.eE+-]+) \| running_mean_test_total=([\d.eE+-]+)"
)
RE_TRAIN_FINAL = re.compile(
    r"\[(\d+)/(\d+)\] train \| batches=(\d+) \| wall_time=[\d.]+s \| mean_total_loss=([\d.eE+-]+)"
)
RE_VALID_FINAL = re.compile(
    r"\[(\d+)/(\d+)\] valid\+test \| batches=(\d+) \| wall_time=[\d.]+s \| "
    r"val_mean_total=([\d.eE+-]+) \| test_mean_total=([\d.eE+-]+)"
)


def _merge_batch_points(
    points: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Deduplicate by batch index; keep last occurrence (interval + final line)."""
    by_b: dict[int, tuple[int, int, float]] = {}
    for b, btot, y in points:
        by_b[b] = (b, btot, y)
    return sorted(by_b.values(), key=lambda z: z[0])


def _parse_float_token(s: str | None) -> float | None:
    if s is None:
        return None
    t = s.strip().lower()
    if t in ("n/a", "na", "none", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_head_segment(seg: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in re.finditer(r"(\w+)=([\d.eE+-]+)", seg):
        if m.group(1) == "total":
            continue
        out[m.group(1)] = float(m.group(2))
    return out


def parse_train_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    epoch_rows: list[dict[str, Any]] = []
    for m in RE_EPOCH_DONE.finditer(text):
        ep_s, _total_ep, tr_s, va_s, te_s = m.groups()
        epoch_rows.append(
            {
                "epoch": int(ep_s),
                "train_loss": float(tr_s),
                "valid_loss": _parse_float_token(va_s),
                "test_loss": _parse_float_token(te_s),
            }
        )

    by_ep: dict[int, dict[str, Any]] = {}
    for row in epoch_rows:
        by_ep[row["epoch"]] = row
    epochs_sorted = [by_ep[k] for k in sorted(by_ep)]

    per_head: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for m in RE_PER_HEAD.finditer(text):
        ep_s, _, split, seg, _tot = m.groups()
        ep = int(ep_s)
        for name, val in parse_head_segment(seg).items():
            per_head[split][name][ep] = val

    lr_after: dict[int, float] = {}
    for m in RE_LR.finditer(text):
        lr_after[int(m.group(1))] = float(m.group(2))

    train_batches_raw: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for m in RE_TRAIN_BATCH.finditer(text):
        ep = int(m.group(1))
        b, btot = int(m.group(3)), int(m.group(4))
        train_batches_raw[ep].append((b, btot, float(m.group(5))))
    for m in RE_TRAIN_FINAL.finditer(text):
        ep = int(m.group(1))
        btot = int(m.group(3))
        train_batches_raw[ep].append((btot, btot, float(m.group(4))))
    train_batches: dict[int, list[tuple[int, int, float]]] = {
        ep: _merge_batch_points(pts) for ep, pts in train_batches_raw.items()
    }

    valid_batches_raw: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    for m in RE_VALID_BATCH.finditer(text):
        ep = int(m.group(1))
        b, btot = int(m.group(3)), int(m.group(4))
        valid_batches_raw[ep].append(
            (b, btot, float(m.group(5)), float(m.group(6)))
        )
    for m in RE_VALID_FINAL.finditer(text):
        ep = int(m.group(1))
        btot = int(m.group(3))
        valid_batches_raw[ep].append(
            (btot, btot, float(m.group(4)), float(m.group(5)))
        )

    def _merge_eval_points(
        pts: list[tuple[int, int, float, float]],
    ) -> list[tuple[int, int, float, float]]:
        by_b: dict[int, tuple[int, int, float, float]] = {}
        for b, btot, v, t in pts:
            by_b[b] = (b, btot, v, t)
        return sorted(by_b.values(), key=lambda z: z[0])

    valid_batches: dict[int, list[tuple[int, int, float, float]]] = {
        ep: _merge_eval_points(pts) for ep, pts in valid_batches_raw.items()
    }

    return {
        "epochs": epochs_sorted,
        "per_head": {k: dict(v) for k, v in per_head.items()},
        "lr_after_epoch": lr_after,
        "train_batch_curves": train_batches,
        "valid_test_batch_curves": valid_batches,
    }


def load_snapshot_metrics(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return None


def plot_totals_and_lr(
    parsed: dict[str, Any],
    out_path: Path,
    title: str | None,
) -> None:
    rows = parsed["epochs"]
    if not rows:
        raise SystemExit("No 'Epoch ... done' lines found in log.")

    eps = [r["epoch"] for r in rows]
    train = np.array([r["train_loss"] for r in rows], dtype=float)
    valid = np.array(
        [r["valid_loss"] if r["valid_loss"] is not None else np.nan for r in rows],
        dtype=float,
    )
    test = np.array(
        [r["test_loss"] if r["test_loss"] is not None else np.nan for r in rows],
        dtype=float,
    )
    has_valid = bool(np.any(np.isfinite(valid)))
    has_test = bool(np.any(np.isfinite(test)))

    fig, ax0 = plt.subplots(1, 1, figsize=(9, 5))

    ax0.plot(eps, train, "o-", label="Train slice", color="C0")
    if has_valid:
        ax0.plot(eps, valid, "s-", label="Validation future slice", color="C1")
    if has_test:
        ax0.plot(eps, test, "^-", label="Test future slice", color="C2")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Mean total cross-entropy loss")
    ax0.set_xticks(eps)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")
    ax0.set_title(
        title or "Sequence model improves next-encounter forecasting"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _split_panel_title(split: str) -> str:
    return {"train": "Train", "valid": "Val", "test": "Test"}.get(split, split)


def plot_per_head_lines(
    per_head: dict[str, dict[str, dict[int, float]]],
    splits: list[str],
    out_path: Path,
    _title_prefix: str,
) -> None:
    all_heads: set[str] = set()
    for sp in splits:
        for h in per_head.get(sp, {}):
            all_heads.add(h)
    if not all_heads:
        return

    heads_sorted = sorted(all_heads)
    n = len(heads_sorted)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = {h: cmap(i % cmap.N) for i, h in enumerate(heads_sorted)}

    fig, axes = plt.subplots(
        len(splits), 1, figsize=(10, 3.2 * len(splits)), sharex=True
    )
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        sub = per_head.get(split, {})
        if not sub:
            ax.set_visible(False)
            continue
        eps_set: set[int] = set()
        for _h, ep_map in sub.items():
            eps_set.update(ep_map.keys())
        eps = sorted(eps_set)
        for h in heads_sorted:
            if h not in sub:
                continue
            ys = [sub[h].get(e, np.nan) for e in eps]
            ax.plot(eps, ys, "o-", label=h, color=colors[h], linewidth=1.2, markersize=3)
        ax.set_ylabel("Mean cross-entropy")
        ax.set_title(_split_panel_title(split))
        ax.grid(True, alpha=0.3)
        ax.set_xticks(eps)

    axes[-1].set_xlabel("epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_run_batch_series(
    parsed: dict[str, Any],
    out_path: Path,
    _stem: str,
) -> None:
    """Train and eval running means vs global batch index (each phase uses its own batch counter)."""
    train_by_ep = parsed["train_batch_curves"]
    eval_by_ep = parsed["valid_test_batch_curves"]
    epoch_totals = {r["epoch"]: r for r in parsed["epochs"]}

    if not train_by_ep and not eval_by_ep:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # --- Global train batches ---
    ax = axes[0]
    gx: list[float] = []
    gy: list[float] = []
    offset = 0.0
    eps_tr = sorted(train_by_ep.keys())
    for i, ep in enumerate(eps_tr):
        curve = train_by_ep[ep]
        if not curve:
            continue
        btot = max(bt for _, bt, _ in curve)
        for b, bt, y in curve:
            btot = bt
            gx.append(offset + float(b))
            gy.append(y)
        row = epoch_totals.get(ep)
        if row is not None:
            ax.scatter(
                offset + float(btot),
                row["train_loss"],
                s=40,
                zorder=5,
                color="C0",
                marker="o",
                edgecolors="white",
                linewidths=0.6,
                label="epoch summary train_loss" if i == 0 else "",
            )
        offset += float(btot)
        if i < len(eps_tr) - 1:
            ax.axvline(offset, color="0.55", ls="--", lw=0.85, alpha=0.85)

    if gx:
        ax.plot(gx, gy, "-", color="C0", lw=0.85, alpha=0.88, label="running mean (log intervals)")
    ax.set_ylabel("Mean total loss")
    ax.set_xlabel("Global train batch index")
    ax.set_title("Train")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [h for h, lb in zip(handles, labels) if lb],
        [lb for lb in labels if lb],
        loc="upper right",
        fontsize=8,
    )

    # --- Global eval batches ---
    ax2 = axes[1]
    gx2: list[float] = []
    gv: list[float] = []
    gt: list[float] = []
    offset_e = 0.0
    eps_ev = sorted(eval_by_ep.keys())
    for i, ep in enumerate(eps_ev):
        curve = eval_by_ep[ep]
        if not curve:
            continue
        btot = max(bt for _, bt, _, _ in curve)
        for b, bt, v, t in curve:
            btot = bt
            gx2.append(offset_e + float(b))
            gv.append(v)
            gt.append(t)
        row = epoch_totals.get(ep)
        if row is not None:
            if row.get("valid_loss") is not None:
                ax2.scatter(
                    offset_e + float(btot),
                    row["valid_loss"],
                    s=40,
                    zorder=5,
                    color="C1",
                    marker="s",
                    edgecolors="white",
                    linewidths=0.6,
                    label="epoch summary val loss" if i == 0 else "",
                )
            if row.get("test_loss") is not None:
                ax2.scatter(
                    offset_e + float(btot),
                    row["test_loss"],
                    s=40,
                    zorder=5,
                    color="C2",
                    marker="^",
                    edgecolors="white",
                    linewidths=0.6,
                    label="epoch summary test_loss" if i == 0 else "",
                )
        offset_e += float(btot)
        if i < len(eps_ev) - 1:
            ax2.axvline(offset_e, color="0.55", ls="--", lw=0.85, alpha=0.85)

    if gx2:
        ax2.plot(gx2, gv, "-", color="C1", lw=0.85, label="val running mean")
        ax2.plot(gx2, gt, "-", color="C2", lw=0.85, label="test running mean")
    ax2.set_ylabel("Mean total loss")
    ax2.set_xlabel("Global eval batch index")
    ax2.set_title("Eval")
    ax2.grid(True, alpha=0.3)
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(
        [h for h, lb in zip(h2, l2) if lb],
        [lb for lb in l2 if lb],
        loc="upper right",
        fontsize=8,
        ncol=2,
    )

    fig.suptitle(
        "Mean total loss vs batch index (train / eval)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _snapshot_metric_series(
    snapshot: list[dict[str, Any]],
    key_prefix: str,
    split: str,
) -> dict[str, list[float | None]]:
    keys: set[str] = set()
    for row in snapshot:
        s = row.get(split)
        if not isinstance(s, dict):
            continue
        for k, v in s.items():
            if k.startswith(key_prefix) and isinstance(v, (int, float)):
                keys.add(k)
    out: dict[str, list[float | None]] = {}
    for k in sorted(keys):
        series: list[float | None] = []
        for row in snapshot:
            s = row.get(split)
            if isinstance(s, dict) and k in s:
                series.append(float(s[k]))
            else:
                series.append(None)
        out[k] = series
    return out


def _plot_snapshot_metrics_on_ax(
    ax: Axes,
    eps: list[int],
    bundle: dict[str, list[float | None]],
    *,
    title: str,
    y_axis_label: str,
    legend_strip: str,
) -> bool:
    if not bundle:
        ax.set_visible(False)
        return False
    for i, (name, ys) in enumerate(bundle.items()):
        lab = name[len(legend_strip) :] if name.startswith(legend_strip) else name
        ax.plot(eps, ys, "o-", label=lab, color=f"C{i % 10}")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(y_axis_label)
    ax.set_xticks(eps)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    return True


def plot_snapshot_top1_top5_2x2(
    snapshot: list[dict[str, Any]],
    out_path: Path,
) -> bool:
    """2x2 grid: Val/Test x top-1 / top-5 accuracy from snapshot. Returns True if a file was written."""
    eps = [row["epoch"] for row in snapshot if isinstance(row.get("epoch"), int)]
    if not eps:
        return False

    m_v_acc = _snapshot_metric_series(snapshot, "acc_", "valid")
    m_t_acc = _snapshot_metric_series(snapshot, "acc_", "test")
    m_v_top = _snapshot_metric_series(snapshot, "top5_", "valid")
    m_t_top = _snapshot_metric_series(snapshot, "top5_", "test")
    if not (m_v_acc or m_t_acc or m_v_top or m_t_top):
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    any_acc = _plot_snapshot_metrics_on_ax(
        axes[0, 0],
        eps,
        m_v_acc,
        title="Val — top-1 accuracy",
        y_axis_label="accuracy",
        legend_strip="acc_",
    )
    any_acc_t = _plot_snapshot_metrics_on_ax(
        axes[0, 1],
        eps,
        m_t_acc,
        title="Test — top-1 accuracy",
        y_axis_label="accuracy",
        legend_strip="acc_",
    )
    if any_acc and any_acc_t:
        y0 = min(
            y
            for ax in (axes[0, 0], axes[0, 1])
            for y in ax.get_ylim()
        )
        y1 = max(
            y
            for ax in (axes[0, 0], axes[0, 1])
            for y in ax.get_ylim()
        )
        for ax in (axes[0, 0], axes[0, 1]):
            ax.set_ylim(y0, y1)

    any_t5v = _plot_snapshot_metrics_on_ax(
        axes[1, 0],
        eps,
        m_v_top,
        title="Val — top-5 accuracy",
        y_axis_label="accuracy",
        legend_strip="top5_",
    )
    any_t5t = _plot_snapshot_metrics_on_ax(
        axes[1, 1],
        eps,
        m_t_top,
        title="Test — top-5 accuracy",
        y_axis_label="accuracy",
        legend_strip="top5_",
    )
    if any_t5v and any_t5t:
        y0 = min(
            y
            for ax in (axes[1, 0], axes[1, 1])
            for y in ax.get_ylim()
        )
        y1 = max(
            y
            for ax in (axes[1, 0], axes[1, 1])
            for y in ax.get_ylim()
        )
        for ax in (axes[1, 0], axes[1, 1]):
            ax.set_ylim(y0, y1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot curves from patient_event_model train.log")
    ap.add_argument(
        "log_file",
        type=Path,
        nargs="?",
        default=Path("data/processed/patient_event_model/train.log"),
        help="Path to train.log",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for PNG outputs (default: visuals/gpt_model/ under the datafest26 repo root)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the totals plot (default: sequence-model headline)",
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional training_history_snapshot.json for top-1 and top-5 metric plots",
    )
    ap.add_argument(
        "--no-per-head",
        action="store_true",
        help="Skip per-head CE figures",
    )
    ap.add_argument(
        "--no-batch-series",
        action="store_true",
        help="Skip full-run batch-index plot (train_batch_series_full_run.png)",
    )
    args = ap.parse_args()

    log_path = args.log_file.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log not found: {log_path}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_train_log(log_path)
    stem = log_path.stem

    plot_totals_and_lr(parsed, out_dir / f"{stem}_totals.png", args.title)

    if not args.no_per_head and parsed["per_head"]:
        ph = parsed["per_head"]
        splits_non_empty = [s for s in ("train", "valid", "test") if ph.get(s)]
        if splits_non_empty:
            plot_per_head_lines(
                ph,
                splits_non_empty,
                out_dir / f"{stem}_per_head_ce.png",
                stem,
            )

    if not args.no_batch_series and (
        parsed["train_batch_curves"] or parsed["valid_test_batch_curves"]
    ):
        plot_full_run_batch_series(
            parsed, out_dir / f"{stem}_batch_series_full_run.png", stem
        )

    snap_path = args.snapshot
    if snap_path is None:
        cand = log_path.parent / "training_history_snapshot.json"
        if cand.is_file():
            snap_path = cand
    if snap_path is not None:
        snap = load_snapshot_metrics(snap_path.resolve())
        if snap:
            plot_snapshot_top1_top5_2x2(
                snap,
                out_dir / f"{stem}_accuracy_level.png",
            )

    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
