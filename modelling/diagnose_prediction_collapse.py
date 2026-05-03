#!/usr/bin/env python3
"""Compare model predictions vs majority-class baseline on val/test (temporal split).

If accuracy ≈ frequency of the most common target class, the model may be collapsing to the
majority label. Also reports prediction entropy / diversity.

Example:
  python modelling/diagnose_prediction_collapse.py \\
    --checkpoint data/processed/patient_event_model/patient_event_model.pt \\
    --split test \\
    --max-batches 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modelling.infer_patient_event_model import (  # noqa: E402
    _instantiate_model,
    _merge_args_with_checkpoint,
)
from modelling.train_patient_event_model import (  # noqa: E402
    PatientSequenceDataset,
    SPLIT_TEST,
    SPLIT_VAL,
    load_prepared_pt,
    merge_head_to_target,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--vocab-json", type=Path, default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=0, help="Cap batches for speed (0 = full loader).")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-30)
    return h


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = _merge_args_with_checkpoint(ckpt)
    input_pt = args.input or Path(train_args.input)
    vocab_json = args.vocab_json if args.vocab_json is not None else Path(train_args.vocab_json)
    if not input_pt.exists():
        raise SystemExit(f"Sequence .pt not found: {input_pt}")
    if not vocab_json.exists():
        raise SystemExit(f"Vocab not found: {vocab_json}")

    prepared = load_prepared_pt(input_pt, vocab_json)
    if not prepared.sequences:
        raise SystemExit("No sequences loaded.")

    head_to_target = merge_head_to_target(prepared.external_stream_bases)
    model = _instantiate_model(prepared, train_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    sdoh_inputs = [f"sdoh_{idx}" for idx in range(len(prepared.sdoh_fields))]
    indexed: list[dict[str, Any]] = []
    for seq in prepared.sequences:
        clone = dict(seq)
        for idx, field in enumerate(prepared.sdoh_fields):
            clone[f"sdoh_{idx}"] = clone[field]
        indexed.append(clone)

    ds = PatientSequenceDataset(
        indexed,
        max_seq_len=train_args.max_seq_len,
        sdoh_fields=sdoh_inputs,
        patient_context_fields=prepared.patient_context_fields,
        patient_numeric_fields=prepared.patient_numeric_fields,
        temporal_split=True,
        train_fraction=train_args.temporal_train_frac,
        valid_fraction=train_args.temporal_val_frac,
        test_fraction=train_args.temporal_test_frac,
        external_stream_bases=prepared.external_stream_bases,
        external_per_token_dim=prepared.external_per_token_dim,
        external_feature_key=prepared.external_feature_key,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    want = SPLIT_VAL if args.split == "val" else SPLIT_TEST

    # Per head: target counts, pred counts, correct
    targets_all: dict[str, Counter[int]] = {h: Counter() for h in head_to_target}
    preds_all: dict[str, Counter[int]] = {h: Counter() for h in head_to_target}
    correct: dict[str, int] = {h: 0 for h in head_to_target}
    total: dict[str, int] = {h: 0 for h in head_to_target}

    batches = 0
    with torch.no_grad():
        for batch in loader:
            if args.max_batches and batches >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            split_role = batch["split_role"]
            position_mask = split_role == want
            outputs = model(batch)

            for head, suffix in head_to_target.items():
                tgt = batch[f"target_{suffix}"]
                logits = outputs[head]
                pred = logits.argmax(dim=-1)
                m = position_mask & (tgt != -100)
                if not m.any():
                    continue
                t_sub = tgt[m]
                p_sub = pred[m]
                correct[head] += int((t_sub == p_sub).sum().item())
                n = int(m.sum().item())
                total[head] += n
                for tid, pid in zip(t_sub.tolist(), p_sub.tolist()):
                    targets_all[head][int(tid)] += 1
                    preds_all[head][int(pid)] += 1
            batches += 1

    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "batches_scanned": batches,
        "device": str(device),
        "heads": {},
    }

    for head in head_to_target:
        tot = total[head]
        acc = correct[head] / tot if tot else float("nan")
        tc, pc = targets_all[head], preds_all[head]
        maj_tgt = tc.most_common(1)[0] if tc else (None, 0)
        maj_freq = maj_tgt[1] / tot if tot else float("nan")
        maj_pred = pc.most_common(1)[0] if pc else (None, 0)
        pred_freq = maj_pred[1] / tot if tot else float("nan")
        ent_p = _entropy_from_counts(pc)
        ent_t = _entropy_from_counts(tc)
        uniq_pred = len(pc)

        # "Dumb" baseline: always predict the most frequent *target* class; its accuracy = target_majority_mass.
        acc_minus_maj = acc - maj_freq
        report["heads"][head] = {
            "n_positions": tot,
            "accuracy": acc,
            "target_majority_class_id": maj_tgt[0],
            "target_majority_mass": maj_freq,
            "acc_minus_majority_baseline": acc_minus_maj,
            "predictions_most_common_id": maj_pred[0],
            "prediction_top1_mass": pred_freq,
            "pred_entropy_bits": ent_p,
            "target_entropy_bits": ent_t,
            "unique_predicted_ids": uniq_pred,
            "suspected_constant_majority": bool(
                tot > 500 and maj_pred[0] == maj_tgt[0] and pred_freq > 0.85 and acc - maj_freq < 0.02
            ),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
