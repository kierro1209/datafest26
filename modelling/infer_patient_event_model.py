#!/usr/bin/env python3
"""Export per-timestep predictions from a trained patient event model checkpoint."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modelling.train_patient_event_model import (  # noqa: E402
    PatientEventSequenceModel,
    PatientSequenceDataset,
    PreparedData,
    SPECIAL_TOKENS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    _infer_patient_context_vocab_sizes,
    _mapping_size,
    merge_head_to_target,
    load_prepared_pt,
    split_sequences,
)


class PatientSequenceDatasetWithPatientId(PatientSequenceDataset):
    """Adds ``patient_id`` for export (training loader uses default collate without this)."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        batch = super().__getitem__(idx)
        batch["patient_id"] = str(self.sequences[idx].get("patient_id", "UNKNOWN"))
        return batch


def collate_with_patient_id(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if key == "patient_id":
            out[key] = vals
        else:
            out[key] = torch.stack(vals, dim=0)
    return out


def _head_vocab_maps(prepared: PreparedData) -> dict[str, dict[int, str]]:
    """head_key -> id -> label string (best-effort inverse maps)."""
    maps: dict[str, dict[int, str]] = {}
    core = {
        "type": prepared.type_to_id,
        "event_description": prepared.event_description_to_id,
        "group_code": prepared.group_code_to_id,
        "diagnosis_value": prepared.diagnosis_value_to_id,
        "gap": prepared.gap_to_id,
        "setting": prepared.setting_to_id,
        "dept_type": prepared.dept_type_to_id,
        "dept_specialty": prepared.dept_specialty_to_id,
        "facility_size": prepared.facility_size_to_id,
        "region": prepared.region_to_id,
    }
    for head, m in core.items():
        maps[head] = {int(v): str(k) for k, v in m.items()}
    for base in prepared.external_stream_bases:
        d = prepared.external_to_id.get(base, {})
        maps[base] = {int(v): str(k) for k, v in d.items()}
    return maps


def _instantiate_model(prepared: PreparedData, args: argparse.Namespace) -> PatientEventSequenceModel:
    external_vocab_sizes = {b: _mapping_size(prepared.external_to_id[b]) for b in prepared.external_stream_bases}
    return PatientEventSequenceModel(
        type_size=_mapping_size(prepared.type_to_id),
        event_description_size=_mapping_size(prepared.event_description_to_id),
        group_code_size=_mapping_size(prepared.group_code_to_id),
        diagnosis_value_size=_mapping_size(prepared.diagnosis_value_to_id),
        gap_size=_mapping_size(prepared.gap_to_id),
        setting_size=_mapping_size(prepared.setting_to_id),
        dept_type_size=_mapping_size(prepared.dept_type_to_id),
        dept_specialty_size=_mapping_size(prepared.dept_specialty_to_id),
        facility_size_size=_mapping_size(prepared.facility_size_to_id),
        region_size=_mapping_size(prepared.region_to_id),
        sdoh_size=_mapping_size(prepared.sdoh_status_to_id),
        n_sdoh_streams=len(prepared.sdoh_fields),
        patient_context_vocab_sizes=_infer_patient_context_vocab_sizes(
            prepared.patient_context_to_id,
            prepared.patient_context_fields,
        ),
        patient_numeric_dim=len(prepared.patient_numeric_fields),
        d_model=args.d_model,
        max_seq_len=args.max_seq_len,
        backbone=args.backbone,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pad_token_id=getattr(args, "pad_token_id", SPECIAL_TOKENS["[PAD]"]),
        external_stream_bases=prepared.external_stream_bases,
        external_vocab_sizes=external_vocab_sizes,
        external_per_token_dim=prepared.external_per_token_dim,
        external_feature_key=prepared.external_feature_key,
    )


def _merge_args_with_checkpoint(ckpt: dict[str, Any]) -> argparse.Namespace:
    _argv = sys.argv
    sys.argv = ["train_patient_event_model.py"]
    try:
        from modelling.train_patient_event_model import parse_args as train_parse_args

        base = train_parse_args()
    finally:
        sys.argv = _argv
    merged = vars(base).copy()
    saved = ckpt.get("args")
    if isinstance(saved, dict):
        merged.update(saved)
    if "pad_token_id" not in merged:
        merged["pad_token_id"] = SPECIAL_TOKENS["[PAD]"]
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference and export per-timestep predictions to CSV.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to patient_event_model.pt from training.")
    p.add_argument("--input", type=Path, default=None, help="Sequence .pt (default: training args in checkpoint).")
    p.add_argument("--vocab-json", type=Path, default=None, help="Vocab JSON (default: training args in checkpoint).")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/patient_event_model/predictions_export.csv.gz"),
        help="Output CSV path (.csv or .csv.gz).",
    )
    p.add_argument(
        "--split",
        choices=["val", "test", "both", "train", "all_holdout_valid"],
        default="test",
        help="Temporal: export positions in val/test/train regions. Holdout: use all_holdout_valid for validation patients.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-rows", type=int, default=0, help="Stop after this many CSV rows (0 = no limit).")
    p.add_argument("--decode-labels", action="store_true", help="Add human-readable label columns using vocab maps.")
    p.add_argument(
        "--no-mmap-load",
        action="store_true",
        help="Disable memory-mapped load for the sequence .pt (same as training).",
    )
    return p.parse_args()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def export_predictions() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = _merge_args_with_checkpoint(ckpt)

    input_pt = args.input or Path(train_args.input)
    vocab_json = args.vocab_json if args.vocab_json is not None else Path(train_args.vocab_json)

    prepared = load_prepared_pt(input_pt, vocab_json, mmap_load=not args.no_mmap_load)
    if not prepared.sequences:
        raise SystemExit("No sequences loaded.")

    head_to_target = merge_head_to_target(prepared.external_stream_bases)
    label_maps = _head_vocab_maps(prepared) if args.decode_labels else {}

    model = _instantiate_model(prepared, train_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    sdoh_inputs = [f"sdoh_{idx}" for idx in range(len(prepared.sdoh_fields))]
    temporal = getattr(train_args, "split_strategy", "temporal") == "temporal"

    indexed: list[dict[str, Any]] = []
    for seq in prepared.sequences:
        clone = dict(seq)
        for idx, field in enumerate(prepared.sdoh_fields):
            clone[f"sdoh_{idx}"] = clone[field]
        indexed.append(clone)

    common_kw = dict(
        max_seq_len=train_args.max_seq_len,
        sdoh_fields=sdoh_inputs,
        patient_context_fields=prepared.patient_context_fields,
        patient_numeric_fields=prepared.patient_numeric_fields,
        temporal_split=temporal,
        train_fraction=train_args.temporal_train_frac,
        valid_fraction=train_args.temporal_val_frac,
        test_fraction=train_args.temporal_test_frac,
        external_stream_bases=prepared.external_stream_bases,
        external_per_token_dim=prepared.external_per_token_dim,
        external_feature_key=prepared.external_feature_key,
    )

    if temporal:
        ds = PatientSequenceDatasetWithPatientId(indexed, **common_kw)
    else:
        _train_seqs, valid_seqs = split_sequences(prepared.sequences, train_args.valid_fraction, train_args.seed)
        val_indexed = []
        for seq in valid_seqs:
            clone = dict(seq)
            for idx, field in enumerate(prepared.sdoh_fields):
                clone[f"sdoh_{idx}"] = clone[field]
            val_indexed.append(clone)
        if args.split != "all_holdout_valid":
            print(
                "patient_holdout mode: ignoring --split except all_holdout_valid; exporting validation patient sequences.",
                file=sys.stderr,
            )
        ds = PatientSequenceDatasetWithPatientId(val_indexed, **{**common_kw, "temporal_split": False})

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_with_patient_id,
    )

    fieldnames = [
        "patient_id",
        "timestep",
        "temporal_split_region",
        "head",
        "target_id",
        "predicted_id",
        "correct",
    ]
    if args.decode_labels:
        fieldnames.extend(["target_label", "predicted_label"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with _open_text(args.output) as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()

        with torch.no_grad():
            for batch in loader:
                pids = batch.pop("patient_id")
                batch_tensors = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch_tensors)

                B = batch_tensors["attention_mask"].size(0)
                T = batch_tensors["attention_mask"].size(1)
                split_role = batch_tensors.get("split_role")

                for b in range(B):
                    pid = pids[b]
                    for t in range(T):
                        if batch_tensors["attention_mask"][b, t].item() == 0:
                            continue

                        if temporal and split_role is not None:
                            sr = int(split_role[b, t].item())
                            if sr == SPLIT_TRAIN:
                                region = "train"
                            elif sr == SPLIT_VAL:
                                region = "val"
                            elif sr == SPLIT_TEST:
                                region = "test"
                            else:
                                region = "pad"
                            if args.split == "val" and sr != SPLIT_VAL:
                                continue
                            if args.split == "test" and sr != SPLIT_TEST:
                                continue
                            if args.split == "train" and sr != SPLIT_TRAIN:
                                continue
                            if args.split == "both" and sr not in (SPLIT_VAL, SPLIT_TEST):
                                continue
                        else:
                            region = "valid_patient_holdout"

                        for head, suffix in head_to_target.items():
                            tgt = batch_tensors[f"target_{suffix}"][b, t].item()
                            if tgt == -100:
                                continue
                            logits = outputs[head][b, t]
                            pred = int(logits.argmax(dim=-1).item())
                            row = {
                                "patient_id": pid,
                                "timestep": t,
                                "temporal_split_region": region,
                                "head": head,
                                "target_id": tgt,
                                "predicted_id": pred,
                                "correct": int(pred == tgt),
                            }
                            if args.decode_labels:
                                lm = label_maps.get(head, {})
                                row["target_label"] = lm.get(int(tgt), "")
                                row["predicted_label"] = lm.get(int(pred), "")
                            w.writerow(row)
                            rows_written += 1
                            if args.max_rows and rows_written >= args.max_rows:
                                print(
                                    json.dumps(
                                        {
                                            "status": "stopped_max_rows",
                                            "rows_written": rows_written,
                                            "output": str(args.output.resolve()),
                                        },
                                        indent=2,
                                    )
                                )
                                return

    print(
        json.dumps(
            {
                "status": "done",
                "rows_written": rows_written,
                "output": str(args.output.resolve()),
                "checkpoint": str(ckpt_path.resolve()),
                "split_filter": args.split if temporal else "all_holdout_valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    export_predictions()
