#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_value(value: Any, max_len: int = 80) -> str:
    if value is None:
        return "UNKNOWN"

    s = str(value).strip()
    if not s or s.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return "UNKNOWN"

    missing_map = {
        "*UNSPECIFIED": "ASKED_NOT_ANSWERED_OR_UNABLE",
        "*UNKNOWN": "ASKED_NOT_ANSWERED_OR_UNABLE",
        "*NOT APPLICABLE": "ASKED_NOT_ANSWERED_OR_UNABLE",
        "UNSPECIFIED": "NOT_RECORDED_OR_UNKNOWN",
        "UNKNOWN": "NOT_RECORDED_OR_UNKNOWN",
        "NOT APPLICABLE": "STRUCTURAL_NOT_APPLICABLE",
    }
    if s.upper() in missing_map:
        return missing_map[s.upper()]

    s = s.upper()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s[:max_len] if s else "UNKNOWN"


def get_or_add(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch diagnosis_value_ids and group_code_ids into patient sequence .pt artifact."
    )
    parser.add_argument(
        "--sequence-pt",
        type=Path,
        default=Path("data/processed/sequence_model/patient_sequences_for_model.pt"),
    )
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=Path("data/processed/sequence_model/sequence_model_events.csv.gz"),
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=Path("data/processed/sequence_model/sequence_model_vocab.json"),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path("data/processed/sequence_model/patient_sequences_for_model_with_diagnosis.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/sequence_model/sequence_model_vocab_with_diagnosis.json"),
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    if not args.sequence_pt.exists():
        raise SystemExit(f"Missing sequence pt file: {args.sequence_pt}")

    if not args.event_csv.exists():
        raise SystemExit(
            f"Missing event csv: {args.event_csv}\n"
            "Patch requires sequence_model_events.csv.gz. If you ran tokenizer with --no-event-csv, rerun without that flag."
        )

    log(f"[PATCH] loading sequence artifact: {args.sequence_pt}")
    artifact = torch.load(args.sequence_pt, map_location="cpu")

    sequences = artifact["sequences"]
    log(f"[PATCH] loaded patient sequences: {len(sequences):,}")

    group_code_to_id: dict[str, int] = {"UNKNOWN": 0}
    diagnosis_value_to_id: dict[str, int] = {"UNKNOWN": 0}

    # Build patient -> diagnosis/group arrays from event-level CSV.
    patient_diag: dict[str, list[int]] = {}
    patient_group: dict[str, list[int]] = {}

    expected_cols = [
        "PatientDurableKey",
        "patient_event_index",
        "GroupCode",
        "DiagnosisValue",
    ]

    log(f"[PATCH] reading event csv: {args.event_csv}")
    total_rows = 0

    for chunk in pd.read_csv(
        args.event_csv,
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
        usecols=lambda c: c in expected_cols,
    ):
        missing = [c for c in expected_cols if c not in chunk.columns]
        if missing:
            raise SystemExit(f"Event CSV missing required columns: {missing}")

        chunk["patient_event_index_num"] = pd.to_numeric(
            chunk["patient_event_index"],
            errors="coerce",
        )
        chunk = chunk.sort_values(
            ["PatientDurableKey", "patient_event_index_num"],
            kind="mergesort",
        )

        for row in chunk.itertuples(index=False):
            patient_id = str(getattr(row, "PatientDurableKey")).strip()
            if not patient_id:
                continue

            group_norm = normalize_value(getattr(row, "GroupCode"))
            dx_norm = normalize_value(getattr(row, "DiagnosisValue"), max_len=120)

            group_id = get_or_add(group_code_to_id, group_norm)
            dx_id = get_or_add(diagnosis_value_to_id, dx_norm)

            patient_group.setdefault(patient_id, []).append(int(group_id))
            patient_diag.setdefault(patient_id, []).append(int(dx_id))

        total_rows += len(chunk)
        log(
            f"[PATCH] processed {total_rows:,} event rows; "
            f"group_code_classes={len(group_code_to_id):,}; "
            f"diagnosis_value_classes={len(diagnosis_value_to_id):,}"
        )

    log("[PATCH] attaching diagnosis streams to patient sequences")

    patched = 0
    skipped = 0
    length_mismatch = 0

    for seq in sequences:
        patient_id = str(seq["patient_id"]).strip()
        n_events = len(seq["event_token_ids"])

        group_ids = patient_group.get(patient_id)
        dx_ids = patient_diag.get(patient_id)

        if group_ids is None or dx_ids is None:
            seq["group_code_ids"] = [group_code_to_id["UNKNOWN"]] * n_events
            seq["diagnosis_value_ids"] = [diagnosis_value_to_id["UNKNOWN"]] * n_events
            skipped += 1
            continue

        if len(group_ids) != n_events or len(dx_ids) != n_events:
            # Keep alignment safe. Truncate/pad rather than crashing.
            length_mismatch += 1

            group_ids = group_ids[:n_events]
            dx_ids = dx_ids[:n_events]

            if len(group_ids) < n_events:
                group_ids += [group_code_to_id["UNKNOWN"]] * (n_events - len(group_ids))
            if len(dx_ids) < n_events:
                dx_ids += [diagnosis_value_to_id["UNKNOWN"]] * (n_events - len(dx_ids))

        seq["group_code_ids"] = [int(x) for x in group_ids]
        seq["diagnosis_value_ids"] = [int(x) for x in dx_ids]
        patched += 1

    metadata = artifact.get("metadata", {})
    metadata.update(
        {
            "group_code_classes": int(len(group_code_to_id)),
            "diagnosis_value_classes": int(len(diagnosis_value_to_id)),
            "diagnosis_patch_event_rows": int(total_rows),
            "diagnosis_patch_sequences_patched": int(patched),
            "diagnosis_patch_sequences_missing_patient": int(skipped),
            "diagnosis_patch_length_mismatch": int(length_mismatch),
            "has_group_code_ids": True,
            "has_diagnosis_value_ids": True,
        }
    )

    artifact["sequences"] = sequences
    artifact["group_code_to_id"] = group_code_to_id
    artifact["diagnosis_value_to_id"] = diagnosis_value_to_id
    artifact["metadata"] = metadata

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    log(f"[PATCH] saving patched pt: {args.output_pt}")
    torch.save(artifact, args.output_pt)

    vocab_artifact = {}
    if args.vocab_json.exists():
        vocab_artifact = json.loads(args.vocab_json.read_text(encoding="utf-8"))

    vocab_artifact["group_code_to_id"] = group_code_to_id
    vocab_artifact["diagnosis_value_to_id"] = diagnosis_value_to_id
    vocab_artifact["metadata"] = metadata

    log(f"[PATCH] saving patched vocab json: {args.output_json}")
    args.output_json.write_text(json.dumps(vocab_artifact, indent=2), encoding="utf-8")

    log(
        json.dumps(
            {
                "status": "done",
                "output_pt": str(args.output_pt),
                "output_json": str(args.output_json),
                "metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()