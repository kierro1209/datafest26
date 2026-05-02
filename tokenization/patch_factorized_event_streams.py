#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[UNK]": 3,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({qident(table)})")}


def normalize_value(value: Any, max_len: int = 100) -> str:
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


def first_nonempty(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        val = row.get(name)
        if val is not None and str(val).strip() and str(val).strip().upper() not in {"NAN", "NULL", "NONE"}:
            return val
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patch factorized event streams into sequence artifact and remove grand "
            "event_token_ids."
        )
    )

    parser.add_argument(
        "--sequence-pt",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "patient_sequences_for_model_with_diagnosis_and_context.pt"
        ),
        help="Input sequence artifact. Prefer the diagnosis+context patched artifact.",
    )
    parser.add_argument(
        "--work-db",
        type=Path,
        default=Path("data/interim/event_tokenization_v2.sqlite"),
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "sequence_model_vocab_with_diagnosis_and_context.json"
        ),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "patient_sequences_factorized_with_diagnosis_and_context.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "sequence_model_vocab_factorized_with_diagnosis_and_context.json"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--keep-grand-token",
        action="store_true",
        help="Keep event_token_ids/vocab instead of removing them.",
    )

    args = parser.parse_args()

    if not args.sequence_pt.exists():
        raise SystemExit(f"Missing sequence artifact: {args.sequence_pt}")

    if not args.work_db.exists():
        raise SystemExit(f"Missing work DB: {args.work_db}")

    log(f"[FACT] loading sequence artifact: {args.sequence_pt}")
    artifact = torch.load(args.sequence_pt, map_location="cpu")
    sequences = artifact["sequences"]
    log(f"[FACT] loaded sequences: {len(sequences):,}")

    seq_by_patient: dict[str, dict[str, Any]] = {}
    expected_lengths: dict[str, int] = {}

    for seq in sequences:
        patient_id = str(seq.get("patient_id") or "").strip()
        if not patient_id:
            continue

        # Use any existing event-level stream to determine expected length.
        if "event_token_ids" in seq:
            n = len(seq["event_token_ids"])
        elif "gap_ids" in seq:
            n = len(seq["gap_ids"])
        elif "diagnosis_value_ids" in seq:
            n = len(seq["diagnosis_value_ids"])
        else:
            raise SystemExit(f"Cannot infer event length for patient {patient_id}")

        seq_by_patient[patient_id] = seq
        expected_lengths[patient_id] = n

        seq["event_source_ids"] = []
        seq["type_ids"] = []
        seq["event_description_ids"] = []
        seq["dept_specialty_ids"] = []
        seq["sdoh_domain_ids"] = []

    conn = sqlite3.connect(args.work_db, timeout=120)
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA busy_timeout=120000;")

    if not table_exists(conn, "event_stage"):
        raise SystemExit("event_stage not found in work DB.")

    available = get_columns(conn, "event_stage")

    required = {
        "PatientDurableKey",
        "event_date",
        "event_time",
        "EncounterKey",
        "event_index_within_encounter",
        "event_id",
    }
    missing_required = sorted(required - available)
    if missing_required:
        raise SystemExit(f"event_stage missing required columns: {missing_required}")

    needed_optional = [
        "event_source",
        "event_type",
        "event_description",
        "event_domain",
        "Type",
        "VisitTypeDescription",
        "DepartmentSpecialty",
    ]
    selected = sorted(required | {c for c in needed_optional if c in available})
    select_sql = ", ".join(qident(c) for c in selected)

    event_source_to_id: dict[str, int] = {"UNKNOWN": 0}
    type_to_id: dict[str, int] = {"UNKNOWN": 0}
    event_description_to_id: dict[str, int] = {"UNKNOWN": 0}
    dept_specialty_to_id: dict[str, int] = {"UNKNOWN": 0}
    sdoh_domain_to_id: dict[str, int] = {"UNKNOWN": 0}

    query = f"""
    SELECT {select_sql}
    FROM event_stage
    ORDER BY PatientDurableKey, event_date, event_time, EncounterKey, event_index_within_encounter, event_id
    """

    log("[FACT] streaming event_stage in the same patient/event order used by tokenizer")
    cursor = conn.execute(query)
    columns = [d[0] for d in cursor.description]

    total_seen = 0
    total_attached = 0
    start = time.monotonic()

    while True:
        rows = cursor.fetchmany(args.chunksize)
        if not rows:
            break

        for tup in rows:
            row = dict(zip(columns, tup))
            patient_id = str(row.get("PatientDurableKey") or "").strip()

            seq = seq_by_patient.get(patient_id)
            if seq is None:
                continue

            event_source = normalize_value(row.get("event_source"), max_len=40)
            event_type = normalize_value(
                first_nonempty(row, ["Type", "event_type"]),
                max_len=80,
            )
            event_description = normalize_value(
                first_nonempty(row, ["VisitTypeDescription", "event_description"]),
                max_len=100,
            )
            dept_specialty = normalize_value(row.get("DepartmentSpecialty"), max_len=100)
            sdoh_domain = normalize_value(row.get("event_domain"), max_len=80)

            seq["event_source_ids"].append(
                int(get_or_add(event_source_to_id, event_source))
            )
            seq["type_ids"].append(
                int(get_or_add(type_to_id, event_type))
            )
            seq["event_description_ids"].append(
                int(get_or_add(event_description_to_id, event_description))
            )
            seq["dept_specialty_ids"].append(
                int(get_or_add(dept_specialty_to_id, dept_specialty))
            )
            seq["sdoh_domain_ids"].append(
                int(get_or_add(sdoh_domain_to_id, sdoh_domain))
            )

            total_attached += 1

        total_seen += len(rows)
        elapsed = time.monotonic() - start
        rate = total_seen / max(elapsed, 1e-9)

        log(
            f"[FACT] scanned {total_seen:,} event_stage rows; "
            f"attached {total_attached:,} rows; "
            f"{rate:,.0f} rows/sec"
        )

    conn.close()

    log("[FACT] validating sequence lengths")
    mismatch = 0
    missing = 0

    for patient_id, seq in seq_by_patient.items():
        n = expected_lengths[patient_id]

        fields = [
            "event_source_ids",
            "type_ids",
            "event_description_ids",
            "dept_specialty_ids",
            "sdoh_domain_ids",
        ]

        bad = False
        for field in fields:
            if len(seq[field]) != n:
                bad = True
                break

        if bad:
            mismatch += 1

            # Keep artifact usable: truncate/pad each new stream to expected length.
            for field in fields:
                vals = seq[field][:n]
                if len(vals) < n:
                    vals += [0] * (n - len(vals))
                seq[field] = vals

        if n == 0:
            missing += 1

        if not args.keep_grand_token:
            seq.pop("event_token_ids", None)

    metadata = artifact.get("metadata", {})
    metadata.update(
        {
            "has_factorized_event_streams": True,
            "grand_event_token_removed": bool(not args.keep_grand_token),
            "factorized_event_streams": [
                "event_source_ids",
                "type_ids",
                "event_description_ids",
                "dept_specialty_ids",
                "sdoh_domain_ids",
            ],
            "event_source_classes": int(len(event_source_to_id)),
            "type_classes": int(len(type_to_id)),
            "event_description_classes": int(len(event_description_to_id)),
            "dept_specialty_classes": int(len(dept_specialty_to_id)),
            "sdoh_domain_classes": int(len(sdoh_domain_to_id)),
            "factorized_patch_rows_attached": int(total_attached),
            "factorized_patch_length_mismatch": int(mismatch),
            "factorized_patch_empty_sequences": int(missing),
        }
    )

    if not args.keep_grand_token:
        # Remove the old grand vocabulary to avoid confusion and reduce final artifact size.
        artifact.pop("vocab", None)
        metadata.pop("event_vocab_size", None)

    artifact["sequences"] = sequences
    artifact["event_source_to_id"] = event_source_to_id
    artifact["type_to_id"] = type_to_id
    artifact["event_description_to_id"] = event_description_to_id
    artifact["dept_specialty_to_id"] = dept_specialty_to_id
    artifact["sdoh_domain_to_id"] = sdoh_domain_to_id
    artifact["metadata"] = metadata

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    log(f"[FACT] saving factorized artifact: {args.output_pt}")
    torch.save(artifact, args.output_pt)

    vocab_artifact = {}
    if args.vocab_json.exists():
        vocab_artifact = json.loads(args.vocab_json.read_text(encoding="utf-8"))

    if not args.keep_grand_token:
        vocab_artifact.pop("event_vocab", None)
        vocab_artifact.pop("vocab", None)

    vocab_artifact["event_source_to_id"] = event_source_to_id
    vocab_artifact["type_to_id"] = type_to_id
    vocab_artifact["event_description_to_id"] = event_description_to_id
    vocab_artifact["dept_specialty_to_id"] = dept_specialty_to_id
    vocab_artifact["sdoh_domain_to_id"] = sdoh_domain_to_id
    vocab_artifact["metadata"] = metadata

    log(f"[FACT] saving factorized vocab json: {args.output_json}")
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