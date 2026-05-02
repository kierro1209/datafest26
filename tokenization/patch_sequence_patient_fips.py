#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch


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


def normalize_fips(value: Any) -> str | None:
    if value is None:
        return None

    s = str(value).strip()
    if not s or s.upper() in {"NA", "NAN", "NULL", "NONE", "UNKNOWN", "*UNSPECIFIED"}:
        return None

    # Preserve identifiers as strings. Leading zeros can matter.
    if s.endswith(".0"):
        s = s[:-2]

    # Keep alphanumeric characters only.
    s = "".join(ch for ch in s if ch.isalnum())

    return s if s else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch CensusBlockGroupFipsCode into patient_context_values only."
    )
    parser.add_argument(
        "--sequence-pt",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "patient_sequences_encounter_only_with_sdoh_status.pt"
        ),
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
            "sequence_model_vocab_encounter_only_with_sdoh_status.json"
        ),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "patient_sequences_encounter_only_with_sdoh_status_and_fips.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "sequence_model_vocab_encounter_only_with_sdoh_status_and_fips.json"
        ),
    )
    args = parser.parse_args()

    if not args.sequence_pt.exists():
        raise SystemExit(f"Missing sequence artifact: {args.sequence_pt}")

    if not args.work_db.exists():
        raise SystemExit(f"Missing work DB: {args.work_db}")

    log(f"[FIPS] loading sequence artifact: {args.sequence_pt}")
    artifact = torch.load(args.sequence_pt, map_location="cpu")
    sequences = artifact["sequences"]
    log(f"[FIPS] loaded sequences: {len(sequences):,}")

    conn = sqlite3.connect(args.work_db, timeout=120)
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA busy_timeout=120000;")

    if not table_exists(conn, "event_stage"):
        raise SystemExit("event_stage not found in work DB.")

    available = get_columns(conn, "event_stage")

    if "CensusBlockGroupFipsCode" not in available:
        raise SystemExit("event_stage does not contain CensusBlockGroupFipsCode.")

    log("[FIPS] reading one CensusBlockGroupFipsCode per patient")

    query = """
    SELECT
      PatientDurableKey,
      CensusBlockGroupFipsCode
    FROM event_stage
    GROUP BY PatientDurableKey
    """

    patient_fips: dict[str, str | None] = {}

    for patient_id, fips in conn.execute(query):
        patient_id = str(patient_id or "").strip()
        if not patient_id:
            continue
        patient_fips[patient_id] = normalize_fips(fips)

    conn.close()

    log(f"[FIPS] loaded FIPS values for {len(patient_fips):,} patients")

    patched = 0
    missing = 0
    non_missing = 0

    for seq in sequences:
        patient_id = str(seq.get("patient_id") or "").strip()
        fips_value = patient_fips.get(patient_id)

        seq.setdefault("patient_context_values", {})
        seq["patient_context_values"]["patient_census_block_group_fips"] = fips_value

        if fips_value is None:
            missing += 1
        else:
            non_missing += 1

        patched += 1

    metadata = artifact.get("metadata", {})
    metadata.update(
        {
            "has_patient_census_block_group_fips": True,
            "patient_census_block_group_fips_location": "patient_context_values.patient_census_block_group_fips",
            "patient_census_block_group_fips_sequences_patched": int(patched),
            "patient_census_block_group_fips_non_missing": int(non_missing),
            "patient_census_block_group_fips_missing": int(missing),
            "patient_census_block_group_fips_source": "event_stage.CensusBlockGroupFipsCode",
            "patient_census_block_group_fips_added_to_context_ids": False,
            "patient_census_block_group_fips_added_to_context_raw": False,
        }
    )

    artifact["sequences"] = sequences
    artifact["metadata"] = metadata

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    log(f"[FIPS] saving patched artifact: {args.output_pt}")
    torch.save(artifact, args.output_pt)

    vocab_artifact = {}
    if args.vocab_json.exists():
        vocab_artifact = json.loads(args.vocab_json.read_text(encoding="utf-8"))

    vocab_artifact["metadata"] = metadata

    log(f"[FIPS] saving patched vocab json: {args.output_json}")
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