#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


SDOH_STATUS_TO_ID = {
    "UNKNOWN_NOT_YET_MEASURED": 0,
    "NEGATIVE_NO_NEED": 1,
    "POSITIVE_NEED_OR_RISK": 2,
    "OTHER_DECLINED_UNABLE_UNSPECIFIED": 3,
}

BASE_GAP_BINS = [
    "START",
    "0D",
    "1_7D",
    "8_30D",
    "31_90D",
    "91_180D",
    "181_365D",
    "365PLUS",
    "UNKNOWN",
]

# Expected domains from the docs, plus Postpartum Depression because it appears in some documentation.
CANONICAL_SDOH_DOMAINS = [
    "Alcohol Use",
    "Depression",
    "Financial Resource Strain",
    "Food Insecurity",
    "Housing Stability",
    "Intimate Partner Violence",
    "Physical Activity",
    "Postpartum Depression",
    "Social Connections",
    "Stress",
    "Transportation Needs",
    "Utilities",
]


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


def normalize_value(value: Any, max_len: int = 120) -> str:
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


def domain_to_field(domain: Any) -> str:
    norm = normalize_value(domain, max_len=80)
    return f"sdoh_{norm.lower()}_latest_status_ids"


def parse_date(value: Any):
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    s10 = s[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s10, fmt).date()
        except Exception:
            pass

    try:
        return datetime.fromisoformat(s.replace("Z", "")[:19]).date()
    except Exception:
        return None


def gap_bin(days: int | None, encounter_index: int) -> str:
    if encounter_index == 1:
        return "START"
    if days is None:
        return "UNKNOWN"
    if days == 0:
        return "0D"
    if 1 <= days <= 7:
        return "1_7D"
    if 8 <= days <= 30:
        return "8_30D"
    if 31 <= days <= 90:
        return "31_90D"
    if 91 <= days <= 180:
        return "91_180D"
    if 181 <= days <= 365:
        return "181_365D"
    if days > 365:
        return "365PLUS"
    return "UNKNOWN"


def classify_sdoh_answer(answer: Any) -> str:
    """
    Generic SDOH answer classifier.

    This is intentionally broad. It maps the latest answer in a domain into:
      0 unknown/not measured
      1 negative/no need/low risk
      2 positive/need/risk
      3 declined/unable/unspecified/other missing

    This can be improved later with domain-specific scoring.
    """
    if answer is None:
        return "UNKNOWN_NOT_YET_MEASURED"

    raw = str(answer).strip()
    if not raw:
        return "UNKNOWN_NOT_YET_MEASURED"

    s = raw.upper().strip()

    missing_terms = [
        "*UNSPECIFIED",
        "*UNKNOWN",
        "*NOT APPLICABLE",
        "UNSPECIFIED",
        "UNKNOWN",
        "NOT APPLICABLE",
        "REFUSED",
        "DECLINED",
        "UNABLE",
        "PATIENT DECLINED",
        "ASKED BUT UNKNOWN",
    ]
    if any(term in s for term in missing_terms):
        return "OTHER_DECLINED_UNABLE_UNSPECIFIED"

    # Numeric answers: 0 usually means negative/no symptoms/no events; >0 means something observed.
    try:
        x = float(s.replace(",", ""))
        if x == 0:
            return "NEGATIVE_NO_NEED"
        if x > 0:
            return "POSITIVE_NEED_OR_RISK"
    except Exception:
        pass

    negative_terms = [
        "NO",
        "FALSE",
        "NEVER",
        "NOT AT ALL",
        "NONE",
        "NO NEED",
        "NO RISK",
        "STABLE",
        "HAVE HOUSING",
        "I DO NOT",
        "DID NOT",
        "NO PROBLEM",
        "NOT TRUE",
        "HARDLY EVER",
    ]

    positive_terms = [
        "YES",
        "TRUE",
        "OFTEN TRUE",
        "SOMETIMES TRUE",
        "SEVERAL DAYS",
        "MORE THAN HALF",
        "NEARLY EVERY DAY",
        "DAILY",
        "WEEKLY",
        "MONTHLY",
        "ALMOST DAILY",
        "UNABLE TO PAY",
        "WORRIED",
        "RAN OUT",
        "HOMELESS",
        "SHELTER",
        "THREATENED",
        "AFRAID",
        "HURT",
        "HUMILIATED",
        "FORCED",
    ]

    # Check positive first for phrases like "Sometimes true".
    if any(term in s for term in positive_terms):
        return "POSITIVE_NEED_OR_RISK"

    if any(term in s for term in negative_terms):
        return "NEGATIVE_NO_NEED"

    return "OTHER_DECLINED_UNABLE_UNSPECIFIED"


def infer_expected_len(seq: dict[str, Any]) -> int:
    for field in [
        "type_ids",
        "event_description_ids",
        "dept_specialty_ids",
        "gap_ids",
        "setting_ids",
        "dept_type_ids",
        "facility_size_ids",
        "region_ids",
        "group_code_ids",
        "diagnosis_value_ids",
        "event_source_ids",
        "sdoh_domain_ids",
        "event_token_ids",
    ]:
        if field in seq and isinstance(seq[field], list):
            return len(seq[field])
    raise ValueError(f"Cannot infer event length for patient {seq.get('patient_id')}")


def is_event_level_list(seq: dict[str, Any], field: str, expected_len: int) -> bool:
    return isinstance(seq.get(field), list) and len(seq[field]) == expected_len


def copy_kept_event_fields(
    old_seq: dict[str, Any],
    new_seq: dict[str, Any],
    old_pos: int,
    expected_len: int,
    drop_fields: set[str],
) -> None:
    for field, value in old_seq.items():
        if field in drop_fields:
            continue

        if is_event_level_list(old_seq, field, expected_len):
            new_seq.setdefault(field, []).append(value[old_pos])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mixed ENCOUNTER+SDOH sequence artifact into encounter-only "
            "sequence artifact with rolling latest SDOH status streams."
        )
    )

    parser.add_argument(
        "--sequence-pt",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "patient_sequences_factorized_with_diagnosis_and_context.pt"
        ),
        help="Input factorized diagnosis+context artifact.",
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
            "sequence_model_vocab_factorized_with_diagnosis_and_context.json"
        ),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "patient_sequences_encounter_only_with_sdoh_status.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/processed/sequence_model/"
            "sequence_model_vocab_encounter_only_with_sdoh_status.json"
        ),
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--min-encounters-per-patient", type=int, default=2)

    args = parser.parse_args()

    if not args.sequence_pt.exists():
        raise SystemExit(f"Missing sequence artifact: {args.sequence_pt}")
    if not args.work_db.exists():
        raise SystemExit(f"Missing work DB: {args.work_db}")

    log(f"[ENCONLY] loading sequence artifact: {args.sequence_pt}")
    artifact = torch.load(args.sequence_pt, map_location="cpu")
    old_sequences = artifact["sequences"]
    log(f"[ENCONLY] loaded sequences: {len(old_sequences):,}")

    seq_by_patient: dict[str, dict[str, Any]] = {}
    expected_len_by_patient: dict[str, int] = {}
    old_pos_by_patient: dict[str, int] = {}

    for seq in old_sequences:
        patient_id = str(seq.get("patient_id") or "").strip()
        if not patient_id:
            continue
        seq_by_patient[patient_id] = seq
        expected_len_by_patient[patient_id] = infer_expected_len(seq)
        old_pos_by_patient[patient_id] = 0

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
        "EncounterKey",
        "event_date",
        "event_time",
        "event_id",
        "event_index_within_encounter",
        "event_source",
    }
    missing_required = sorted(required - available)
    if missing_required:
        raise SystemExit(f"event_stage missing required columns: {missing_required}")

    domain_col = "event_domain" if "event_domain" in available else None
    answer_col = None
    for candidate in ["event_answer_text", "event_value", "event_sdoh_answer_token"]:
        if candidate in available:
            answer_col = candidate
            break

    if domain_col is None or answer_col is None:
        raise SystemExit(
            f"Need SDOH domain/answer columns in event_stage; found domain={domain_col}, answer={answer_col}"
        )

    # Build all domain fields from canonical docs + any actual domains observed in the DB.
    canonical_domain_fields = [domain_to_field(d) for d in CANONICAL_SDOH_DOMAINS]
    observed_domain_fields: set[str] = set(canonical_domain_fields)

    log("[ENCONLY] scanning observed SDOH domains")
    domain_query = f"""
    SELECT DISTINCT {qident(domain_col)}
    FROM event_stage
    WHERE event_source = 'SDOH_RESPONSE'
      AND {qident(domain_col)} IS NOT NULL
      AND TRIM(COALESCE({qident(domain_col)}, '')) <> ''
    """
    for (domain,) in conn.execute(domain_query):
        observed_domain_fields.add(domain_to_field(domain))

    sdoh_status_fields = sorted(observed_domain_fields)
    log(f"[ENCONLY] SDOH status fields: {len(sdoh_status_fields):,}")
    for f in sdoh_status_fields:
        log(f"[ENCONLY]   {f}")

    # Pass 1: collect SDOH updates by patient+encounter.
    # This lets same-encounter SDOH answers update status before the encounter row is appended.
    log("[ENCONLY] pass 1/2: collecting SDOH updates by patient+encounter")

    sdoh_updates: dict[tuple[str, str], dict[str, int]] = {}

    update_query = f"""
    SELECT
      PatientDurableKey,
      EncounterKey,
      {qident(domain_col)} AS domain_value,
      {qident(answer_col)} AS answer_value
    FROM event_stage
    WHERE event_source = 'SDOH_RESPONSE'
    """

    cursor = conn.execute(update_query)
    total_sdoh_rows = 0
    start = time.monotonic()

    while True:
        rows = cursor.fetchmany(args.chunksize)
        if not rows:
            break

        for patient_id, encounter_key, domain_value, answer_value in rows:
            patient_id = str(patient_id or "").strip()
            encounter_key = str(encounter_key or "").strip()
            if not patient_id or not encounter_key:
                continue

            field = domain_to_field(domain_value)
            status_label = classify_sdoh_answer(answer_value)
            status_id = SDOH_STATUS_TO_ID[status_label]

            sdoh_updates.setdefault((patient_id, encounter_key), {})[field] = status_id

        total_sdoh_rows += len(rows)
        elapsed = time.monotonic() - start
        rate = total_sdoh_rows / max(elapsed, 1e-9)
        log(
            f"[ENCONLY] collected {total_sdoh_rows:,} SDOH rows; "
            f"{len(sdoh_updates):,} patient-encounter update keys; "
            f"{rate:,.0f} rows/sec"
        )

    log("[ENCONLY] pass 2/2: building encounter-only sequences")

    selected_cols = [
        "PatientDurableKey",
        "EncounterKey",
        "event_date",
        "event_time",
        "event_id",
        "event_index_within_encounter",
        "event_source",
    ]
    select_sql = ", ".join(qident(c) for c in selected_cols)

    stream_query = f"""
    SELECT {select_sql}
    FROM event_stage
    ORDER BY PatientDurableKey, event_date, event_time, EncounterKey, event_index_within_encounter, event_id
    """

    drop_fields = {
        # Remove grand token and SDOH-as-event/source streams.
        "event_token_ids",
        "event_source_ids",
        "sdoh_domain_ids",

        # Recompute encounter-only gap.
        "gap_ids",
    }

    # Non-list keys to copy into new sequence objects.
    static_keys_to_copy = [
        "patient_id",
        "patient_context_ids",
        "patient_context_raw",
        "patient_context_values",
    ]

    new_seq_by_patient: dict[str, dict[str, Any]] = {}
    latest_status_by_patient: dict[str, dict[str, int]] = {}
    last_encounter_date_by_patient: dict[str, Any] = {}
    encounter_index_by_patient: dict[str, int] = {}

    gap_to_id = artifact.get("gap_to_id", {v: i for i, v in enumerate(BASE_GAP_BINS)})
    for g in BASE_GAP_BINS:
        if g not in gap_to_id:
            gap_to_id[g] = len(gap_to_id)

    scanned = 0
    kept_encounters = 0
    removed_sdoh_events = 0
    skipped_patients = 0
    length_overflow = 0
    start = time.monotonic()

    cursor = conn.execute(stream_query)

    while True:
        rows = cursor.fetchmany(args.chunksize)
        if not rows:
            break

        for patient_id, encounter_key, event_date, event_time, event_id, event_index, event_source in rows:
            patient_id = str(patient_id or "").strip()
            encounter_key = str(encounter_key or "").strip()
            event_source = str(event_source or "").strip()

            old_seq = seq_by_patient.get(patient_id)
            if old_seq is None:
                skipped_patients += 1
                continue

            old_pos = old_pos_by_patient[patient_id]
            expected_len = expected_len_by_patient[patient_id]

            if old_pos >= expected_len:
                length_overflow += 1
                continue

            # Count every row because old sequence position was built over mixed event_stage.
            old_pos_by_patient[patient_id] = old_pos + 1

            if event_source == "SDOH_RESPONSE":
                removed_sdoh_events += 1
                continue

            if event_source != "ENCOUNTER":
                # Future-proof: only keep true encounter rows.
                continue

            if patient_id not in new_seq_by_patient:
                new_seq = {}
                for key in static_keys_to_copy:
                    if key in old_seq:
                        new_seq[key] = old_seq[key]
                new_seq["patient_id"] = patient_id

                for field in sdoh_status_fields:
                    new_seq[field] = []

                new_seq_by_patient[patient_id] = new_seq
                latest_status_by_patient[patient_id] = {
                    field: SDOH_STATUS_TO_ID["UNKNOWN_NOT_YET_MEASURED"]
                    for field in sdoh_status_fields
                }
                last_encounter_date_by_patient[patient_id] = None
                encounter_index_by_patient[patient_id] = 0

            new_seq = new_seq_by_patient[patient_id]

            # Apply same-encounter SDOH updates before appending this encounter.
            updates = sdoh_updates.get((patient_id, encounter_key))
            if updates:
                for field, status_id in updates.items():
                    if field not in latest_status_by_patient[patient_id]:
                        # Very rare: new domain not seen during distinct scan, but keep robust.
                        latest_status_by_patient[patient_id][field] = SDOH_STATUS_TO_ID["UNKNOWN_NOT_YET_MEASURED"]
                        new_seq[field] = [SDOH_STATUS_TO_ID["UNKNOWN_NOT_YET_MEASURED"]] * encounter_index_by_patient[patient_id]
                        if field not in sdoh_status_fields:
                            sdoh_status_fields.append(field)
                    latest_status_by_patient[patient_id][field] = int(status_id)

            # Copy all existing event-level streams except dropped/recomputed fields.
            copy_kept_event_fields(
                old_seq=old_seq,
                new_seq=new_seq,
                old_pos=old_pos,
                expected_len=expected_len,
                drop_fields=drop_fields,
            )

            encounter_index_by_patient[patient_id] += 1
            encounter_idx = encounter_index_by_patient[patient_id]

            current_date = parse_date(event_date)
            prev_date = last_encounter_date_by_patient[patient_id]
            if prev_date is None or current_date is None:
                days_since_previous = None
            else:
                days_since_previous = (current_date - prev_date).days

            g = gap_bin(days_since_previous, encounter_idx)
            new_seq.setdefault("gap_ids", []).append(int(gap_to_id.get(g, gap_to_id["UNKNOWN"])))

            for field in sdoh_status_fields:
                new_seq.setdefault(field, [])
                status = latest_status_by_patient[patient_id].get(
                    field,
                    SDOH_STATUS_TO_ID["UNKNOWN_NOT_YET_MEASURED"],
                )
                new_seq[field].append(int(status))

            if current_date is not None:
                last_encounter_date_by_patient[patient_id] = current_date

            kept_encounters += 1

        scanned += len(rows)
        elapsed = time.monotonic() - start
        rate = scanned / max(elapsed, 1e-9)
        log(
            f"[ENCONLY] scanned {scanned:,} event rows; "
            f"kept {kept_encounters:,} encounters; "
            f"removed {removed_sdoh_events:,} SDOH rows; "
            f"{rate:,.0f} rows/sec"
        )

    conn.close()

    log("[ENCONLY] filtering patients with enough encounters and validating lengths")

    new_sequences = []
    dropped_short = 0
    mismatch = 0

    event_level_fields_to_check = [
        "type_ids",
        "event_description_ids",
        "dept_specialty_ids",
        "setting_ids",
        "dept_type_ids",
        "facility_size_ids",
        "region_ids",
        "group_code_ids",
        "diagnosis_value_ids",
        "gap_ids",
    ] + sdoh_status_fields

    for patient_id, seq in new_seq_by_patient.items():
        n = len(seq.get("gap_ids", []))

        if n < args.min_encounters_per_patient:
            dropped_short += 1
            continue

        bad = False
        for field in event_level_fields_to_check:
            if field in seq and isinstance(seq[field], list) and len(seq[field]) != n:
                bad = True
                break

        if bad:
            mismatch += 1

            # Keep usable: truncate/pad with 0.
            for field in event_level_fields_to_check:
                if field not in seq:
                    continue
                vals = seq[field][:n]
                if len(vals) < n:
                    vals += [0] * (n - len(vals))
                seq[field] = vals

        # Ensure removed fields are absent.
        seq.pop("event_token_ids", None)
        seq.pop("event_source_ids", None)
        seq.pop("sdoh_domain_ids", None)

        new_sequences.append(seq)

    metadata = artifact.get("metadata", {})
    metadata.update(
        {
            "encounter_only": True,
            "sdoh_response_events_removed": True,
            "grand_event_token_removed": True,
            "event_source_ids_removed": True,
            "sdoh_domain_ids_removed": True,
            "gap_ids_recomputed_for_encounters_only": True,
            "sdoh_status_streams_added": True,
            "sdoh_status_to_id": SDOH_STATUS_TO_ID,
            "sdoh_status_fields": sdoh_status_fields,
            "n_sequences_before_encounter_only": int(len(old_sequences)),
            "n_sequences_after_encounter_only": int(len(new_sequences)),
            "n_encounters_kept": int(kept_encounters),
            "n_sdoh_response_events_removed": int(removed_sdoh_events),
            "n_patients_dropped_lt_min_encounters": int(dropped_short),
            "encounter_only_length_mismatch": int(mismatch),
            "encounter_only_length_overflow": int(length_overflow),
            "encounter_only_min_encounters_per_patient": int(args.min_encounters_per_patient),
            "sdoh_answer_status_classifier": "generic_text_heuristic_v1",
        }
    )

    # Remove old grand vocab if still present.
    artifact.pop("vocab", None)
    metadata.pop("event_vocab_size", None)

    artifact["sequences"] = new_sequences
    artifact["gap_to_id"] = gap_to_id
    artifact["sdoh_status_to_id"] = SDOH_STATUS_TO_ID
    artifact["metadata"] = metadata

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    log(f"[ENCONLY] saving encounter-only artifact: {args.output_pt}")
    torch.save(artifact, args.output_pt)

    vocab_artifact = {}
    if args.vocab_json.exists():
        vocab_artifact = json.loads(args.vocab_json.read_text(encoding="utf-8"))

    vocab_artifact.pop("event_vocab", None)
    vocab_artifact.pop("vocab", None)
    vocab_artifact["gap_to_id"] = gap_to_id
    vocab_artifact["sdoh_status_to_id"] = SDOH_STATUS_TO_ID
    vocab_artifact["sdoh_status_fields"] = sdoh_status_fields
    vocab_artifact["metadata"] = metadata

    log(f"[ENCONLY] saving encounter-only vocab json: {args.output_json}")
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