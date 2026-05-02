#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[UNK]": 3,
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

BASE_SETTING_BINS = [
    "ED",
    "INPATIENT",
    "HOSP_ADMIT",
    "HOSP_OP",
    "OBS",
    "OP_FACE",
    "NONE",
    "UNKNOWN",
]

BASE_DEPT_VOLUME_BINS = [
    "VERY_LOW",
    "LOW",
    "MID",
    "HIGH",
    "VERY_HIGH",
    "MISSING",
    "UNKNOWN",
]

EVENT_STAGE_COLUMNS = [
    "event_id",
    "PatientDurableKey",
    "EncounterKey",
    "event_date",
    "event_time",
    "event_index_within_encounter",
    "event_source",
    "event_grain",

    "PatientBirthYearBin",
    "SexAssignedAtBirth",
    "OmbRace",
    "OmbEthnicity",
    "MaritalStatus",
    "SmokingStatus",
    "VitalStatus",
    "MyChartStatus",
    "patient_geography_known_flag",
    "patient_block_population_bin",
    "sdoh_any_observed",
    "sdoh_num_questions_answered",
    "sdoh_num_domains_answered",
    "sdoh_domains_observed",

    "event_type",
    "event_subtype",
    "event_description",
    "event_domain",
    "Type",
    "VisitTypeDescription",
    "AdmissionSource",
    "AdmissionType",
    "primary_diagnosis_missing_class",
    "GroupCode",
    "DiagnosisValue",

    "DepartmentKey",
    "DepartmentType",
    "DepartmentSpecialty",
    "department_County",
    "department_City",
    "department_PostalCode",

    "provider_Type",
    "provider_PrimarySpecialty",
    "attending_provider_Type",
    "discharge_provider_Type",

    "IsEdVisit",
    "IsHospitalAdmission",
    "IsHospitalOutpatientVisit",
    "IsInpatientAdmission",
    "IsObservation",
    "IsOutpatientFaceToFaceVisit",
]


def log(message: str) -> None:
    print(message, flush=True)


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


def scalar(conn: sqlite3.Connection, sql: str) -> Any:
    row = conn.execute(sql).fetchone()
    return row[0] if row else None


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


def trueish(value: Any) -> bool:
    return normalize_value(value) in {"1", "TRUE", "T", "YES", "Y"}


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


def gap_bin(days: int | None, patient_event_index: int) -> str:
    if patient_event_index == 1:
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


def setting_bin(row: dict[str, Any]) -> str:
    if trueish(row.get("IsEdVisit")):
        return "ED"
    if trueish(row.get("IsInpatientAdmission")):
        return "INPATIENT"
    if trueish(row.get("IsHospitalAdmission")):
        return "HOSP_ADMIT"
    if trueish(row.get("IsObservation")):
        return "OBS"
    if trueish(row.get("IsHospitalOutpatientVisit")):
        return "HOSP_OP"
    if trueish(row.get("IsOutpatientFaceToFaceVisit")):
        return "OP_FACE"
    return "NONE"


def dept_key_norm(value: Any) -> str:
    out = normalize_value(value, max_len=80)
    return out if out != "UNKNOWN" else "MISSING"


def volume_bin(n: int | float | None) -> str:
    if n is None:
        return "UNKNOWN"
    if n < 100:
        return "VERY_LOW"
    if n < 1000:
        return "LOW"
    if n < 10000:
        return "MID"
    if n < 100000:
        return "HIGH"
    return "VERY_HIGH"


def region_label(row: dict[str, Any]) -> str:
    county = normalize_value(row.get("department_County"), max_len=60)
    if county != "UNKNOWN":
        return f"COUNTY_{county}"

    city = normalize_value(row.get("department_City"), max_len=60)
    if city != "UNKNOWN":
        return f"CITY_{city}"

    postal = normalize_value(row.get("department_PostalCode"), max_len=20)
    if postal != "UNKNOWN":
        return f"ZIP_{postal}"

    return "UNKNOWN"


def get_or_add(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def make_event_token(row: dict[str, Any], setting: str, dept_volume: str, gap: str) -> str:
    """
    Controlled composite token for sequence modeling.

    Avoids very high-cardinality identifiers like DepartmentKey/provider IDs.
    DiagnosisValue is intentionally excluded by default; GroupCode is used instead.
    """
    parts = [
        f"SRC={normalize_value(row.get('event_source'), 40)}",
        f"GRAIN={normalize_value(row.get('event_grain'), 40)}",
        f"TYPE={normalize_value(row.get('Type') or row.get('event_type'), 60)}",
        f"DESC={normalize_value(row.get('VisitTypeDescription') or row.get('event_description'), 70)}",
        f"DXG={normalize_value(row.get('GroupCode'), 40)}",
        f"SETTING={setting}",
        f"DEPT_TYPE={normalize_value(row.get('DepartmentType'), 50)}",
        f"DEPT_SPEC={normalize_value(row.get('DepartmentSpecialty'), 60)}",
        f"VOL={dept_volume}",
        f"GAP={gap}",
        f"SDOH={normalize_value(row.get('event_domain'), 50)}",
    ]

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            out.append(part)

    return "EVENT_COMPOSITE::" + "|".join(out)


def build_department_counts(conn: sqlite3.Connection) -> dict[str, int]:
    log("[SEQ] computing department event counts from event_stage")
    counts: dict[str, int] = {}
    query = """
    SELECT
      COALESCE(NULLIF(TRIM(DepartmentKey), ''), 'MISSING') AS dept,
      COUNT(*) AS n
    FROM event_stage
    GROUP BY COALESCE(NULLIF(TRIM(DepartmentKey), ''), 'MISSING')
    """
    for dept, n in conn.execute(query):
        counts[dept_key_norm(dept)] = int(n)
    log(f"[SEQ] department count entries: {len(counts):,}")
    return counts


def flush_patient_sequence(
    sequences: list[dict[str, Any]],
    current: dict[str, Any] | None,
    min_events_per_patient: int,
) -> None:
    if current is None:
        return
    if len(current["event_token_ids"]) >= min_events_per_patient:
        sequences.append(current)


def write_vocab_json(
    output_path: Path,
    *,
    event_vocab: dict[str, int],
    gap_to_id: dict[str, int],
    setting_to_id: dict[str, int],
    dept_type_to_id: dict[str, int],
    facility_size_to_id: dict[str, int],
    region_to_id: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    artifact = {
        "special_tokens": SPECIAL_TOKENS,
        "event_vocab": event_vocab,
        "gap_to_id": gap_to_id,
        "setting_to_id": setting_to_id,
        "dept_type_to_id": dept_type_to_id,
        "facility_size_to_id": facility_size_to_id,
        "region_to_id": region_to_id,
        "metadata": metadata,
    }
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact patient sequence model inputs from existing event_stage."
    )
    parser.add_argument("--work-db", type=Path, default=Path("data/interim/event_tokenization_v2.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sequence_model"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")
    parser.add_argument("--smoke-rowid-max", type=int, default=None, help="Restrict event_stage source rows for fast smoke test.")
    parser.add_argument("--min-events-per-patient", type=int, default=2)
    parser.add_argument("--no-event-csv", action="store_true", help="Skip writing sequence_model_events.csv.gz.")
    args = parser.parse_args()

    log("[SEQ] script started")
    log(f"[SEQ] work_db={args.work_db}")
    log(f"[SEQ] output_dir={args.output_dir}")

    if not args.work_db.exists():
        raise SystemExit(f"work DB not found: {args.work_db}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_csv_path = args.output_dir / "sequence_model_events.csv.gz"
    pt_path = args.output_dir / "patient_sequences_for_model.pt"
    vocab_path = args.output_dir / "sequence_model_vocab.json"

    if event_csv_path.exists():
        event_csv_path.unlink()
    if pt_path.exists():
        pt_path.unlink()
    if vocab_path.exists():
        vocab_path.unlink()

    conn = sqlite3.connect(args.work_db, timeout=120)
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA busy_timeout=120000;")

    if not table_exists(conn, "event_stage"):
        raise SystemExit("event_stage not found in work DB.")

    n_stage = scalar(conn, "SELECT COUNT(*) FROM event_stage")
    log(f"[SEQ] event_stage rows: {n_stage:,}")

    available = get_columns(conn, "event_stage")
    required = {
        "event_id",
        "PatientDurableKey",
        "EncounterKey",
        "event_date",
        "event_time",
        "event_index_within_encounter",
    }
    missing_required = sorted(required - available)
    if missing_required:
        raise SystemExit(f"event_stage missing required columns: {missing_required}")

    selected = [c for c in EVENT_STAGE_COLUMNS if c in available]
    missing_optional = [c for c in EVENT_STAGE_COLUMNS if c not in available]
    if missing_optional:
        log(f"[SEQ] optional columns skipped: {missing_optional}")

    dept_counts = build_department_counts(conn)

    event_vocab = dict(SPECIAL_TOKENS)
    gap_to_id = {v: i for i, v in enumerate(BASE_GAP_BINS)}
    setting_to_id = {v: i for i, v in enumerate(BASE_SETTING_BINS)}
    dept_type_to_id = {"UNKNOWN": 0}
    facility_size_to_id = {v: i for i, v in enumerate(BASE_DEPT_VOLUME_BINS)}
    region_to_id = {"UNKNOWN": 0}

    select_sql = ", ".join(qident(c) for c in selected)

    source_where = ""
    if args.smoke_rowid_max:
        source_where = f"WHERE rowid <= {args.smoke_rowid_max}"

    limit_sql = f"LIMIT {args.limit}" if args.limit else ""

    query = f"""
    SELECT {select_sql}
    FROM event_stage
    {source_where}
    ORDER BY PatientDurableKey, event_date, event_time, EncounterKey, event_index_within_encounter, event_id
    {limit_sql}
    """

    log("[SEQ] streaming ordered events from SQLite")
    if args.smoke_rowid_max:
        log(f"[SEQ] smoke source cap active: rowid <= {args.smoke_rowid_max:,}")
    if args.limit:
        log(f"[SEQ] output row limit active: {args.limit:,}")

    sequences: list[dict[str, Any]] = []
    current_patient_id: str | None = None
    current_sequence: dict[str, Any] | None = None
    last_event_date = None
    patient_event_index = 0

    total_events = 0
    total_patients_seen = 0
    start = time.monotonic()

    event_csv_file = None
    event_writer = None

    try:
        if not args.no_event_csv:
            event_csv_file = gzip.open(event_csv_path, "wt", encoding="utf-8", newline="")
            event_writer = csv.DictWriter(
                event_csv_file,
                fieldnames=[
                    "PatientDurableKey",
                    "event_id",
                    "EncounterKey",
                    "event_date",
                    "patient_event_index",
                    "event_token_id",
                    "event_token",
                    "gap_id",
                    "gap_bin",
                    "setting_id",
                    "setting_bin",
                    "dept_type_id",
                    "dept_type",
                    "facility_size_id",
                    "facility_size_bin",
                    "region_id",
                    "region_label",
                    "DepartmentKey",
                    "GroupCode",
                    "DiagnosisValue",
                ],
            )
            event_writer.writeheader()

        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(args.chunksize)
            if not rows:
                break

            for tup in rows:
                row = dict(zip(columns, tup))

                patient_id = str(row.get("PatientDurableKey") or "").strip()
                if not patient_id:
                    continue

                if patient_id != current_patient_id:
                    flush_patient_sequence(sequences, current_sequence, args.min_events_per_patient)
                    current_patient_id = patient_id
                    total_patients_seen += 1
                    patient_event_index = 0
                    last_event_date = None
                    current_sequence = {
                        "patient_id": patient_id,
                        "event_token_ids": [],
                        "gap_ids": [],
                        "setting_ids": [],
                        "dept_type_ids": [],
                        "facility_size_ids": [],
                        "region_ids": [],
                    }

                patient_event_index += 1

                current_date = parse_date(row.get("event_date"))
                if last_event_date is None or current_date is None:
                    days_since_previous = None
                else:
                    days_since_previous = (current_date - last_event_date).days

                gap = gap_bin(days_since_previous, patient_event_index)
                setting = setting_bin(row)

                dept_key = dept_key_norm(row.get("DepartmentKey"))
                dept_volume = volume_bin(dept_counts.get(dept_key))
                dept_type = normalize_value(row.get("DepartmentType"), max_len=60)
                region = region_label(row)

                event_token = make_event_token(row, setting, dept_volume, gap)

                event_token_id = get_or_add(event_vocab, event_token)
                gap_id = get_or_add(gap_to_id, gap)
                setting_id = get_or_add(setting_to_id, setting)
                dept_type_id = get_or_add(dept_type_to_id, dept_type)
                facility_size_id = get_or_add(facility_size_to_id, dept_volume)
                region_id = get_or_add(region_to_id, region)

                assert current_sequence is not None
                current_sequence["event_token_ids"].append(int(event_token_id))
                current_sequence["gap_ids"].append(int(gap_id))
                current_sequence["setting_ids"].append(int(setting_id))
                current_sequence["dept_type_ids"].append(int(dept_type_id))
                current_sequence["facility_size_ids"].append(int(facility_size_id))
                current_sequence["region_ids"].append(int(region_id))

                if event_writer is not None:
                    event_writer.writerow(
                        {
                            "PatientDurableKey": patient_id,
                            "event_id": row.get("event_id"),
                            "EncounterKey": row.get("EncounterKey"),
                            "event_date": row.get("event_date"),
                            "patient_event_index": patient_event_index,
                            "event_token_id": event_token_id,
                            "event_token": event_token,
                            "gap_id": gap_id,
                            "gap_bin": gap,
                            "setting_id": setting_id,
                            "setting_bin": setting,
                            "dept_type_id": dept_type_id,
                            "dept_type": dept_type,
                            "facility_size_id": facility_size_id,
                            "facility_size_bin": dept_volume,
                            "region_id": region_id,
                            "region_label": region,
                            "DepartmentKey": row.get("DepartmentKey"),
                            "GroupCode": row.get("GroupCode"),
                            "DiagnosisValue": row.get("DiagnosisValue"),
                        }
                    )

                last_event_date = current_date if current_date is not None else last_event_date
                total_events += 1

            elapsed = time.monotonic() - start
            rate = total_events / max(elapsed, 1e-9)
            log(
                f"[SEQ] processed {total_events:,} events, "
                f"{total_patients_seen:,} patients seen, "
                f"{len(sequences):,} completed sequences, "
                f"{rate:,.0f} events/sec"
            )

    finally:
        if event_csv_file is not None:
            event_csv_file.close()
        conn.close()

    flush_patient_sequence(sequences, current_sequence, args.min_events_per_patient)

    metadata = {
        "n_events_processed": int(total_events),
        "n_patients_seen": int(total_patients_seen),
        "n_training_sequences": int(len(sequences)),
        "event_vocab_size": int(len(event_vocab)),
        "gap_classes": int(len(gap_to_id)),
        "setting_classes": int(len(setting_to_id)),
        "dept_type_classes": int(len(dept_type_to_id)),
        "facility_size_classes": int(len(facility_size_to_id)),
        "region_classes": int(len(region_to_id)),
        "source": str(args.work_db),
        "source_table": "event_stage",
    }

    log(f"[SEQ] saving sequences: {pt_path}")
    torch.save(
        {
            "sequences": sequences,
            "vocab": event_vocab,
            "gap_to_id": gap_to_id,
            "setting_to_id": setting_to_id,
            "dept_type_to_id": dept_type_to_id,
            "facility_size_to_id": facility_size_to_id,
            "region_to_id": region_to_id,
            "metadata": metadata,
        },
        pt_path,
    )

    log(f"[SEQ] saving vocab/metadata: {vocab_path}")
    write_vocab_json(
        vocab_path,
        event_vocab=event_vocab,
        gap_to_id=gap_to_id,
        setting_to_id=setting_to_id,
        dept_type_to_id=dept_type_to_id,
        facility_size_to_id=facility_size_to_id,
        region_to_id=region_to_id,
        metadata=metadata,
    )

    if not args.no_event_csv:
        log(f"[SEQ] event-level token CSV: {event_csv_path}")

    log(json.dumps({"status": "done", **metadata}, indent=2))


if __name__ == "__main__":
    main()