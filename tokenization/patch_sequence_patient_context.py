#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch


PATIENT_CONTEXT_CATEGORICAL_COLUMNS = [
    "PatientBirthYearBin",
    "SexAssignedAtBirth",
    "FirstRace",
    "OmbRace",
    "OmbEthnicity",
    "MaritalStatus",
    "SmokingStatus",
    "VitalStatus",
    "MyChartStatus",
    "SexualOrientation",
    "patient_geography_known_flag",
    "patient_birth_year_bin_missing_class",
    "mychart_status_missing_class",
    "smoking_status_missing_class",
    "patient_block_population_bin",
    "sdoh_any_observed",
    "sdoh_num_questions_answered",
    "sdoh_num_domains_answered",
    "sdoh_domains_observed",
    "sdoh_AlcoholUse_observed",
    "sdoh_Depression_observed",
    "sdoh_FinancialResourceStrain_observed",
    "sdoh_FoodInsecurity_observed",
    "sdoh_HousingStability_observed",
    "sdoh_IntimatePartnerViolence_observed",
    "sdoh_PhysicalActivity_observed",
    "sdoh_SocialConnections_observed",
    "sdoh_Stress_observed",
    "sdoh_TransportationNeeds_observed",
    "sdoh_Utilities_observed",
]

PATIENT_CONTEXT_NUMERIC_COLUMNS = [
    "PopulationValue",
    "CENTLAT",
    "CENTLON",
    "CENTLONG",
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


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    s = str(value).strip().replace(",", "")
    if not s or s.upper() in {"NA", "NAN", "NULL", "NONE", "UNKNOWN"}:
        return None

    try:
        return float(s)
    except Exception:
        return None


def get_or_add(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def bin_count(value: Any) -> str:
    x = parse_float(value)
    if x is None:
        return "UNKNOWN"

    n = int(x)

    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"

    return "3PLUS"


def normalize_context_value(column: str, value: Any) -> str:
    if column in {"sdoh_num_questions_answered", "sdoh_num_domains_answered"}:
        return bin_count(value)

    return normalize_value(value)


def build_patient_context_lookup(
    conn: sqlite3.Connection,
    available: set[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> dict[str, dict[str, Any]]:
    selected = ["PatientDurableKey"]

    for col in categorical_cols + numeric_cols:
        if col in available and col not in selected:
            selected.append(col)

    select_sql = ", ".join(qident(c) for c in selected)

    # event_stage has patient-level fields repeated across events.
    # GROUP BY PatientDurableKey gives one representative row per patient.
    query = f"""
    SELECT {select_sql}
    FROM event_stage
    GROUP BY PatientDurableKey
    """

    log("[CTX] reading one context row per patient from event_stage")

    lookup: dict[str, dict[str, Any]] = {}

    cursor = conn.execute(query)
    columns = [d[0] for d in cursor.description]

    n = 0

    while True:
        rows = cursor.fetchmany(200_000)

        if not rows:
            break

        for tup in rows:
            row = dict(zip(columns, tup))
            patient_id = str(row.get("PatientDurableKey") or "").strip()

            if not patient_id:
                continue

            lookup[patient_id] = row
            n += 1

        log(f"[CTX] loaded context for {n:,} patients")

    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch patient profile IDs and patient lat/lon into sequence model artifact."
    )

    parser.add_argument(
        "--sequence-pt",
        type=Path,
        default=Path("data/processed/sequence_model/patient_sequences_for_model_with_diagnosis.pt"),
    )
    parser.add_argument(
        "--work-db",
        type=Path,
        default=Path("data/interim/event_tokenization_v2.sqlite"),
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=Path("data/processed/sequence_model/sequence_model_vocab_with_diagnosis.json"),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path("data/processed/sequence_model/patient_sequences_for_model_with_diagnosis_and_context.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/sequence_model/sequence_model_vocab_with_diagnosis_and_context.json"),
    )

    args = parser.parse_args()

    if not args.sequence_pt.exists():
        raise SystemExit(f"Missing sequence artifact: {args.sequence_pt}")

    if not args.work_db.exists():
        raise SystemExit(f"Missing work DB: {args.work_db}")

    log(f"[CTX] loading sequence artifact: {args.sequence_pt}")
    artifact = torch.load(args.sequence_pt, map_location="cpu")

    sequences = artifact["sequences"]
    log(f"[CTX] loaded sequences: {len(sequences):,}")

    conn = sqlite3.connect(args.work_db, timeout=120)
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA busy_timeout=120000;")

    if not table_exists(conn, "event_stage"):
        raise SystemExit("event_stage not found in work DB.")

    available = get_columns(conn, "event_stage")

    categorical_cols = [
        col for col in PATIENT_CONTEXT_CATEGORICAL_COLUMNS
        if col in available
    ]

    numeric_cols = [
        col for col in PATIENT_CONTEXT_NUMERIC_COLUMNS
        if col in available
    ]

    log(f"[CTX] categorical context columns found: {categorical_cols}")
    log(f"[CTX] numeric context columns found: {numeric_cols}")

    if "CENTLAT" not in numeric_cols:
        log("[CTX] warning: CENTLAT not found; patient_lat will be None.")

    if "CENTLON" not in numeric_cols and "CENTLONG" not in numeric_cols:
        log("[CTX] warning: neither CENTLON nor CENTLONG found; patient_lon will be None.")

    context_lookup = build_patient_context_lookup(
        conn=conn,
        available=available,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    conn.close()

    # One mapping per feature. This means one separate embedding/dimension family
    # per patient profile feature, not one giant concatenated token.
    patient_context_to_id: dict[str, dict[str, int]] = {
        col: {"UNKNOWN": 0}
        for col in categorical_cols
    }

    patched = 0
    missing = 0
    lat_nonnull = 0
    lon_nonnull = 0
    pop_nonnull = 0

    log("[CTX] attaching context to patient sequences")

    for seq in sequences:
        patient_id = str(seq.get("patient_id") or "").strip()
        row = context_lookup.get(patient_id)

        if row is None:
            seq["patient_context_ids"] = {
                col: 0
                for col in categorical_cols
            }
            seq["patient_context_raw"] = {
                col: "UNKNOWN"
                for col in categorical_cols
            }
            seq["patient_context_values"] = {
                "patient_lat": None,
                "patient_lon": None,
                "patient_population": None,
            }
            missing += 1
            continue

        context_ids: dict[str, int] = {}
        context_raw: dict[str, str] = {}

        for col in categorical_cols:
            norm = normalize_context_value(col, row.get(col))
            context_raw[col] = norm
            context_ids[col] = int(get_or_add(patient_context_to_id[col], norm))

        patient_lat = parse_float(row.get("CENTLAT"))

        if "CENTLON" in row:
            patient_lon = parse_float(row.get("CENTLON"))
        elif "CENTLONG" in row:
            patient_lon = parse_float(row.get("CENTLONG"))
        else:
            patient_lon = None

        patient_population = parse_float(row.get("PopulationValue"))

        if patient_lat is not None:
            lat_nonnull += 1
        if patient_lon is not None:
            lon_nonnull += 1
        if patient_population is not None:
            pop_nonnull += 1

        seq["patient_context_ids"] = context_ids
        seq["patient_context_raw"] = context_raw
        seq["patient_context_values"] = {
            "patient_lat": patient_lat,
            "patient_lon": patient_lon,
            "patient_population": patient_population,
        }

        patched += 1

    metadata = artifact.get("metadata", {})

    metadata.update(
        {
            "has_patient_context_ids": True,
            "has_patient_context_raw": True,
            "has_patient_context_values": True,
            "patient_context_sequences_patched": int(patched),
            "patient_context_sequences_missing": int(missing),
            "patient_context_categorical_fields": categorical_cols,
            "patient_context_numeric_fields_found": numeric_cols,
            "patient_context_class_counts": {
                col: int(len(mapping))
                for col, mapping in patient_context_to_id.items()
            },
            "patient_lat_source": "CENTLAT" if "CENTLAT" in numeric_cols else None,
            "patient_lon_source": (
                "CENTLON"
                if "CENTLON" in numeric_cols
                else ("CENTLONG" if "CENTLONG" in numeric_cols else None)
            ),
            "patient_population_source": "PopulationValue" if "PopulationValue" in numeric_cols else None,
            "patient_lat_nonnull_sequences": int(lat_nonnull),
            "patient_lon_nonnull_sequences": int(lon_nonnull),
            "patient_population_nonnull_sequences": int(pop_nonnull),
        }
    )

    artifact["sequences"] = sequences
    artifact["patient_context_to_id"] = patient_context_to_id
    artifact["metadata"] = metadata

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    log(f"[CTX] saving patched artifact: {args.output_pt}")
    torch.save(artifact, args.output_pt)

    vocab_artifact = {}

    if args.vocab_json.exists():
        vocab_artifact = json.loads(args.vocab_json.read_text(encoding="utf-8"))

    vocab_artifact["patient_context_to_id"] = patient_context_to_id
    vocab_artifact["metadata"] = metadata

    log(f"[CTX] saving patched vocab json: {args.output_json}")
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